import io
import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import colorsys

# python precipitates/create_precipitate_dataset_real.py --source_dir "path/to/exported_good_data" --output_path "my_real_precipitates.parquet"

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RANDOM_SEED = 42
TEST_RATIO = 0.10


# ---------------------------------
# Improved HSV-based color estimation
# ---------------------------------
def estimate_precipitate_color(tile: Image.Image, mask: Image.Image) -> str:
    """
    Robust coarse color estimation using HSV.
    Returns: reddish | violet | gray | unknown
    """
    tile_np = np.array(tile)
    mask_np = np.array(mask.convert("L")) > 0

    if mask_np.sum() < 50:
        return "unknown"

    region = tile_np[mask_np] / 255.0
    r, g, b = np.median(region, axis=0)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Gray: very low saturation
    if s < 0.12:
        return "gray"

    # Hue is [0,1]
    hue_deg = h * 360

    # Red / pink
    if hue_deg < 25 or hue_deg > 330:
        return "reddish"

    # Violet / indigo / purple
    if 240 <= hue_deg <= 310:
        return "violet"

    return "unknown"


# ---------------------------------
# Text description
# ---------------------------------
def generate_precipitate_description(mask_array: np.ndarray, color: str) -> str:
    if np.count_nonzero(mask_array) == 0:
        return (
            "Histopathological image of lung tissue, "
            "hematoxylin and eosin staining, "
            "clean tissue with no precipitate artifact."
        )

    base = [
        "Histopathological image of lung tissue with a precipitate artifact.",
        "H&E stained lung tissue containing a precipitate artifact.",
        "Microscopy slide of lung tissue with a large precipitate artifact.",
        "Lung tissue sample with histopathological precipitate artifact.",
    ]

    desc = random.choice(base)

    if color != "unknown":
        desc = desc.replace("precipitate artifact", f"{color} precipitate artifact")

    return desc


# ---------------------------------
# Dataset creator
# ---------------------------------
def create_precipitate_parquet(source_dir: str, output_path: str):
    random.seed(RANDOM_SEED)

    tiles_dir = os.path.join(source_dir, "tiles")
    masks_dir = os.path.join(source_dir, "masks")

    records = []
    tile_files = sorted([f for f in os.listdir(tiles_dir) if f.endswith(".png")])

    # Shuffle once for random split
    random.shuffle(tile_files)

    n_test = int(len(tile_files) * TEST_RATIO)
    test_set = set(tile_files[:n_test])

    logging.info(f"Assigning {n_test}/{len(tile_files)} samples to test split")

    for filename in tqdm(tile_files, desc="Processing precipitate tiles"):
        split = "test" if filename in test_set else "train"

        tile_path = os.path.join(tiles_dir, filename)
        base, ext = os.path.splitext(filename)
        mask_path = os.path.join(masks_dir, f"{base}_primary_mask{ext}")

        if not os.path.exists(mask_path):
            logging.warning(f"Mask missing for {filename}, skipping.")
            continue

        try:
            tile = Image.open(tile_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            color = estimate_precipitate_color(tile, mask)
            mask_np = np.array(mask) > 0
            text = generate_precipitate_description(mask_np, color)

            tile_buf = io.BytesIO()
            mask_buf = io.BytesIO()
            tile.save(tile_buf, format="PNG")
            mask.save(mask_buf, format="PNG")

            records.append(
                {
                    "image": tile_buf.getvalue(),
                    "mask": mask_buf.getvalue(),
                    "text": text,
                    "artifact": "precipitate",
                    "source": "real",
                    "color": color,
                    "filename": filename,
                    "split": split,
                }
            )

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    logging.info(f"Saved {len(df)} samples to {output_path}")
    logging.info("Color distribution:")
    logging.info(df["color"].value_counts())
    logging.info("Split distribution:")
    logging.info(df["split"].value_counts())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract real precipitates into a Parquet dataset.")
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing 'tiles' and 'masks' subfolders."
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="lung_precipitate_real.parquet", 
        help="Path for the output Parquet file."
    )
    
    args = parser.parse_args()

    create_precipitate_parquet(args.source_dir, args.output_path)