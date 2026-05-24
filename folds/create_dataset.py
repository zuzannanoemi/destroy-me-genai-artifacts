import os
import io
import argparse
import logging
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_image_description_simple(mask_array: np.ndarray) -> str:
    """
    Generates a simplified text description focusing on tissue type and artefact identity.
    Ignores position and size as those are defined by the binary mask.
    """
    white_pixels = np.count_nonzero(mask_array)

    # If mask is empty, it's a clean image
    if white_pixels == 0:
        return "Histopathological image of lung tissue, hematoxylin and eosin staining, clean tissue with no artifacts."

    # If mask exists, describe the TEXTURE we want to create
    # Use synonyms to help the model generalize
    descriptions = [
        "Histopathological image of lung tissue with a tissue fold artifact.",
        "H&E stained lung tissue containing a folding artifact.",
        "Microscopy slide of lung tissue with overlapping tissue fold artifact.",
        "Lung tissue sample with a histopathology fold artifact.",
    ]

    # Return a random choice to add slight variety to the text embedding
    import random

    return random.choice(descriptions)


# --- 1. The Description Generator ---
def generate_image_description(mask_array: np.ndarray) -> str:
    """
    Generates a text description for an image based on its mask's size and position.
    """
    white_pixels = np.count_nonzero(mask_array)
    total_pixels = mask_array.size

    if white_pixels == 0:
        return "Histopathological image of lung tissue, hematoxylin and eosin staining, with no fold artifact visible."

    white_ratio = white_pixels / total_pixels

    # Case: Huge mask (> 80%)
    if white_ratio > 0.80:
        return (
            "Histopathological image of lung tissue, hematoxylin and eosin staining, "
            "with a fold artifact covering most of the image."
        )

    # Size logic
    if white_ratio < 0.20:
        size_desc = "a small part"
    elif white_ratio < 0.50:
        size_desc = "a moderate part"
    else:
        size_desc = "a large part"

    # Position logic
    try:
        cy, cx = center_of_mass(mask_array)
        height, width = mask_array.shape

        # Vertical (Y) - 0 is top
        if cy < height * 0.33:
            v_pos = "upper"
        elif cy > height * 0.66:
            v_pos = "lower"
        else:
            v_pos = "center"

        # Horizontal (X)
        if cx < width * 0.33:
            h_pos = "left"
        elif cx > width * 0.66:
            h_pos = "right"
        else:
            h_pos = "center"

        # Combine
        if v_pos == "center" and h_pos == "center":
            pos_desc = "in the center"
        elif v_pos == "center":
            pos_desc = f"on the {h_pos}"
        elif h_pos == "center":
            pos_desc = f"in the {v_pos} part"
        else:
            pos_desc = f"in the {v_pos} {h_pos}"

    except ValueError:
        pos_desc = "in an indeterminate position"

    return (
        "Histopathological image of lung tissue, hematoxylin and eosin staining, "
        f"with a fold artifact visible {pos_desc} covering {size_desc} of the image."
    )


# --- 2. The Parquet Creator ---
def create_dataset_parquet(source_dir: str, output_path: str):
    """
    Reads tile and mask patches, processes them, and saves to a Parquet file.
    """
    source_tiles_path = os.path.join(source_dir, "tiles")
    source_masks_path = os.path.join(source_dir, "masks")

    if not os.path.isdir(source_tiles_path):
        logging.error(f"Source directory for tiles not found: '{source_tiles_path}'")
        return

    data_records = []

    # Get all PNG files
    tile_filenames = sorted(
        [f for f in os.listdir(source_tiles_path) if f.endswith(".png")]
    )

    for filename in tqdm(tile_filenames, desc="Processing Images"):
        # --- Split Logic ---
        # Adjust this logic if your naming convention for test set differs
        split = "test" if "TCGA-38" in filename else "train"

        tile_path = os.path.join(source_tiles_path, filename)

        # --- Mask Name Logic ---
        # Tile: image.png -> Mask: image_primary_mask.png
        base_name, ext = os.path.splitext(filename)
        mask_filename = f"{base_name}_primary_mask{ext}"
        mask_path = os.path.join(source_masks_path, mask_filename)

        if not os.path.exists(mask_path):
            logging.warning(f"Mask not found: {mask_filename}. Skipping.")
            continue

        try:
            with Image.open(tile_path) as tile, Image.open(mask_path) as mask:
                # A. Convert images to bytes (No dictionary wrapper!)
                tile_buffer = io.BytesIO()
                mask_buffer = io.BytesIO()

                tile.save(tile_buffer, format="PNG")
                mask.save(mask_buffer, format="PNG")

                tile_bytes = tile_buffer.getvalue()
                mask_bytes = mask_buffer.getvalue()

                # B. Generate text description
                # Convert to grayscale numpy array
                mask_gray = np.array(mask.convert("L"))
                # Ensure binary (0 or 1) for accurate calculation
                mask_binary = np.where(mask_gray > 0, 1, 0)

                text_description = generate_image_description_simple(mask_binary)

                # C. Append flat record
                data_records.append(
                    {
                        "image": tile_bytes,  # Raw bytes
                        "mask": mask_bytes,  # Raw bytes
                        "text": text_description,  # String
                        "filename": filename,  # String (Metadata)
                        "split": split,  # String
                    }
                )

        except Exception as e:
            logging.error(f"Could not process file pair '{filename}'. Error: {e}")

    if not data_records:
        logging.error("No data was processed. The Parquet file will not be created.")
        return

    logging.info(f"Creating DataFrame with {len(data_records)} records...")
    df = pd.DataFrame(data_records)

    logging.info(f"Saving DataFrame to Parquet file at '{output_path}'...")
    df.to_parquet(output_path, engine="pyarrow", index=False)

    logging.info("--- Parquet file created successfully! ---")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Parquet dataset for fold artifacts.")
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="Path to the folder containing 'tiles' and 'masks' subfolders."
    )
    
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="lung_folds_dataset_simple_desc_MPP_09.parquet", 
        help="Path/filename for the output Parquet file."
    )
    
    args = parser.parse_args()

    create_dataset_parquet(args.source_dir, args.output_path)
