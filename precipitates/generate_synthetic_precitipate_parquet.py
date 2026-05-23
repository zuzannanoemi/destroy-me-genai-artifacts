import io
import logging
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from precitipate_artifact_synth_advanced import HistologyArtifactGenerator

# ---------------------------------
# CONFIG
# ---------------------------------
SOURCE_CLEAN_PARQUET = "lung_clean_for_synthetic.parquet"
OUTPUT_PARQUET = "lung_synthetic_precipitates_MPP_09.parquet"

BATCH_SIZE = 128

# Probability of generating synthetic precipitate per clean image
# 0.5 → ~50% of clean images become synthetic
SYNTH_PROB = 1.0

# ---------------------------------
# LOGGING
# ---------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------
# HELPERS
# ---------------------------------
def assign_split(filename: str) -> str:
    return "test" if hash(filename) % 10 == 0 else "train"


def binarize_mask(mask: Image.Image, threshold=16):
    # arr = np.array(mask)
    arr = np.array(mask.convert("L"))
    return Image.fromarray((arr > threshold).astype(np.uint8) * 255)


def precipitate_prompt():
    return random.choice(
        [
            "Histopathological image of lung tissue with a precipitate artifact.",
            "H&E stained lung tissue containing a precipitate artifact.",
            "Lung tissue sample with a histopathology precipitate artifact.",
            "Microscopy image of lung tissue showing a precipitate artifact.",
        ]
    )


# ---------------------------------
# MAIN (STREAMING)
# ---------------------------------
def generate_synthetic_precipitates_streaming():
    parquet_file = pq.ParquetFile(SOURCE_CLEAN_PARQUET)

    writer = None
    seen = 0
    written = 0
    split_counts = {"train": 0, "test": 0}
    generator = HistologyArtifactGenerator(size=756)

    logging.info(f"Streaming clean data from {SOURCE_CLEAN_PARQUET}")

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        total=parquet_file.metadata.num_rows // BATCH_SIZE + 1,
        desc="Generating synthetic precipitates",
    ):
        rows = batch.to_pylist()
        out_rows = []

        for row in rows:
            seen += 1

            if random.random() > SYNTH_PROB:
                continue

            img = Image.open(io.BytesIO(row["image"])).convert("RGB")

            # out_img, out_mask = apply_precipitate_artifact(img)
            out_img, out_mask = generator.apply_random_artifact(img)

            # ensure binary mask for training
            out_mask = binarize_mask(out_mask)

            if seen < 10:
                out_img.save(f"synthetic/synthetic_artifact_{seen}.png")
                out_mask.save(f"synthetic/synthetic_mask_{seen}.png")

            img_buf = io.BytesIO()
            mask_buf = io.BytesIO()

            out_img.save(img_buf, format="PNG")
            out_mask.save(mask_buf, format="PNG")

            filename = row.get("filename", f"synthetic_{seen}.png")
            split = assign_split(filename)

            split_counts[split] += 1

            out_rows.append(
                {
                    "image": img_buf.getvalue(),
                    "mask": mask_buf.getvalue(),
                    "text": precipitate_prompt(),
                    "artifact": "precipitate",
                    "source": "synthetic",
                    "filename": filename,
                    "split": split,
                }
            )

        if not out_rows:
            continue

        out_table = pa.Table.from_pylist(out_rows)

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_PARQUET,
                out_table.schema,
                compression="snappy",
            )

        writer.write_table(out_table)
        written += len(out_rows)

    if writer:
        writer.close()

    logging.info("=== SYNTHETIC PRECIPITATE GENERATION DONE ===")
    logging.info(f"Seen clean images: {seen}")
    logging.info(f"Generated synthetic: {written}")
    logging.info(f"Train: {split_counts['train']}")
    logging.info(f"Test:  {split_counts['test']}")


# ---------------------------------
# ENTRY
# ---------------------------------
if __name__ == "__main__":
    generate_synthetic_precipitates_streaming()
