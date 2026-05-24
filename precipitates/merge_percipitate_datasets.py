import logging

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REAL_PRECIPITATE_PARQUET = "lung_precipitate_real_09_MPP.parquet"
SYNTH_PRECIPITATE_PARQUET = "lung_synthetic_precipitates_MPP_09.parquet"
CLEAN_ONLY_PARQUET = "lung_clean_only_subset.parquet"

OUTPUT_PARQUET = "lung_precipitate_training_dataset_MPP_09_384px.parquet"

BATCH_SIZE = 512

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# CANONICAL SCHEMA
# --------------------------------------------------
CANONICAL_SCHEMA = pa.schema(
    [
        ("image", pa.binary()),
        ("mask", pa.binary()),
        ("text", pa.string()),
        ("artifact", pa.string()),
        ("source", pa.string()),
        ("color", pa.string()),
        ("filename", pa.string()),
        ("split", pa.string()),
    ]
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def stream_copy(
    parquet_path,
    writer,
    allow_test,
    source_name,
    stats,
):
    pf = pq.ParquetFile(parquet_path)

    logging.info(f"Streaming {parquet_path}")

    for batch in tqdm(
        pf.iter_batches(batch_size=BATCH_SIZE),
        total=pf.metadata.num_rows // BATCH_SIZE + 1,
        desc=f"Copying {source_name}",
    ):
        rows = batch.to_pylist()
        out_rows = []

        for row in rows:
            split = row.get("split", "train")

            if split == "test" and not allow_test:
                continue

            normalized = {}

            for field in CANONICAL_SCHEMA.names:
                if field == "source":
                    normalized[field] = source_name
                else:
                    normalized[field] = row.get(field)

            if normalized["color"] is None:
                normalized["color"] = "unknown"

            out_rows.append(normalized)

            stats[f"{source_name}_{split}"] = stats.get(f"{source_name}_{split}", 0) + 1

        if not out_rows:
            continue

        table = pa.Table.from_pylist(out_rows, schema=CANONICAL_SCHEMA)

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_PARQUET,
                CANONICAL_SCHEMA,
                compression="snappy",
            )

        writer.write_table(table)

    return writer


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def merge_datasets_streaming():
    writer = None
    stats = {}

    # --- real precipitates ---
    writer = stream_copy(
        REAL_PRECIPITATE_PARQUET,
        writer,
        allow_test=True,
        source_name="real",
        stats=stats,
    )

    # --- synthetic precipitates (TRAIN ONLY) ---
    writer = stream_copy(
        SYNTH_PRECIPITATE_PARQUET,
        writer,
        allow_test=False,
        source_name="synthetic",
        stats=stats,
    )

    # --- clean-only ---
    writer = stream_copy(
        CLEAN_ONLY_PARQUET,
        writer,
        allow_test=True,
        source_name="clean",
        stats=stats,
    )

    if writer:
        writer.close()

    logging.info("=== MERGE DONE ===")
    for k, v in sorted(stats.items()):
        logging.info(f"{k}: {v}")


# --------------------------------------------------
# ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    merge_datasets_streaming()
