import pandas as pd
import numpy as np
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
INPUT_FILE = "lung_precipitate_training_dataset_MPP_09_384px.parquet"
OUTPUT_FILE = "lung_precipitate_training_dataset_MPP_09_384px_split.parquet"
RANDOM_SEED = 42

# Split Ratios (Must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def is_mask_black(mask_bytes):
    """
    Decodes a byte string into an image and checks if all pixels are black (0).
    Returns True if mask is empty/black, False otherwise.
    """
    if mask_bytes is None:
        return True

    try:
        img = Image.open(io.BytesIO(mask_bytes))
        extrema = img.getextrema()

        if isinstance(extrema, tuple) and isinstance(extrema[0], tuple):
            return all(x[1] == 0 for x in extrema)
        elif isinstance(extrema, tuple):
            return extrema[1] == 0
        else:
            return np.array(img).max() == 0

    except Exception as e:
        print(f"Error processing mask: {e}")
        return True


def main():
    print("--- Step 1: Loading Mask + Source column only (Memory Optimization) ---")
    df_meta = pd.read_parquet(INPUT_FILE, columns=["mask", "source"])
    print(f"Loaded {len(df_meta)} records.")

    print("--- Step 2: Analyzing Masks (Empty vs Non-Empty) ---")
    tqdm.pandas(desc="Checking masks")
    df_meta["is_empty"] = df_meta["mask"].progress_apply(is_mask_black)

    # Separate synthetic vs real
    df_synth = df_meta[df_meta["source"] == "synthetic"]
    df_real = df_meta[df_meta["source"] != "synthetic"]

    print(f"\nSynthetic samples (forced to train): {len(df_synth)}")
    print(f"Real samples (used for stratified split): {len(df_real)}")

    # Global stats only for REAL data (since only this part is stratified)
    total_real = len(df_real)
    total_empty_real = df_real["is_empty"].sum()
    global_empty_ratio_real = (total_empty_real / total_real) * 100

    print("\nReal Data Stats:")
    print(f"Total Real Records: {total_real}")
    print(f"Empty Masks (Real): {total_empty_real} ({global_empty_ratio_real:.2f}%)")

    print("\n--- Step 3: Stratified Split on REAL data only ---")

    indices_real = df_real.index.to_numpy()
    labels_real = df_real["is_empty"].to_numpy()

    test_val_ratio = VAL_RATIO + TEST_RATIO

    # 1. Train vs (Val + Test)
    X_train_real, X_temp, y_train_real, y_temp = train_test_split(
        indices_real,
        labels_real,
        stratify=labels_real,
        test_size=test_val_ratio,
        random_state=RANDOM_SEED,
    )

    # 2. Val vs Test
    relative_test_size = TEST_RATIO / test_val_ratio
    X_val_real, X_test_real, y_val_real, y_test_real = train_test_split(
        X_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_test_size,
        random_state=RANDOM_SEED,
    )

    # Build split map
    split_map = {}

    # Real splits
    for idx in X_train_real:
        split_map[idx] = "train"
    for idx in X_val_real:
        split_map[idx] = "val"
    for idx in X_test_real:
        split_map[idx] = "test"

    # Synthetic → always train
    for idx in df_synth.index:
        split_map[idx] = "train"

    df_meta["new_split"] = df_meta.index.map(split_map)

    print("\n--- Step 4: VERIFICATION ---")
    print(f"{'Split':<10} | {'Total':<10} | {'Empty Count':<12} | {'Empty %':<10}")
    print("-" * 60)

    for split_name in ["train", "val", "test"]:
        subset = df_meta[df_meta["new_split"] == split_name]
        count = len(subset)
        empty_count = subset["is_empty"].sum()
        ratio = (empty_count / count) * 100 if count > 0 else 0

        print(f"{split_name:<10} | {count:<10} | {empty_count:<12} | {ratio:.2f}%")

    print("-" * 60)
    print(f"Target Real Empty %: {global_empty_ratio_real:.2f}%")
    print("Val/Test percentages should closely match real-data empty ratio.")

    print("\n--- Step 5: Loading Full Data and Saving ---")

    df_full = pd.read_parquet(INPUT_FILE)

    df_full["split"] = df_full.index.map(split_map)

    df_full.to_parquet(OUTPUT_FILE)

    print(f"Success! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
