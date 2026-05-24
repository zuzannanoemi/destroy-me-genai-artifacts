import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Added for drawing rectangles
import logging
from collections import Counter
from typing import Dict, List

Image.MAX_IMAGE_PIXELS = None  # to inrease limit to handle large files

ARTIFACTS_NAMES = [
    "air_bubble",
    "clean",
    "clear",
    "dust",
    "folds",
    "ink",
    "knife_marks",
    "knife_scratch",
    "marker",
    "precipitates",
    "scratches",
    "waves",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_dimension_histogram(
    data_path: str,
    subdirectories: list[str] = ["masks", "tiles"],
    hist_name: str = "dimension_histogram.png",
) -> None:
    """
    Analyzes the dimensions of images in 'masks' and 'tiles' subdirectories
    and creates histograms of their dimensions.

    Args:
        data_path (str): The main path to the directory containing the
                         'masks' and 'tiles' subdirectories.
    """
    subdirectories = subdirectories
    dimensions_list = []

    # Check if the main directory exists
    if not os.path.isdir(data_path):
        logging.error(f"The specified directory does not exist: {data_path}")
        return

    # Loop through the 'masks' and 'tiles' subdirectories
    for subdirectory in subdirectories:
        current_path = os.path.join(data_path, subdirectory)

        if not os.path.isdir(current_path):
            logging.warning(f"Subdirectory not found: {current_path}")
            continue

        logging.info(f"Processing images in directory: {current_path}")

        # Loop through all files in the current subdirectory
        for filename in os.listdir(current_path):
            file_path = os.path.join(current_path, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                try:
                    # Open the image and get its dimensions
                    with Image.open(file_path) as img:
                        width, height = img.size
                        dimensions_list.append((width, height))
                except (IOError, SyntaxError) as e:
                    logging.warning(
                        f"Failed to open file as an image: {file_path} - {e}"
                    )

    if not dimensions_list:
        logging.error("No valid images were found to analyze.")
        return

    logging.info(f"Found and analyzed {len(dimensions_list)} images.")

    # Prepare data for plotting
    widths = [dim[0] for dim in dimensions_list]
    heights = [dim[1] for dim in dimensions_list]

    # Create the visualizations
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Analysis of Histopathological Image Dimensions", fontsize=16)

    # 1. Histogram of widths
    axs[0].hist(widths, bins=30, color="skyblue", edgecolor="black")
    axs[0].set_title("Image Widths Histogram")
    axs[0].set_xlabel("Width (pixels)")
    axs[0].set_ylabel("Number of Images")

    # 2. Histogram of heights
    axs[1].hist(heights, bins=30, color="salmon", edgecolor="black")
    axs[1].set_title("Image Heights Histogram")
    axs[1].set_xlabel("Height (pixels)")
    axs[1].set_ylabel("Number of Images")

    # 3. Scatter plot (width vs. height)
    axs[2].scatter(widths, heights, alpha=0.6, color="mediumseagreen")
    axs[2].set_title("Dimension Distribution (Width vs Height)")
    axs[2].set_xlabel("Width (pixels)")
    axs[2].set_ylabel("Height (pixels)")
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(hist_name)


def analyse_number_of_tiles_per_artifact(
    data_path: str, prefixes: List[str] = ARTIFACTS_NAMES
) -> Dict[str, int]:
    """
    Iterates through the file names in the directory
    and extracts the first part of the name based on a list of allowed prefixes.

    Args:
        data_path: Path to files directory
        prefixes: List of acceptable prefixes (artefacts types).

    Returns:
        Dictionary with key: artifact name, vale: number of occurances.
    """
    if not os.path.isdir(data_path):
        print(f"Error: Path '{data_path}' is not directory or does not exsist.")
        return {}

    found_prefixes: List[str] = []

    sorted_prefixes = sorted(prefixes, key=len, reverse=True)

    try:
        for file_name_with_ext in os.listdir(data_path):
            full_path = os.path.join(data_path, file_name_with_ext)

            if os.path.isfile(full_path):
                file_name = os.path.splitext(file_name_with_ext)[0]

                prefix_found = None

                for prefix in sorted_prefixes:

                    if file_name.startswith(prefix):
                        end_index = len(prefix)

                        if end_index == len(file_name) or file_name[end_index] == "_":
                            prefix_found = prefix
                            break  

                if prefix_found:
                    found_prefixes.append(prefix_found)


        cnt = Counter(found_prefixes)
        return dict(cnt)

    except Exception as e:
        print(f"Error during directory processing {e}")
        return {}


def visualize_artifacts_tiles_occurences(
    artefacts: Dict[str, int],
    fig_name: str = "tiles_per_artifact.png",
    title: str = "Number of artifacts (tiles)",
):
    """
    Visualize number of tiles per artifact
    """
    if not artefacts:
        print("No data for visualization")
        return

    names = list(artefacts.keys())
    numbers = list(artefacts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, numbers, color="darkgreen")

    plt.xlabel("Artifacts")
    plt.ylabel("Tiles")
    plt.title(title)

    plt.xticks(rotation=45, ha="right")

    for i, liczba in enumerate(numbers):
        plt.text(i, liczba, str(liczba), ha="center", va="bottom")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_name)


def create_mask_size_boxplot_by_artifact(
    data_path: str,
    allowed_prefixes: List[str] = ARTIFACTS_NAMES,
    output_filename: str = "mask_size_boxplot_by_artifact_limits.png",
    max_y_limit: int | None = None,  # NEW PARAMETER: Maximum value for the Y-axis
) -> None:
    """
    Analyzes the dimensions (area) of mask images and creates a box plot
    of mask sizes grouped by the file name prefix (artifact name),
    with an optional Y-axis limit to handle outliers.

    Args:
        data_path (str): The main path to the directory containing the 'masks' subdirectory.
        allowed_prefixes (List[str]): List of valid file name prefixes (artifact names).
        output_filename (str): The output filename for the generated plot.
        max_y_limit (int, optional): The maximum limit for the Y-axis.
                                     If set, it helps visualize compressed data by ignoring extreme outliers.
    """
    masks_path = os.path.join(data_path, "masks")

    # Dictionary to store mask sizes (area), grouped by prefix.
    mask_sizes_by_artifact: Dict[str, List[int]] = {
        prefix: [] for prefix in allowed_prefixes
    }

    if not os.path.isdir(masks_path):
        logging.error(f"Masks directory does not exist: {masks_path}")
        return

    # Sort prefixes for correct parsing of multi-word prefixes
    sorted_prefixes = sorted(allowed_prefixes, key=len, reverse=True)

    logging.info(f"Processing masks in directory: {masks_path}")
    total_masks = 0

    # ... (Image loading and prefix matching logic remains the same) ...
    for full_filename in os.listdir(masks_path):
        file_path = os.path.join(masks_path, full_filename)

        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mask_area = width * height
                    total_masks += 1

                filename_without_extension = os.path.splitext(full_filename)[0]
                matched_prefix = None

                for prefix in sorted_prefixes:
                    if filename_without_extension.startswith(prefix):
                        end_index = len(prefix)

                        if (
                            end_index == len(filename_without_extension)
                            or filename_without_extension[end_index] == "_"
                        ):
                            matched_prefix = prefix
                            break

                if matched_prefix in mask_sizes_by_artifact:
                    mask_sizes_by_artifact[matched_prefix].append(mask_area)

            except (IOError, SyntaxError) as e:
                logging.warning(f"Failed to open file as an image: {file_path} - {e}")

    logging.info(f"Found and analyzed {total_masks} masks.")

    # 4. Prepare data for the Box Plot
    data_to_plot = [
        mask_sizes_by_artifact[prefix]
        for prefix in allowed_prefixes
        if mask_sizes_by_artifact[prefix]
    ]
    labels_to_plot = [
        prefix for prefix in allowed_prefixes if mask_sizes_by_artifact[prefix]
    ]

    if not data_to_plot:
        logging.error("No masks matched the allowed prefixes. Cannot create plot.")
        return

    # 5. Create and save the Box Plot
    plt.figure(figsize=(14, 8))

    # Draw the Box Plot
    plt.boxplot(
        data_to_plot,
        labels=labels_to_plot,
        vert=True,
        patch_artist=True,
        medianprops={"color": "red"},
        boxprops={"facecolor": "lightblue"},
    )

    # *** Implementation of Y-axis limit to handle outliers ***
    if max_y_limit is not None and isinstance(max_y_limit, int) and max_y_limit > 0:
        plt.ylim(0, max_y_limit)
        plt.title(
            f"Distribution of Mask Sizes (Area) (Y-axis capped at {max_y_limit})",
            fontsize=16,
        )
        logging.info(f"Plot Y-axis capped at {max_y_limit} to handle extreme outliers.")
    else:
        plt.title("Distribution of Mask Sizes (Area) by Artifact Prefix", fontsize=16)

    plt.xlabel("Artifact Name (File Prefix)", fontsize=12)
    plt.ylabel("Mask Size (Number of Pixels)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()

    plt.savefig(output_filename)
    logging.info(f"Box plot saved as: {output_filename}")


def _calculate_stride_size(patch_size: int, overlap: float) -> int:
    stride = int(patch_size * (1 - overlap))
    if stride <= 0:
        stride = patch_size
    return stride


def _save_patch_and_mask(
    tile_patch: Image.Image,
    mask_patch: Image.Image,
    row_idx: int,
    col_idx: int,
    base_name: str,
    extension: str,
    output_tiles_path: str,
    output_masks_path: str,
) -> None:
    patch_tile_filename = f"{base_name}_r{row_idx}_c{col_idx}{extension}"
    # patch_mask_filename = f"{base_name}_mask_r{row_idx}_c{col_idx}{extension}"
    patch_mask_filename = f"{base_name}_r{row_idx}_c{col_idx}_primary_mask.png"

    tile_patch.save(os.path.join(output_tiles_path, patch_tile_filename))
    mask_patch.save(os.path.join(output_masks_path, patch_mask_filename))
    return


def create_patches_from_image(
    tile: Image.Image,
    mask: Image.Image,
    patch_size: int,
    overlap: float,
    base_name: str,
    extension: str,
    output_tiles_path: str,
    output_masks_path: str,
) -> int:
    width, height = tile.size
    #  Calculate Stride based on Overlap
    stride = _calculate_stride_size(patch_size, overlap)
    patch_count = 0
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            box = (x, y, x + patch_size, y + patch_size)
            if box[2] > width or box[3] > height:
                continue

            tile_patch = tile.crop(box)
            mask_patch = mask.crop(box)

            #  Generate New Filename and Save
            row_idx = y // stride
            col_idx = x // stride

            _save_patch_and_mask(
                tile_patch,
                mask_patch,
                row_idx,
                col_idx,
                base_name,
                extension,
                output_tiles_path,
                output_masks_path,
            )

            patch_count += 1

    return patch_count


def create_patches_from_images(
    source_dir: str, output_dir: str, patch_size: int = 256, overlap: float = 0.5
) -> None:
    """
    Cuts large images and their corresponding masks into smaller patches.

    This function iterates through images in `source_dir/tiles`, constructs the
    matching mask name (e.g., 'image1.png' -> 'image1_mask.png'), resizes the
    mask, and then generates overlapping patches for both.

    Args:
        source_dir (str): Path to the directory containing 'tiles' and 'masks' subfolders.
        output_dir (str): Path to the directory where patched images will be saved.
        patch_size (int): The width and height of the square patches (default: 256).
        overlap (float): The fraction of overlap between adjacent patches (0.0 to 1.0).
    """
    # --- 1. Setup Paths and Directories ---
    source_tiles_path = os.path.join(source_dir, "tiles")
    source_masks_path = os.path.join(source_dir, "masks")
    output_tiles_path = os.path.join(output_dir, "tiles")
    output_masks_path = os.path.join(output_dir, "masks")

    if not os.path.isdir(source_tiles_path) or not os.path.isdir(source_masks_path):
        logging.error(
            f"Source directory '{source_dir}' must contain 'tiles' and 'masks' subfolders."
        )
        return

    os.makedirs(output_tiles_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)
    logging.info(f"Output will be saved to '{output_dir}'")

    # --- 2. Process Each Image ---
    tile_filenames = sorted(os.listdir(source_tiles_path))

    for filename in tile_filenames:
        tile_path = os.path.join(source_tiles_path, filename)

        # Construct the corresponding mask filename ---
        base_name, extension = os.path.splitext(filename)
        mask_filename = f"{base_name}_mask{extension}"
        mask_path = os.path.join(source_masks_path, mask_filename)

        if not os.path.isfile(mask_path):
            logging.warning(
                f"Mask '{mask_filename}' not found for tile '{filename}'. Skipping."
            )
            continue

        try:
            with Image.open(tile_path) as tile, Image.open(mask_path) as mask:
                logging.info(
                    f"Processing '{filename}' and its mask '{mask_filename}'..."
                )

                # --- 3. Resize Mask to Match Tile ---
                if tile.size != mask.size:
                    logging.info(f"  Resizing mask from {mask.size} to {tile.size}...")
                    mask = mask.resize(tile.size, Image.NEAREST)

                patch_count = create_patches_from_image(
                    tile,
                    mask,
                    patch_size,
                    overlap,
                    base_name,
                    extension,
                    output_tiles_path,
                    output_masks_path,
                )

                logging.info(f"  --> Saved {patch_count} patches for '{filename}'.")

        except Exception as e:
            logging.error(f"Failed to process file '{filename}'. Error: {e}")

    logging.info("--- Patch creation complete! ---")
    return


def create_patches_for_artifact(
    source_dir: str,
    output_dir: str,
    artifact_name: str,
    allowed_prefixes: List[str],  # Needed for parsing logic
    patch_size: int = 256,
    overlap: float = 0.5,
) -> None:
    """
    Cuts large images and their corresponding primary masks into smaller patches,
    processing only files matching the specified artifact name (prefix).

    Args:
        source_dir (str): Path to the directory containing 'tiles' and 'masks' subfolders.
        output_dir (str): Path to the directory where patched images will be saved.
        artifact_name (str): The specific prefix (e.g., 'air_bubble') to process.
        allowed_prefixes (List[str]): List of all known prefixes (used for sorting and parsing).
        patch_size (int): The width and height of the square patches (default: 256).
        overlap (float): The fraction of overlap between adjacent patches (0.0 to 1.0).
    """
    # --- 1. Setup Paths and Directories ---
    source_tiles_path = os.path.join(source_dir, "tiles")
    source_masks_path = os.path.join(source_dir, "masks")
    output_tiles_path = os.path.join(output_dir, "tiles")
    output_masks_path = os.path.join(output_dir, "masks")

    if not os.path.isdir(source_tiles_path) or not os.path.isdir(source_masks_path):
        logging.error(
            f"Source directory '{source_dir}' must contain 'tiles' and 'masks' subfolders."
        )
        return

    if artifact_name not in allowed_prefixes:
        logging.error(
            f"Artifact name '{artifact_name}' is not in the list of allowed prefixes."
        )
        return

    os.makedirs(output_tiles_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)
    logging.info(f"Output will be saved to '{output_dir}'")

    # Sort prefixes for correct parsing
    sorted_prefixes = sorted(allowed_prefixes, key=len, reverse=True)

    # --- 2. Process Only Matching Images ---
    tile_filenames = sorted(os.listdir(source_tiles_path))
    processed_count = 0

    logging.info(f"Starting patch creation for artifact: '{artifact_name}'")

    for filename in tile_filenames:
        base_name, extension = os.path.splitext(filename)

        # --- FILTRATION LOGIC ---

        # 2a. Determine the file's prefix using the robust logic
        file_prefix = None
        for prefix in sorted_prefixes:
            if base_name.startswith(prefix):
                end_index = len(prefix)
                # Check if followed by '_' or end of name
                if end_index == len(base_name) or base_name[end_index] == "_":
                    file_prefix = prefix
                    break

        # 2b. Skip file if it doesn't match the required artifact name
        if file_prefix != artifact_name:
            continue

        # --- PATCHING LOGIC (Only executed if filtered) ---

        tile_path = os.path.join(source_tiles_path, filename)

        # Change: Construct the corresponding mask filename with the new suffix
        mask_filename = f"{base_name}_primary_mask.png"
        mask_path = os.path.join(source_masks_path, mask_filename)

        if not os.path.isfile(mask_path):
            logging.warning(
                f"Primary mask '{mask_filename}' not found for tile '{filename}'. Skipping."
            )
            continue

        try:
            with Image.open(tile_path) as tile, Image.open(mask_path) as mask:
                logging.info(
                    f"Processing '{filename}' and its mask '{mask_filename}'..."
                )

                # Resize Mask to Match Tile (Existing Logic)
                if tile.size != mask.size:
                    logging.info(f"  Resizing mask from {mask.size} to {tile.size}...")
                    mask = mask.resize(tile.size, Image.NEAREST)

                patch_count = create_patches_from_image(
                    tile,
                    mask,
                    patch_size,
                    overlap,
                    base_name,
                    extension,
                    output_tiles_path,
                    output_masks_path,
                )

                logging.info(f"  --> Saved {patch_count} patches for '{filename}'.")
                processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process file '{filename}'. Error: {e}")

    logging.info(
        f"--- Patch creation complete! Processed {processed_count} tiles for artifact '{artifact_name}'. ---"
    )
    return


def filter_patches_by_mask_content(
    source_dir: str,
    output_dir: str,
    min_white_ratio: float = 0.2,
    max_white_ratio: float = 0.75,
) -> None:
    """
    Filters image patches based on the percentage of white pixels in their masks.

    This function iterates through mask patches, calculates the ratio of white pixels,
    and copies both the mask and its corresponding tile to the output directory if
    the ratio falls within the specified range [min_white_ratio, max_white_ratio].

    Args:
        source_dir (str): Path to the directory containing patched 'tiles' and 'masks'.
        output_dir (str): Path to the directory where filtered patches will be saved.
        min_white_ratio (float): The minimum fraction of white pixels required (inclusive).
        max_white_ratio (float): The maximum fraction of white pixels allowed (inclusive).
    """
    # --- 1. Setup Paths and Directories ---
    source_tiles_path = os.path.join(source_dir, "tiles")
    source_masks_path = os.path.join(source_dir, "masks")
    output_tiles_path = os.path.join(output_dir, "tiles")
    output_masks_path = os.path.join(output_dir, "masks")

    if not os.path.isdir(source_masks_path):
        logging.error(f"Source directory for masks not found: '{source_masks_path}'")
        return

    # Create output directories, clearing them if they already exist for a clean run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.warning(f"Removed existing output directory: '{output_dir}'")

    os.makedirs(output_tiles_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)
    logging.info(f"Filtered patches will be saved to '{output_dir}'")

    # --- 2. Iterate Through Mask Patches and Filter ---
    mask_filenames = sorted(os.listdir(source_masks_path))
    kept_count = 0
    discarded_count = 0

    for mask_filename in mask_filenames:
        mask_path = os.path.join(source_masks_path, mask_filename)

        try:
            with Image.open(mask_path) as mask_img:
                # Convert mask to a NumPy array for efficient calculation
                # 'L' mode ensures it's treated as a single-channel grayscale image
                mask_array = np.array(mask_img.convert("L"))

                # --- 3. Calculate White Pixel Ratio ---
                # White pixels are typically 255. We count non-zero pixels for robustness.
                white_pixels = np.count_nonzero(mask_array)
                total_pixels = mask_array.size
                white_ratio = white_pixels / total_pixels

                # --- 4. Apply Filtering Logic ---
                if min_white_ratio <= white_ratio <= max_white_ratio:
                    # This patch is good, so we copy it and its corresponding tile

                    # --- 5. Find and Copy Corresponding Tile and Mask ---
                    # Construct the tile filename from the mask filename
                    tile_filename = mask_filename.replace("_primary_mask", "")
                    source_tile_path = os.path.join(source_tiles_path, tile_filename)

                    if os.path.exists(source_tile_path):
                        # Define destination paths
                        dest_mask_path = os.path.join(output_masks_path, mask_filename)
                        dest_tile_path = os.path.join(output_tiles_path, tile_filename)

                        # Use shutil.copy2 to preserve file metadata
                        shutil.copy2(mask_path, dest_mask_path)
                        shutil.copy2(source_tile_path, dest_tile_path)
                        kept_count += 1
                    else:
                        logging.warning(
                            f"Tile not found for mask '{mask_filename}'. Skipping."
                        )
                        discarded_count += 1
                else:
                    # This patch is outside our desired range
                    discarded_count += 1

        except Exception as e:
            logging.error(f"Could not process mask file '{mask_filename}'. Error: {e}")
            discarded_count += 1

    logging.info("--- Filtering complete! ---")
    logging.info(f"Total patches processed: {len(mask_filenames)}")
    logging.info(f"Patches kept: {kept_count}")
    logging.info(f"Patches discarded: {discarded_count}")


def visualize_patching_strategy(
    tile_path: str,
    mask_path: str,
    patch_size: int = 256,
    overlap: float = 0.5,
    min_white_ratio: float = 0.2,
    max_white_ratio: float = 0.75,
    output_plot_path: str | None = None,
) -> None:
    """
    Visualizes the tile with mask overlay and draws rectangles for patches.
    Kept (Lime Green) patches are drawn last to ensure visibility over Discarded (Red) patches.
    """
    if not os.path.exists(tile_path) or not os.path.exists(mask_path):
        logging.error("Tile or Mask file not found.")
        return

    try:
        # 1. Load Images
        tile = Image.open(tile_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load as grayscale

        # Resize mask if needed
        if tile.size != mask.size:
            mask = mask.resize(tile.size, Image.NEAREST)

        mask_arr = np.array(mask)
        width, height = tile.size

        # 2. Setup Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(tile)
        del tile
        del mask
        import gc

        gc.collect()
        # Overlay mask (reverted to solid overlay as requested)
        # Using a distinct color map (e.g., 'jet') and transparency
        ax.imshow(mask_arr, cmap="jet", alpha=0.3)

        # 3. Calculate Patches and Store Status
        stride = _calculate_stride_size(patch_size, overlap)
        patches_to_draw = []
        total_patches = 0
        kept_patches = 0

        # Loop 1: Determine all patch coordinates and status
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                box = (x, y, x + patch_size, y + patch_size)

                if box[2] > width or box[3] > height:
                    continue

                patch_mask_data = mask_arr[y : y + patch_size, x : x + patch_size]
                white_pixels = np.count_nonzero(patch_mask_data)
                total_pixels = patch_mask_data.size
                white_ratio = white_pixels / total_pixels

                is_kept = min_white_ratio <= white_ratio <= max_white_ratio

                patches_to_draw.append({"x": x, "y": y, "kept": is_kept})

                if is_kept:
                    kept_patches += 1
                total_patches += 1

        # 4. Drawing: Discarded (Red) first
        for p in patches_to_draw:
            if not p["kept"]:
                rect = patches.Rectangle(
                    (p["x"], p["y"]),
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

        # 5. Drawing: Kept (Lime Green) second, ensuring they are on top
        for p in patches_to_draw:
            if p["kept"]:
                rect = patches.Rectangle(
                    (p["x"], p["y"]),
                    patch_size,
                    patch_size,
                    linewidth=2,  # Using a thicker line for emphasis
                    edgecolor="lime",
                    facecolor="none",
                )
                ax.add_patch(rect)

        # 6. Finalize Plot
        title_base = os.path.basename(tile_path)
        ax.set_title(
            f"Patch Strategy for: {title_base}\n"
            f"Kept: {kept_patches}/{total_patches} | Range: {min_white_ratio}-{max_white_ratio}"
        )
        ax.axis("off")

        if output_plot_path:
            plt.savefig(output_plot_path, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"Visualization saved to {output_plot_path}")
        else:
            plt.show()

    except Exception as e:
        logging.error(f"Visualization failed for {tile_path}. Error: {e}")


## 🔁 Wrapper Function for Batch Processing


def create_visualizations_for_all_tiles(
    source_dir: str,
    output_vis_dir: str,
    patch_size: int = 256,
    overlap: float = 0.5,
    min_white_ratio: float = 0.2,
    max_white_ratio: float = 0.75,
) -> None:
    """
    Wraps visualize_patching_strategy to create visualizations for all tile/mask pairs
    in the source directory.
    """
    source_tiles_path = os.path.join(source_dir, "tiles")
    source_masks_path = os.path.join(source_dir, "masks")

    if not os.path.isdir(source_tiles_path) or not os.path.isdir(source_masks_path):
        logging.error(
            f"Source directory '{source_dir}' must contain 'tiles' and 'masks' subfolders."
        )
        return

    os.makedirs(output_vis_dir, exist_ok=True)
    logging.info(f"All visualizations will be saved to '{output_vis_dir}'")

    tile_filenames = sorted(os.listdir(source_tiles_path))

    for filename in tile_filenames:
        tile_path = os.path.join(source_tiles_path, filename)

        # Construct the corresponding mask filename
        base_name, extension = os.path.splitext(filename)
        # Assuming the mask filename structure follows your original script's logic:
        # e.g., 'image1.png' -> 'image1_mask.png'
        mask_filename = f"{base_name}_mask{extension}"
        mask_path = os.path.join(source_masks_path, mask_filename)

        if not os.path.isfile(mask_path):
            logging.warning(
                f"Mask '{mask_filename}' not found for tile '{filename}'. Skipping."
            )
            continue

        # Define the output path for the visualization plot
        output_name = f"{base_name}_patch_vis.png"
        output_vis_path = os.path.join(output_vis_dir, output_name)

        logging.info(f"Creating visualization for: {filename}")

        visualize_patching_strategy(
            tile_path=tile_path,
            mask_path=mask_path,
            patch_size=patch_size,
            overlap=overlap,
            min_white_ratio=min_white_ratio,
            max_white_ratio=max_white_ratio,
            output_plot_path=output_vis_path,  # Saves the file instead of showing it
        )

    logging.info("--- Batch visualization complete! ---")
