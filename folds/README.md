# Folds generation

This directory contains the pipeline for training, tuning, and generating synthetic tissue fold artifacts using Stable Diffusion inpainting.

### 1. `create_dataset.py`
Converts raw image tiles and their corresponding primary masks into a structured Parquet dataset. It automatically generates text descriptions based on the mask's size and position.
    
**Usage:**
```bash
python folds/create_dataset.py \
  --source_dir "path/to/tiles_and_masks_folder" \
  --output_path "lung_folds_dataset.parquet"
```
### 2. `train_inpainting_folds.py`
Fine-tunes a Stable Diffusion UNet model using an inpainting pipeline.
```
python folds/train_inpainting_folds.py
```

### 3. tune_inference_hyperparameters.py
Runs inference over a grid of parameters (e.g., guidance scale and denoising strength) and evaluates the outputs using KID, FID, CLIP, and Color Distance to find the optimal generation settings.
```
python folds/tune_inference_hyperparameters.py \
  --checkpoint "path/to/saved/model_checkpoint" \
  --parquet_source "path/to/test_dataset.parquet" \
  --output_csv "tuning_results.csv" \
  --device "cuda"
```

### 4. tune_inference_hyperparameters.py

Applies the procedural fold artifacts to clean datasets using fine-tuned model. It supports providing a JSON manifest to selectively apply artifacts to specific files and can pull real mask shapes from a Parquet file.
```
python folds/generate_folds.py \
  --checkpoint "path/to/saved/model_checkpoint" \
  --input_source "path/to/clean_images_folder" \
  --output_dir "path/to/save_generated_images" \
  --parquet_for_masks "path/to/real_masks.parquet" \
  --json_filter "manifest.json"
```
