# Destroy me - Histopathology Artifacts Generation

**Research use only — Not for clinical or diagnostic use.** This software is intended for research and development purposes only. It has not been validated for clinical decision making, diagnostics, or any regulated medical use. Do not use in patient care or production medical environments.

Repository of **Destroy Me**  a dual pathway framework that unifies deep generative models and physics-based pipelines to synthesize six distinct, high-fidelity histological artifact types (tissue folds, precipitates, blur, stitching errors, dust, and pen marks) while preserving structural morphological continuity.

This repository was developed for the research work titled: "Synthesizing Realistic Imperfections: A Data-Centric Framework for Robustness in Computational Pathology".



## Project Structure

### `data_preprocessing/`
Scripts dedicated to preparing, analyzing, and patching dataset images.
* **`data_preprocessing.py`**: Analyzes image dimensions, creates histograms and boxplots of mask sizes, and handles cutting large high-resolution tissue images and their corresponding masks into smaller overlapping patches for training.
### `evaluation/`
Scripts to measure the fidelity and semantic grounding of the generated artifacts.
* **`judge_results.py`**: Computes standard image quality and generative metrics such as Frechet Inception Distance (FID), Kernel Inception Distance (KID), Masked SSIM, and Wasserstein color distance.
* **`judge_results_PLIP.py`**: Calculates the PLIP (Pathology Language-Image Pretraining) Semantic Grounding score using PLIP to evaluate text-to-image alignment.

### `helpers/`
* **`generate_manifests.py`**: A utility that creates deterministic JSON manifests, specifying exactly which images should receive which artifacts.
  
### `folds/`
Scripts focused on generating tissue fold artifacts using deep learning models.
* **`create_dataset.py`**: Converts raw image tiles and masks into a structured Parquet dataset, generating simple text descriptions based on mask size and position.
* **`train_inpainting_folds.py`**: Fine-tunes a Stable Diffusion UNet model using an inpainting pipeline to generate realistic tissue folds.
* **`tune_inference_hyperparameters.py`**: Evaluates different guidance scales and denoising strengths during inference to optimize FID and KID scores.
* **`generate_folds.py`**: Uses the fine-tuned Stable Diffusion model to apply fold artifacts to clean datasets.
### `dust/`
Tools for creating procedural dust and dirt artifacts.
* **`generate_dust.py`**: Contains `DustGenerator` and `DustAugmentor` classes that procedurally generate organic "smudge" or "particulate" dust artifacts on clean tissue, including tissue defocus simulations.
### `marker/`
* **`generate_marker.py`**: Procedurally generates pen marks artifacts using  geometric logic and subtractive color blending to simulate real marker ink on glass slides.
### `precipitates/`
A comprehensive folder handling the procedural and generative creation of precipitate (dye/crystal) artifacts.
* **`precitipate_artifact_synth_advanced.py`**: Procedurally generates complex artifact shapes like crystals (Convex Hulls), fluid blobs, striated fibers, and granular dirt, complete with physics-based refractive blending.
* **`generate_synthetic_precitipate_parquet.py`**: Wraps the procedural generator to build a synthetic parquet dataset of precipitates in a streaming fashion.
* **`create_precipitate_dataset_real.py`**: Extracts real precipitate instances into a Parquet dataset and uses HSV logic to assign colors (reddish, violet, gray) to the prompts.
* **`merge_percipitate_datasets.py`**: Merges real, synthetic, and clean parquet files into a single master training dataset.
* **`add_train_val_test_split.py`**: Splits the master precipitate dataset using stratified logic based on whether masks are empty or not.
* **`train_inpainting_perticipates.py`**: The training loop for fine-tuning a Stable Diffusion model specifically for precipitates, handling both generation (drawing the artifact) and regularization (leaving clean tissue alone).
* **`generate_precipitates.py`**: Script to execute inference
* **`tune_inference_parameters_precitipates.py`**:  Script to tune hyperparameter scales for the trained precipitate models.

### `scanning_artifacts/`
Tools for simulating optical and mechanical errors during the slide scanning process.
* **`scanning_artefacts.py`**: Core engine that defines `BlurConfig` and `StitchConfig` to simulate optical defocus (feathered gaussian blur patches) and mechanical stitching errors (pixel shifting across axes).
* **`generate_scanning_artifacts.py`**: Wraps the scanning artifact engine, safely applying randomized horizontal/vertical shifts and blurred strips while avoiding the edges of the image tiles.

## Environment Setup
This project uses a standard Python virtual environment. To set it up and install the necessary dependencies, run the following commands in your terminal:

**1. Create the virtual environment:**
```bash
python3 -m venv venv
```
**2. Activate environment:**

```bash
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

## License

See [LICENSE.md](LICENSE.md)

## Usage Guide

### 1. Generating Manifests
Before generating artifacts, creation of JSON manifests is needed. These manifests deterministically allocate which clean images will receive which artifacts, ensuring there is no uncontrolled overlap (unless desired).

Use the `generate_manifests.py` script to create these files. You can specify the exact fraction of your dataset to allocate to each artifact using the `--artifacts` flag.

```
python helpers/generate_manifests.py \
  --input_dir "path/to/clean_images" \
  --output_dir "path/to/save_manifests" \
  --artifacts "dust:0.1 marker:0.1 fold:0.2 prec:0.2 blur:0.1 stitch:0.1"
```
### 2. Generating Artifacts

**generate_folds**
```
python folds/generate_folds.py \
  --checkpoint "path/to/folds_model_checkpoint" \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_folds" \
  --parquet_for_masks "path/to/real_fold_masks.parquet" \
  --json_filter "path/to/manifests/manifest_stress_fold_02.json"
```
**generate_precipitate**
```
python precipitates/generate_precipitates.py \
  --checkpoint "path/to/prec_model_checkpoint" \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_precipitates" \
  --parquet_for_masks "path/to/real_prec_masks.parquet" \
  --json_filter "path/to/manifests/manifest_stress_prec_02.json"
```

**Generate Pen Marker**
```
python marker/generate_marker.py \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_marker" \
  --json_filter "path/to/manifests/manifest_stress_marker_01.json"
```
**Generate Dust**
```
python dust/generate_dust.py \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_dust" \
  --json_filter "path/to/manifests/manifest_stress_dust_01.json"
```

**Generate Scanning Artifacts (Optical Blur)**
```
python scanning_artifacts/generate_scanning_artifacts.py \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_blur" \
  --artifact_type "blur" \
  --json_filter "path/to/manifests/manifest_stress_blur_01.json"
```

**Generate Scanning Artifacts (Mechanical Stitch)**
```
python scanning_artifacts/generate_scanning_artifacts.py \
  --input_source "path/to/clean_images" \
  --output_dir "path/to/output_stitch" \
  --artifact_type "stitch" \
  --json_filter "path/to/manifests/manifest_stress_stitch_01.json"
```
## Training models

**Precipitates**
```
accelerate launch precipitates/train_inpainting_perticipates.py \
  --data "path/to/lung_precipitate_training_dataset.parquet" \
  --pretrained_model "stable-diffusion-v1-5/stable-diffusion-inpainting" \
  --output_dir "trained_precipitates_model" \
  --batch_size 2 \
  --max_steps 15000
```

**Folds**
Folds training script has its configuration and dataset path hardcoded within the main function of the file.
```
accelerate launch folds/train_inpainting_folds.py
```
## Evaluation

To evaluate the fidelity of the generated artifacts, use the `judge_results.py` and `judge_results_PLIP.py` scripts located in the `evaluation/` directory.

### Standard Metrics (FID, KID, SSIM, Color Distance)
Run `judge_results.py` by pointing it to real reference dataset and the folder containing your generated artifacts. 

**Example usage:**
```bash
python evaluation/judge_results.py \
  --real_source path/to/real_training_dataset.parquet \
  --gen_folder path/to/generated_artifacts_folder \
  --artifact_type [marker, dust, folds, precipitates] \
  --clean_source path/to/clean_reference_images/ \
  --calc_ssim \
  --output_file metrics_results.txt
```
### PLIP Score
To evaluate how well the generated artifacts align with pathological text descriptions, use the `judge_results_PLIP.py` script.

```bash
python evaluation/judge_results_PLIP.py \
  --gen_folder path/to/generated_artifacts_folder \
  --artifact_type [folds, precipitates] \
  --output_file plip_results.txt
```

## Model weights
Model weights for folds and precitipates generation along with artifacts masks (stored in parquet files) are accesible for download under following [link](https://dataverse.harvard.edu/previewurl.xhtml?token=438b2fa7-3999-4739-87f1-c3a31f4de838)

## External datasets

External data used:
TCGA-LUAD: Albertina, B., Watson, M., Holback, C., Jarosz, R., Kirk, S., Lee, Y., Rieger-Christ, K., & Lemmerman, J. (2016). The Cancer Genome Atlas Lung Adenocarcinoma Collection (TCGA-LUAD) (Version 4) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.JGNIHEP5

ANORAK: A large-scale, annotated lung cancer dataset. [Link](https://zenodo.org/records/10016027)

DHMC: Lung adenocarcinoma whole-slide images from the Dartmouth-Hitchcock Medical Center. [Link](https://bmirds.github.io/LungCancer/)

## Authors
anonymous
