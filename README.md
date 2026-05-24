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
* **`evaluate_fidelity_with_CI.sh` / `evaluate_fidelity_PLIP.sh`**: Shell scripts to batch-execute the evaluation routines across different artifact datasets (marker, dust, folds, precipitates).
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
