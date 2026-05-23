import argparse
import torch
import os
import io
import pandas as pd
import numpy as np
import random
from PIL import Image
from diffusers import AutoPipelineForInpainting
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur, crop
from tqdm.auto import tqdm

# Metrics
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from scipy.stats import wasserstein_distance

# --- HELPER: Soft Mask Processing (Crucial for avoiding dots) ---
def process_mask_inference(pil_mask):
    # 1. Resize Nearest (Keep shape)
    t = transforms.ToTensor()(pil_mask.resize((512,512), Image.Resampling.NEAREST))
    # 2. Dilate (Cover edge)
    dilated = torch.nn.functional.max_pool2d(t.unsqueeze(0), kernel_size=15, stride=1, padding=7).squeeze(0)
    # 3. Blur (Soft gradient for VAE)
    blurred = gaussian_blur(dilated, kernel_size=(11, 11), sigma=2.0)
    return transforms.ToPILImage()(blurred)

# --- HELPER: Bounding Box ---
def get_mask_bbox(mask_tensor):
    indices = torch.nonzero(mask_tensor > 0)
    if len(indices) == 0:
        return 0, 0, mask_tensor.shape[1], mask_tensor.shape[2]
    
    y_min, y_max = indices[:, 1].min(), indices[:, 1].max()
    x_min, x_max = indices[:, 2].min(), indices[:, 2].max()
    
    h, w = y_max - y_min, x_max - x_min
    # Add small margin for context
    margin = 5
    top = max(0, int(y_min) - margin)
    left = max(0, int(x_min) - margin)
    height = min(mask_tensor.shape[1] - top, int(h) + 2*margin)
    width = min(mask_tensor.shape[2] - left, int(w) + 2*margin)
    return top, left, height, width

# --- HELPER: Color Distance ---
def calculate_color_distance(gen_tensor, real_tensor):
    dist = 0
    # Compare per channel
    for c in range(3):
        hist_gen, _ = np.histogram(gen_tensor[c].cpu().numpy(), bins=64, range=(0, 1), density=True)
        hist_real, _ = np.histogram(real_tensor[c].cpu().numpy(), bins=64, range=(0, 1), density=True)
        dist += wasserstein_distance(np.arange(64), np.arange(64), hist_gen, hist_real)
    return dist / 3.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--parquet_source", type=str, required=True, help="Path to Parquet file")
    parser.add_argument("--output_csv", type=str, default="hyperparameter_tuning_results.csv")
    parser.add_argument("--device", type=str, default="cuda")
    # KID subset size must be smaller than total samples. 50 is standard for small datasets.
    parser.add_argument("--kid_subset_size", type=int, default=50) 
    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading Data...")
    df = pd.read_parquet(args.parquet_source)
    
    # We need two lists:
    # A. Clean Test Images (Canvas to paint on) -> STRICTLY TEST SET
    # B. Real Artifacts (Ground Truth for Metrics) -> Can be Train+Test (Better Reference Distribution)
    
    clean_test_images = []
    real_artifact_crops = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pre-processing Data"):
        try:
            m_bytes = row['mask']
            if m_bytes is None: continue
            
            # Load Mask
            mask_pil = Image.open(io.BytesIO(m_bytes)).convert("L")
            is_clean = np.array(mask_pil).max() == 0
            
            # A. Clean Test Images
            if is_clean and row.get('split') == 'test':
                clean_test_images.append(row)
                
            # B. Real Artifacts (Reference)
            elif not is_clean:
                # Prepare crop for Metrics immediately to save memory
                img_pil = Image.open(io.BytesIO(row['image'])).convert("RGB")
                m_tensor = transforms.ToTensor()(mask_pil)
                top, left, h, w = get_mask_bbox(m_tensor)
                
                # Only keep if artifact is big enough to be meaningful
                if h > 32 and w > 32:
                    crop_pil = img_pil.crop((left, top, left+w, top+h))
                    # Resize to 299x299 for Inception (FID/KID)
                    crop_tensor = transforms.ToTensor()(crop_pil)
                    crop_resized = torch.nn.functional.interpolate(crop_tensor.unsqueeze(0), size=(299,299), mode='bilinear').squeeze(0)
                    real_artifact_crops.append(crop_resized)
                    
        except Exception as e:
            continue

    print(f"Found {len(clean_test_images)} Clean Test Images for generation.")
    print(f"Found {len(real_artifact_crops)} Real Artifact Crops for Ground Truth.")

    if len(clean_test_images) < 10:
        print("Error: Not enough clean test images found!")
        return

    # --- 2. Setup Models ---
    print("Loading AI Models...")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16, safety_checker=None
    ).to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Metrics
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(args.device)
    
    # FID: feature=64 for speed/stability on small sets, or 2048 if you want standard
    # KID: The star of the show for small datasets
    fid_metric = FrechetInceptionDistance(feature=64).to(args.device)
    kid_metric = KernelInceptionDistance(subset_size=args.kid_subset_size).to(args.device)

    # --- 3. Parameter Grid ---
    # Adjust these to find your sweet spot
    scales = [5.0, 7.5, 9.0] 
    strengths = [0.8, 0.9, 1.0]
    
    results = []
    PROMPT = "Histopathological image of lung tissue, hematoxylin and eosin staining, with a fold artifact visible."

    # --- 4. Tuning Loop ---
    total_steps = len(scales) * len(strengths)
    pbar = tqdm(total=total_steps, desc="Hyperparameter Tuning")

    for strength in strengths:
        for scale in scales:
            pbar.set_description(f"Eval: Str={strength} / Scale={scale}")
            
            # Reset Metrics
            fid_metric.reset()
            kid_metric.reset()
            
            # A. Feed Real Data (Reference)
            # Batch process to save VRAM
            batch_size = 50
            for i in range(0, len(real_artifact_crops), batch_size):
                batch = real_artifact_crops[i:i+batch_size]
                batch_t = torch.stack(batch).to(args.device) # [N, 3, 299, 299]
                # Convert to uint8 [0-255] for torchmetrics
                batch_uint8 = (batch_t * 255).byte()
                
                fid_metric.update(batch_uint8, real=True)
                kid_metric.update(batch_uint8, real=True)

            # B. Generate & Feed Fake Data
            epoch_clip = 0.0
            epoch_color = 0.0
            count = 0
            
            for i, row in enumerate(clean_test_images):
                # Load Clean Image
                original_img = Image.open(io.BytesIO(row['image'])).convert("RGB").resize((512,512))
                
                # Borrow a mask (Simulate artifact shape) - Deterministic selection
                # We assume we have real masks. If not, generate procedural (not implemented here for brevity).
                # We use a simple modulo to rotate through available real artifact shapes
                ref_idx = i % len(real_artifact_crops) # Just to pick a shape, we don't have the original mask bytes easily here unless we saved them.
                # Actually, we need the MASK bytes to generate.
                # Let's grab a random row from DF again to get a mask? 
                # For efficiency in this script, let's just grab a random mask from the DF on the fly.
                
                # Quick mask fetch (inefficient but works)
                mask_row = df.iloc[random.randint(0, len(df)-1)]
                while mask_row['mask'] is None or np.array(Image.open(io.BytesIO(mask_row['mask'])).convert("L")).max() == 0:
                     mask_row = df.iloc[random.randint(0, len(df)-1)]
                
                raw_mask = Image.open(io.BytesIO(mask_row['mask'])).convert("L")
                #soft_mask = process_mask_inference(raw_mask) # <--- Soft edges applied here!
                soft_mask = raw_mask.resize((512, 512), Image.Resampling.NEAREST)
                
                # Generate
                generator = torch.Generator(args.device).manual_seed(42 + i)
                with torch.autocast(args.device):
                    out = pipeline(
                        prompt=PROMPT, image=original_img, mask_image=soft_mask,
                        guidance_scale=scale, strength=strength, num_inference_steps=25, generator=generator
                    ).images[0]

                # --- METRIC PREP ---
                # Crop the generated artifact
                m_tensor = transforms.ToTensor()(soft_mask)
                top, left, h, w = get_mask_bbox(m_tensor)
                
                if h > 32 and w > 32:
                    out_tensor = transforms.ToTensor()(out)
                    crop_gen = crop(out_tensor, top, left, h, w)
                    
                    # 1. Resize for FID/KID
                    crop_gen_299 = torch.nn.functional.interpolate(crop_gen.unsqueeze(0), size=(299,299), mode='bilinear').squeeze(0)
                    crop_uint8 = (crop_gen_299 * 255).byte().unsqueeze(0).to(args.device)
                    
                    fid_metric.update(crop_uint8, real=False)
                    kid_metric.update(crop_uint8, real=False)
                    
                    # 2. CLIP (Semantic)
                    # Pass the whole crop or whole image? Usually whole crop focuses attention.
                    # Torchmetrics expects [N, C, H, W] uint8
                    clip_score = clip_metric(crop_uint8, [PROMPT])
                    epoch_clip += clip_score.item()
                    
                    # 3. Color
                    # Compare against random real reference
                    ref_crop = real_artifact_crops[random.randint(0, len(real_artifact_crops)-1)].to(args.device)
                    epoch_color += calculate_color_distance(crop_gen.to(args.device), ref_crop)
                    
                    count += 1
            
            # Compute Final Scores
            try:
                final_fid = fid_metric.compute().item()
                final_kid = kid_metric.compute()[0].item() # Tuple (mean, std)
                avg_clip = epoch_clip / count if count > 0 else 0
                avg_color = epoch_color / count if count > 0 else 0
                
                results.append({
                    "strength": strength,
                    "scale": scale,
                    "kid_masked": round(final_kid, 4), # PRIMARY METRIC
                    "fid_masked": round(final_fid, 2),
                    "clip_score": round(avg_clip, 2),
                    "color_dist": round(avg_color, 2)
                })
                
                print(f"\n[Result] Str:{strength} Scale:{scale} | KID:{final_kid:.4f} FID:{final_fid:.2f}")
            except Exception as e:
                print(f"Metric Eval Failed for {strength}/{scale}: {e}")
            
            pbar.update(1)

    # 5. Save & Display
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)
    
    print("\n" + "="*40)
    print("HYPERPARAMETER TUNING RESULTS (Sorted by KID - Lower is Better)")
    print("="*40)
    print(df_res.sort_values(by="kid_masked"))

if __name__ == "__main__":
    main()