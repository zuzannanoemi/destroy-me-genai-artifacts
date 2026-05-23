import argparse
import torch
import numpy as np
import os
import random
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPProcessor, CLIPModel

# --- Helper: Bootstrapping Stats ---
def calculate_metric_ci(scores_list, n_bootstraps=1000):
    """Calculates Mean, Std, and 95% CI for list-based metrics via bootstrapping."""
    scores = np.array(scores_list)
    mean = np.mean(scores)
    std = np.std(scores)
    
    if len(scores) < 2:
        return mean, std, (mean, mean)
        
    boot_means = []
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(boot_sample))
        
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    
    return mean, std, (ci_low, ci_high)


def main():
    # --- LOCK RANDOM SEEDS ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_folder", type=str, required=True)
    parser.add_argument(
        "--artifact_type",
        type=str,
        choices=["folds", "precipitates", "marker", "dust"],
        default="folds",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="plip_scores_log.txt",
        help="File to append results",
    )
    args = parser.parse_args()

    PROMPT_TEXT = f"Histopathological image of lung tissue, hematoxylin and eosin staining, with {args.artifact_type} artifact."

    print("\nLoading PLIP Model and Processor for Text-Image Semantic Grounding...")
    # Ładujemy model i procesor natywnie z HuggingFace
    plip_model = CLIPModel.from_pretrained("vinid/plip").to(args.device)
    plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
    plip_model.eval()

    # Znajdujemy wszystkie wygenerowane obrazy
    files = [f for f in os.listdir(args.gen_folder) if f.lower().endswith((".png", ".jpg"))]
    
    print(f"\nProcessing {len(files)} Generated Artifacts ({args.artifact_type.upper()})...")

    all_plip_scores = []

    with torch.no_grad():
        for filename in tqdm(files, desc="PLIP Evaluation"):
            img_path = os.path.join(args.gen_folder, filename)
            image = Image.open(img_path).convert("RGB")


            inputs = plip_processor(
                text=[PROMPT_TEXT], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            # Przejście przez model
            outputs = plip_model(**inputs)
            
            img_embeds = outputs.image_embeds
            txt_embeds = outputs.text_embeds

            # Normalizacja wektorów i obliczenie podobieństwa cosinusowego
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
            cos_sim = torch.matmul(img_embeds, txt_embeds.t()).item()
            
            # torchmetrics CLIPScore stosuje mnożnik x100 i ucina wartości poniżej 0
            score = max(100.0 * cos_sim, 0.0)
            all_plip_scores.append(score)

    # --- COMPUTE MEANS, STDS, AND CONFIDENCE INTERVALS ---
    print("\nCalculating Final Statistics & Bootstrapping Confidence Intervals...")
    
    plip_m, plip_s, plip_ci = calculate_metric_ci(all_plip_scores)
    plip_str = f"{plip_m:.4f} ± {plip_s:.4f} (95% CI: [{plip_ci[0]:.4f}, {plip_ci[1]:.4f}])"

    # --- FORMAT AND LOG REPORT ---
    report = f"""
==================================================
 PLIP SEMANTIC GROUNDING REPORT: {args.artifact_type.upper()}
 Generated Folder: {args.gen_folder}
==================================================
Prompt used: "{PROMPT_TEXT}"
PLIP Score:  {plip_str}
==================================================
"""

    print(report)

    with open(args.output_file, "a") as f:
        f.write(report + "\n")
    print(f"-> Results successfully appended to {args.output_file}")


if __name__ == "__main__":
    main()