import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import io
import gc
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPModel

# --- GLOBAL PLIP INSTANCE TO PREVENT OOM DEEPCOPY ---
GLOBAL_PLIP_VISION = None


class PLIPFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        global GLOBAL_PLIP_VISION
        x = x.float() / 255.0
        x = F.resize(x, [224, 224], antialias=True)
        x = F.normalize(
            x,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return GLOBAL_PLIP_VISION(pixel_values=x).pooler_output


# --- Helper: Bounding Box Crop ---
def get_bbox(mask_pil):
    return mask_pil.getbbox()


# --- Helper: Masked SSIM ---
def calculate_masked_ssim(clean_batch, gen_batch, mask_batch):
    b_size = clean_batch.size(0)
    scores = []

    for i in range(b_size):
        clean_np = clean_batch[i].permute(1, 2, 0).cpu().numpy()
        gen_np = gen_batch[i].permute(1, 2, 0).cpu().numpy()
        mask_np = mask_batch[i, 0].cpu().numpy()

        _, diff = ssim(clean_np, gen_np, channel_axis=-1, full=True, data_range=255)
        diff_gray = diff.mean(axis=-1)

        valid_pixels = diff_gray[mask_np > 0.5]
        if len(valid_pixels) > 0:
            scores.append(valid_pixels.mean())

    return scores


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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Mathematical calculation of FID given means and covariances."""
    diff = mu1 - mu2
    
    # Zmiana: usunięto disp=False, co naprawia DeprecationWarning w nowym SciPy
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def bootstrap_fid(real_feats, fake_feats, n_bootstraps=50):
    """Calculates FID Std and 95% CI by bootstrapping on extracted features."""
    print("    -> Obliczanie macierzy kowariancji dla obrazów referencyjnych...")
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    
    n_fake = len(fake_feats)
    bootstrap_scores = []
    
    print(f"    -> Bootstrapping FID ({n_bootstraps} iteracji). To potrwa kilka minut na CPU...")
    
    # Dodany tqdm, żeby pokazywać czas do końca i pasek postępu!
    for _ in tqdm(range(n_bootstraps), desc="FID Bootstrap", leave=False):
        idx = np.random.choice(n_fake, size=n_fake, replace=True)
        sampled_fake = fake_feats[idx]
        
        mu_fake = np.mean(sampled_fake, axis=0)
        sigma_fake = np.cov(sampled_fake, rowvar=False)
        
        score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        bootstrap_scores.append(score)
        
    bootstrap_scores = np.array(bootstrap_scores)
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)
    ci_low = np.percentile(bootstrap_scores, 2.5)
    ci_high = np.percentile(bootstrap_scores, 97.5)
    
    return mean, std, (ci_low, ci_high)


# --- Datasets ---
class RealArtifactDataset(Dataset):
    def __init__(self, parquet_path, transform=None):
        print(f"Loading reference data from {parquet_path}...")
        try:
            df = pd.read_parquet(parquet_path, columns=["image", "mask", "source"])
            df = df[df["source"] != "synthetic"]
        except ValueError:
            df = pd.read_parquet(parquet_path, columns=["image", "mask"])

        images = df["image"].tolist()
        masks = df["mask"].tolist()

        del df
        gc.collect()

        self.data = []
        for img_bytes, m_bytes in tqdm(
            zip(images, masks), total=len(images), desc="Preparing Real Crops"
        ):
            if m_bytes is None or img_bytes is None:
                continue
            try:
                with Image.open(io.BytesIO(m_bytes)) as m_img:
                    if m_img.convert("L").getextrema() == (0, 0):
                        continue
                self.data.append((img_bytes, m_bytes))
            except Exception:
                continue

        print(f"Found {len(self.data)} usable REAL artifact crops.")
        del images
        del masks
        gc.collect()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_bytes, mask_bytes = self.data[idx]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        bbox = get_bbox(mask)
        if bbox:
            image = image.crop(bbox)
        if self.transform:
            image = self.transform(image)
        return image


class GeneratedDataset(Dataset):
    def __init__(self, folder_path, clean_source=None, transform=None, calc_ssim=False):
        self.folder_path = folder_path
        self.mask_folder = os.path.join(folder_path, "masks")
        self.calc_ssim = calc_ssim
        self.transform = transform

        self.files = [
            f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg"))
        ]
        self.has_masks = os.path.exists(self.mask_folder)

        self.clean_mode = None
        self.clean_dict = {}
        self.found_cleans = set()

        if self.calc_ssim and clean_source:
            if os.path.isdir(clean_source):
                self.clean_mode = "folder"
                self.clean_dir = clean_source
                print(f"SSIM Mode: Linking clean images from folder: {clean_source}")
                self._check_clean_in_folder()

            elif clean_source.endswith(".parquet"):
                self.clean_mode = "parquet"
                print(
                    f"SSIM Mode: Extracting clean images from massive Parquet: {clean_source}..."
                )
                self._extract_clean_from_parquet(clean_source)
            else:
                self.calc_ssim = False
        elif self.calc_ssim and not clean_source:
            self.calc_ssim = False

    def _check_clean_in_folder(self):
        for f in self.files:
            if os.path.exists(os.path.join(self.clean_dir, f)):
                self.found_cleans.add(f)

    def _extract_clean_from_parquet(self, parquet_path):
        try:
            df = pd.read_parquet(parquet_path, columns=["filename", "image"])
        except Exception as e:
            raise RuntimeError(
                f"Failed to load parquet for clean images. Ensure 'filename' column exists! Error: {e}"
            )

        needed_filenames = set(self.files)
        needed_basenames = {os.path.splitext(f)[0] for f in self.files}

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Mapping pristine images to RAM"
        ):
            row_id = str(row["filename"])
            if row_id in needed_filenames:
                self.clean_dict[row_id] = row["image"]
                self.found_cleans.add(row_id)
            elif row_id in needed_basenames:
                full_filename = next(
                    f for f in self.files if os.path.splitext(f)[0] == row_id
                )
                self.clean_dict[full_filename] = row["image"]
                self.found_cleans.add(full_filename)

            if len(self.found_cleans) == len(self.files):
                break

        del df
        gc.collect()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.folder_path, filename)

        image = Image.open(img_path).convert("RGB")
        clean = image.copy()
        has_clean = False

        if self.calc_ssim and filename in self.found_cleans:
            try:
                if self.clean_mode == "folder":
                    clean = Image.open(os.path.join(self.clean_dir, filename)).convert(
                        "RGB"
                    )
                    has_clean = True
                elif self.clean_mode == "parquet":
                    clean = Image.open(io.BytesIO(self.clean_dict[filename])).convert(
                        "RGB"
                    )
                    has_clean = True
            except Exception:
                has_clean = False

        mask = Image.new("L", image.size, 255)
        if self.has_masks:
            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(self.mask_folder, f"{base_name}_mask.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.mask_folder, filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")

        bbox = get_bbox(mask)
        if bbox:
            image = image.crop(bbox)
            clean = clean.crop(bbox)
            mask = mask.crop(bbox)

        if self.transform:
            image = self.transform(image)
            clean = self.transform(clean)
            mask_tensor = transforms.Resize((299, 299))(mask)
            mask_tensor = transforms.ToTensor()(mask_tensor)

        return image, clean, mask_tensor, has_clean


def main():
    # --- LOCK RANDOM SEEDS ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_source", type=str, required=True)
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
    parser.add_argument("--calc_ssim", action="store_true")
    parser.add_argument("--clean_source", type=str, default=None)
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics_log.txt",
        help="File to append results",
    )
    args = parser.parse_args()

    USE_TEXT_PROMPT = args.artifact_type in ["folds", "precipitates"]
    PROMPT_TEXT = f"Histopathological image of lung tissue, hematoxylin and eosin staining, with {args.artifact_type} artifact."

    transform = transforms.Compose(
        [transforms.Resize((299, 299)), transforms.ToTensor()]
    )

    real_set = RealArtifactDataset(args.real_source, transform=transform)
    gen_set = GeneratedDataset(
        args.gen_folder,
        clean_source=args.clean_source,
        transform=transform,
        calc_ssim=args.calc_ssim,
    )

    real_loader = DataLoader(real_set, batch_size=32, shuffle=True, num_workers=0)
    gen_loader = DataLoader(gen_set, batch_size=32, shuffle=False, num_workers=0)

    print("\nLoading Metrics and Models...")
    global GLOBAL_PLIP_VISION
    GLOBAL_PLIP_VISION = CLIPModel.from_pretrained(
        "vinid/plip", use_safetensors=True
    ).vision_model.to(args.device)
    GLOBAL_PLIP_VISION.eval()

    fid = FrechetInceptionDistance(feature=2048).to(args.device)

    # Współdzielimy Inception między FID i KID, żeby zaoszczędzić VRAM
    shared_inception = fid.inception
    kid = KernelInceptionDistance(feature=shared_inception, subset_size=50).to(args.device)
    kid_base = KernelInceptionDistance(feature=shared_inception, subset_size=50).to(args.device)

    plip_extractor = PLIPFeatureExtractor().to(args.device)
    kid_plip = KernelInceptionDistance(feature=plip_extractor, subset_size=50).to(args.device)
    kid_plip_base = KernelInceptionDistance(feature=plip_extractor, subset_size=50).to(args.device)

    if USE_TEXT_PROMPT:
        clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
            args.device
        )

    # --- PROCESS REAL ---
    print("\nProcessing Real Artifacts (Baselines & Profiles)...")
    real_histograms = np.zeros((3, 256))
    real_count = 0

    with torch.no_grad():
        for batch in tqdm(real_loader):
            batch_u8 = (batch * 255).to(dtype=torch.uint8).to(args.device)

            fid.update(batch_u8, real=True)
            kid.update(batch_u8, real=True)
            kid_plip.update(batch_u8, real=True)

            half = len(batch_u8) // 2
            if half > 0:
                kid_base.update(batch_u8[:half], real=True)
                kid_base.update(batch_u8[half:], real=False)
                kid_plip_base.update(batch_u8[:half], real=True)
                kid_plip_base.update(batch_u8[half:], real=False)

            for i in range(len(batch)):
                np_img = batch_u8[i].cpu().numpy()
                for c in range(3):
                    hist, _ = np.histogram(np_img[c], bins=256, range=(0, 256))
                    real_histograms[c] += hist
                real_count += 1

    real_histograms = real_histograms / real_count
    for c in range(3):
        real_histograms[c] = real_histograms[c] / real_histograms[c].sum()

    # --- PROCESS GENERATED ---
    print("\nProcessing Generated Artifacts...")

    all_color_scores = []
    all_ssim_scores = []
    all_clip_scores = []

    with torch.no_grad():
        for gen_images, clean_images, masks, has_clean in tqdm(gen_loader):
            batch_u8 = (gen_images * 255).to(dtype=torch.uint8).to(args.device)

            fid.update(batch_u8, real=False)
            kid.update(batch_u8, real=False)
            kid_plip.update(batch_u8, real=False)

            for i in range(len(batch_u8)):
                np_img = batch_u8[i].cpu().numpy()
                d = 0
                for c in range(3):
                    hist_gen, _ = np.histogram(
                        np_img[c], bins=256, range=(0, 256), density=True
                    )
                    d += wasserstein_distance(
                        np.arange(256), np.arange(256), hist_gen, real_histograms[c]
                    )
                all_color_scores.append(d / 3.0)

            if args.calc_ssim:
                valid_ssim_indices = has_clean.nonzero(as_tuple=True)[0]
                if len(valid_ssim_indices) > 0:
                    clean_u8 = (clean_images[valid_ssim_indices] * 255).to(
                        dtype=torch.uint8
                    )
                    batch_u8_cpu = batch_u8[valid_ssim_indices].cpu()
                    valid_masks = masks[valid_ssim_indices]

                    batch_ssim_scores = calculate_masked_ssim(
                        clean_u8, batch_u8_cpu, valid_masks
                    )
                    all_ssim_scores.extend(batch_ssim_scores)

            if USE_TEXT_PROMPT:
                for i in range(len(batch_u8)):
                    clip_val = clip_scorer(batch_u8[i:i+1], [PROMPT_TEXT])
                    all_clip_scores.append(clip_val.item())

    # --- COMPUTE MEANS, STDS, AND CONFIDENCE INTERVALS ---
    print("\nCalculating Final Statistics & Bootstrapping Confidence Intervals...")

    # FID Bootstrap Execution
    try:
        # 1. Najpierw liczymy Prawdziwy, dokładny FID całego zbioru (to zwróci ~77.22)
        true_fid = fid.compute().item()
        
        # 2. Pobieramy cechy z bufora
        real_feats_np = torch.cat([f.cpu() for f in kid.real_features], dim=0).numpy()
        fake_feats_np = torch.cat([f.cpu() for f in kid.fake_features], dim=0).numpy()
        
        # 3. Odpalamy bootstrap, ale bierzemy z niego TYLKO odchylenie standardowe (f_s) i granice
        boot_m, f_s, boot_ci = bootstrap_fid(real_feats_np, fake_feats_np, n_bootstraps=50)
        
        # 4. Aplikujemy "szerokość" przedziału ufności do prawdziwego wyniku
        ci_lower = true_fid - (boot_m - boot_ci[0])
        ci_upper = true_fid + (boot_ci[1] - boot_m)
        
        fid_str = f"{true_fid:.4f} ± {f_s:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
    except Exception as e:
        try:
            fid_val = fid.compute().item()
            fid_str = f"{fid_val:.4f} (CI Calculation failed: {e})"
        except:
            fid_str = "N/A (Dataset too small)"

    # KID calculations
    k_m, k_s = kid.compute()
    kb_m, kb_s = kid_base.compute()
    kp_m, kp_s = kid_plip.compute()
    kpb_m, kpb_s = kid_plip_base.compute()

    kid_str = f"{k_m.item():.5f} ± {k_s.item():.5f} (95% CI: [{k_m.item() - 1.96*k_s.item():.5f}, {k_m.item() + 1.96*k_s.item():.5f}])"
    kid_plip_str = f"{kp_m.item():.5f} ± {kp_s.item():.5f} (95% CI: [{kp_m.item() - 1.96*kp_s.item():.5f}, {kp_m.item() + 1.96*kp_s.item():.5f}])"

    # Per-Image Metric CIs
    color_m, color_s, color_ci = calculate_metric_ci(all_color_scores)
    
    if args.calc_ssim and len(all_ssim_scores) > 0:
        ssim_m, ssim_s, ssim_ci = calculate_metric_ci(all_ssim_scores)
        ssim_str = f"{ssim_m:.4f} ± {ssim_s:.4f} (95% CI: [{ssim_ci[0]:.4f}, {ssim_ci[1]:.4f}])"
    else:
        ssim_str = "Skipped / No clean maps found"

    if USE_TEXT_PROMPT and len(all_clip_scores) > 0:
        clip_m, clip_s, clip_ci = calculate_metric_ci(all_clip_scores)
        clip_str = f"{clip_m:.4f} ± {clip_s:.4f} (95% CI: [{clip_ci[0]:.4f}, {clip_ci[1]:.4f}])"
    else:
        clip_str = "Skipped (Procedural generation)"

    # --- FORMAT AND LOG REPORT ---
    report = f"""
==================================================
 FINAL QUALITY REPORT: {args.artifact_type.upper()}
 Generated Folder: {args.gen_folder}
==================================================
FID Score:             {fid_str}
KID (Inception):       {kid_str}
  └ Baseline:          {kb_m.item():.5f} ± {kb_s.item():.5f}
KID (PLIP):            {kid_plip_str}
  └ Baseline:          {kpb_m.item():.5f} ± {kpb_s.item():.5f}
Wasserstein Color:     {color_m:.4f} ± {color_s:.4f} (95% CI: [{color_ci[0]:.4f}, {color_ci[1]:.4f}])
Masked SSIM:           {ssim_str}
Standard CLIP Score:   {clip_str}
==================================================
"""

    print(report)

    with open(args.output_file, "a") as f:
        f.write(report + "\n")
    print(f"-> Results successfully appended to {args.output_file}")


if __name__ == "__main__":
    main()