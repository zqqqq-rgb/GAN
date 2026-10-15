"""
fid_lpips_eval.py — FID / LPIPS 质量评估脚本

FID: Frechet Inception Distance —— 生成分布与真实分布的距离
LPIPS: Learned Perceptual Image Patch Similarity —— 生成多样性（pairwise 距离越大越多样）

用于 SCI 论文 3.3 节"生成质量评估"表格。

用法:
    # 单次评估
    python fid_lpips_eval.py --real data/cable_defect/defect --fake experiments/E1/generated --name E1_SP_WGAN_GP

    # 批量评估（JSON 配置文件）
    python fid_lpips_eval.py --config eval_config.json --output results/eval_results.json

依赖:
    pip install lpips torchvision pillow numpy tqdm scipy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm


# ============================================================
#  InceptionV3 特征提取器（用于 FID）
# ============================================================
class InceptionFeatureExtractor:
    """提取 InceptionV3 pool3 层的 2048-d 特征"""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载预训练 InceptionV3
        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.model.fc = torch.nn.Identity()  # 去掉分类头
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract_features(self, dataloader, desc="Extracting features"):
        """返回 (N, 2048) numpy array"""
        features = []
        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            # InceptionV3 期望 299x299 输入
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            feat = self.model(batch)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)


# ============================================================
#  FID 计算
# ============================================================
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet Distance between two multivariate Gaussians"""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # sqrtm of product of cov matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_from_features(feat_real, feat_fake):
    """feat_real: (N_real, D), feat_fake: (N_fake, D) -> FID"""
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)
    mu_fake = np.mean(feat_fake, axis=0)
    sigma_fake = np.cov(feat_fake, rowvar=False)
    return compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)


# ============================================================
#  LPIPS 多样性计算
# ============================================================
def compute_lpips_diversity(fake_dir, n_pairs=1000, device=None):
    """
    在生成图之间随机配对计算 LPIPS 距离。
    平均 LPIPS 越大 → 生成多样性越高（mode collapse 越轻）。
    """
    try:
        import lpips
    except ImportError:
        print("[WARN] lpips not installed. Run: pip install lpips")
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = lpips.LPIPS(net="alex").to(device)

    # 收集图片路径
    fake_dir = Path(fake_dir)
    img_paths = sorted(list(fake_dir.rglob("*.png")) + list(fake_dir.rglob("*.jpg")))

    if len(img_paths) < 2:
        print(f"[WARN] Only {len(img_paths)} images in {fake_dir}, need >= 2 for LPIPS")
        return None

    # 随机选 n_pairs 对
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    n_pairs = min(n_pairs, len(img_paths) * (len(img_paths) - 1) // 2)

    indices = np.random.choice(len(img_paths), size=(n_pairs, 2), replace=True)
    # 确保不是同一张图
    for k in range(n_pairs):
        if indices[k, 0] == indices[k, 1]:
            indices[k, 1] = (indices[k, 0] + 1) % len(img_paths)

    lpips_vals = []
    with torch.no_grad():
        for a, b in tqdm(indices, desc="LPIPS pairwise"):
            img_a = transform(Image.open(str(img_paths[a])).convert("RGB")).unsqueeze(0).to(device)
            img_b = transform(Image.open(str(img_paths[b])).convert("RGB")).unsqueeze(0).to(device)
            d = loss_fn(img_a, img_b).item()
            lpips_vals.append(d)

    lpips_mean = float(np.mean(lpips_vals))
    lpips_std = float(np.std(lpips_vals))
    return {"mean": lpips_mean, "std": lpips_std, "n_pairs": n_pairs}


# ============================================================
#  ImageFolder Dataset
# ============================================================
class ImageFolderDataset(Dataset):
    """简单的图片文件夹 Dataset，返回 (tensor, path)"""
    def __init__(self, root, img_size=299):
        self.root = Path(root)
        self.paths = sorted(
            list(self.root.rglob("*.png")) + list(self.root.rglob("*.jpg")) +
            list(self.root.rglob("*.jpeg")) + list(self.root.rglob("*.bmp"))
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(str(self.paths[idx])).convert("RGB")
        return self.transform(img), str(self.paths[idx])


# ============================================================
#  单次评估
# ============================================================
def evaluate_single(real_dir, fake_dir, name, batch_size=32, device=None):
    """评估一组 (real, fake) → {"name", "fid", "lpips", "n_real", "n_fake"}"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Real: {real_dir}")
    print(f"  Fake: {fake_dir}")

    # ---- 加载数据 ----
    dataset_real = ImageFolderDataset(real_dir, img_size=299)
    dataset_fake = ImageFolderDataset(fake_dir, img_size=299)

    if len(dataset_real) == 0:
        raise FileNotFoundError(f"No images found in {real_dir}")
    if len(dataset_fake) == 0:
        raise FileNotFoundError(f"No images found in {fake_dir}")

    print(f"  Real images: {len(dataset_real)}, Fake images: {len(dataset_fake)}")

    loader_real = DataLoader(dataset_real, batch_size=batch_size, num_workers=0, shuffle=False)
    loader_fake = DataLoader(dataset_fake, batch_size=batch_size, num_workers=0, shuffle=False)

    # ---- FID ----
    extractor = InceptionFeatureExtractor(device=device)
    feat_real = extractor.extract_features(loader_real, desc="FID: real features")
    feat_fake = extractor.extract_features(loader_fake, desc="FID: fake features")
    fid = compute_fid_from_features(feat_real, feat_fake)
    print(f"  FID: {fid:.2f}")

    # ---- LPIPS ----
    lpips_result = compute_lpips_diversity(fake_dir, n_pairs=min(1000, len(dataset_fake)), device=device)
    if lpips_result:
        print(f"  LPIPS: {lpips_result['mean']:.4f} ± {lpips_result['std']:.4f}")
    else:
        print(f"  LPIPS: N/A")

    result = {
        "name": name,
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "n_real": len(dataset_real),
        "n_fake": len(dataset_fake),
        "fid": round(fid, 2),
        "lpips": lpips_result,
    }
    return result


# ============================================================
#  批量评估（JSON 配置文件）
# ============================================================
def evaluate_batch(config_path, output_path=None, device=None):
    """config.json: [{"name":..., "real":..., "fake":...}, ...]"""
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    results = []
    for cfg in configs:
        try:
            r = evaluate_single(
                real_dir=cfg["real"],
                fake_dir=cfg["fake"],
                name=cfg["name"],
                device=device,
            )
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {cfg['name']}: {e}")
            results.append({"name": cfg["name"], "error": str(e)})

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    # 打印汇总表
    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'FID':>8} {'LPIPS':>8} {'#Real':>8} {'#Fake':>8}")
    print("-" * 70)
    for r in results:
        fid_str = f"{r['fid']:.2f}" if "fid" in r else "ERR"
        lpips_str = f"{r['lpips']['mean']:.4f}" if r.get("lpips") else "N/A"
        print(f"{r['name']:<30} {fid_str:>8} {lpips_str:>8} {r.get('n_real','?'):>8} {r.get('n_fake','?'):>8}")
    print("=" * 70)

    return results


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FID & LPIPS quality evaluation for GAN-generated images")
    sub = parser.add_subparsers(dest="mode", required=True)

    # 模式 1: 单次评估
    p_single = sub.add_parser("single", help="Evaluate a single (real, fake) pair")
    p_single.add_argument("--real", type=str, required=True)
    p_single.add_argument("--fake", type=str, required=True)
    p_single.add_argument("--name", type=str, default="experiment")
    p_single.add_argument("--output", type=str, default=None, help="Save result JSON")

    # 模式 2: 批量评估（配置文件）
    p_batch = sub.add_parser("batch", help="Batch evaluate from JSON config")
    p_batch.add_argument("--config", type=str, required=True,
                         help="JSON config file: [{'name':..., 'real':..., 'fake':...}, ...]")
    p_batch.add_argument("--output", type=str, default=None,
                         help="Save results JSON")

    args = parser.parse_args()

    if args.mode == "single":
        r = evaluate_single(args.real, args.fake, args.name)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"Result saved to {args.output}")
        # 打印结果
        print(f"\n{'='*40}")
        print(f"Name:  {r['name']}")
        print(f"FID:   {r['fid']:.2f}")
        print(f"LPIPS: {r['lpips']['mean']:.4f} ± {r['lpips']['std']:.4f}" if r.get("lpips") else "LPIPS: N/A")
        print(f"#Real: {r['n_real']}, #Fake: {r['n_fake']}")

    elif args.mode == "batch":
        evaluate_batch(args.config, args.output)


if __name__ == "__main__":
    main()
