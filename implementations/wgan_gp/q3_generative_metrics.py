# -*- coding: utf-8 -*-
"""Unified finite-sample generative metrics for the Q3 manuscript.

The script uses one deterministic protocol for every method:
  * torchvision Inception-v3 pool features with ImageNet normalization;
  * FID on all 193 training-reference images and at most 1000 generated images;
  * unbiased KID with repeated subsets;
  * generated/generated LPIPS, with within-condition values for nested folders;
  * an unpaired multi-scale edge-statistics discrepancy matching the paper.

Absolute FID values from different implementations should not be compared.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], mode: str):
        self.paths = paths
        if mode == "inception":
            self.transform = transforms.Compose([
                transforms.Resize((299, 299), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        elif mode == "minus_one_one":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            raise ValueError(mode)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def inception_features(paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(ImageDataset(paths, "inception"), batch_size=batch_size, shuffle=False, num_workers=0)
    values = []
    for batch in loader:
        values.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float64)


def fid_from_features(real: np.ndarray, fake: np.ndarray) -> float:
    mu_r, mu_f = real.mean(axis=0), fake.mean(axis=0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    cov_mean, _ = linalg.sqrtm(cov_r @ cov_f, disp=False)
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_r.shape[0]) * 1e-6
        cov_mean = linalg.sqrtm((cov_r + offset) @ (cov_f + offset))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    delta = mu_r - mu_f
    return float(delta @ delta + np.trace(cov_r + cov_f - 2.0 * cov_mean))


def polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def kid_from_features(
    real: np.ndarray,
    fake: np.ndarray,
    subset_size: int,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    subset_size = min(subset_size, len(real), len(fake))
    if subset_size < 2:
        raise ValueError("KID requires at least two samples per subset")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        x = real[rng.choice(len(real), subset_size, replace=False)]
        y = fake[rng.choice(len(fake), subset_size, replace=False)]
        k_xx = polynomial_kernel(x, x)
        k_yy = polynomial_kernel(y, y)
        k_xy = polynomial_kernel(x, y)
        m = subset_size
        mmd = (
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2.0 * k_xy.mean()
        )
        values.append(float(mmd))
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "subset_size": subset_size,
        "repeats": repeats,
    }


def load_lpips_batch(paths: list[Path], device: torch.device) -> torch.Tensor:
    transform = ImageDataset(paths, "minus_one_one").transform
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images).to(device)


@torch.inference_mode()
def lpips_pairs(
    paths: list[Path],
    device: torch.device,
    n_pairs: int,
    seed: int,
    batch_size: int,
) -> dict[str, float | int]:
    import lpips

    if len(paths) < 2:
        raise ValueError("LPIPS requires at least two images")
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(paths), size=(n_pairs, 2))
    equal = pairs[:, 0] == pairs[:, 1]
    pairs[equal, 1] = (pairs[equal, 1] + 1) % len(paths)
    metric = lpips.LPIPS(net="alex").eval().to(device)
    values = []
    for start in range(0, n_pairs, batch_size):
        chunk = pairs[start:start + batch_size]
        left = load_lpips_batch([paths[i] for i in chunk[:, 0]], device)
        right = load_lpips_batch([paths[i] for i in chunk[:, 1]], device)
        values.extend(metric(left, right).flatten().cpu().tolist())
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "n_pairs": len(values),
    }


def conditional_subdirs(root: Path) -> list[Path]:
    subdirs = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if len(image_paths(directory)) >= 2:
            subdirs.append(directory)
    return subdirs


def within_condition_lpips(
    root: Path,
    device: torch.device,
    n_pairs: int,
    seed: int,
    batch_size: int,
) -> dict | None:
    subdirs = conditional_subdirs(root)
    if len(subdirs) < 2:
        return None
    base = n_pairs // len(subdirs)
    remainder = n_pairs % len(subdirs)
    per_condition = {}
    weighted = []
    for index, directory in enumerate(subdirs):
        count = base + (1 if index < remainder else 0)
        result = lpips_pairs(image_paths(directory), device, count, seed + index, batch_size)
        per_condition[directory.name] = result
        weighted.extend([result["mean"]] * count)
    return {
        "mean": float(np.mean(weighted)),
        "n_pairs": n_pairs,
        "conditions": per_condition,
    }


def sobel_structure(batch: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
    kx = batch.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return torch.cat([magnitude, gx.abs(), gy.abs()], dim=1)


@torch.inference_mode()
def structure_statistics(
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    scales: tuple[int, ...] = (32, 16, 8),
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    loader = DataLoader(ImageDataset(paths, "minus_one_one"), batch_size=batch_size, shuffle=False, num_workers=0)
    sums = {scale: None for scale in scales}
    sums_sq = {scale: None for scale in scales}
    count = 0
    for batch in loader:
        struct = sobel_structure(batch.to(device))
        count += len(batch)
        for scale in scales:
            values = F.adaptive_avg_pool2d(struct, (scale, scale)).double()
            current_sum = values.sum(dim=0).cpu().numpy()
            current_sq = values.square().sum(dim=0).cpu().numpy()
            sums[scale] = current_sum if sums[scale] is None else sums[scale] + current_sum
            sums_sq[scale] = current_sq if sums_sq[scale] is None else sums_sq[scale] + current_sq
    result = {}
    for scale in scales:
        mean = sums[scale] / count
        variance = np.maximum(sums_sq[scale] / count - mean ** 2, 0.0)
        result[scale] = (mean, np.sqrt(variance + 1e-8))
    return result


def structure_discrepancy(real_stats: dict, fake_stats: dict, std_weight: float = 0.5) -> dict:
    per_scale = {}
    values = []
    for scale in sorted(real_stats, reverse=True):
        real_mean, real_std = real_stats[scale]
        fake_mean, fake_std = fake_stats[scale]
        mean_l1 = float(np.mean(np.abs(fake_mean - real_mean)))
        std_l1 = float(np.mean(np.abs(fake_std - real_std)))
        total = mean_l1 + std_weight * std_l1
        per_scale[str(scale)] = {"mean_l1": mean_l1, "std_l1": std_l1, "weighted": total}
        values.append(total)
    return {"mean": float(np.mean(values)), "std_weight": std_weight, "per_scale": per_scale}


def evaluate(config: dict, args: argparse.Namespace, device: torch.device) -> dict:
    real_root = Path(config["real"])
    fake_root = Path(config["fake"])
    real_paths = image_paths(real_root)
    fake_paths_all = image_paths(fake_root)
    rng = np.random.default_rng(args.seed)
    if len(fake_paths_all) > args.max_fake:
        indices = sorted(rng.choice(len(fake_paths_all), args.max_fake, replace=False).tolist())
        fake_paths = [fake_paths_all[index] for index in indices]
    else:
        fake_paths = fake_paths_all
    if not real_paths or not fake_paths:
        raise FileNotFoundError(f"Missing images for {config['name']}")

    print(f"[FEATURES] {config['name']}: real={len(real_paths)}, fake={len(fake_paths)}/{len(fake_paths_all)}")
    real_features = inception_features(real_paths, device, args.batch_size)
    fake_features = inception_features(fake_paths, device, args.batch_size)
    real_struct = structure_statistics(real_paths, device, args.batch_size)
    fake_struct = structure_statistics(fake_paths, device, args.batch_size)
    result = {
        "name": config["name"],
        "real": str(real_root),
        "fake": str(fake_root),
        "n_real": len(real_paths),
        "n_fake_available": len(fake_paths_all),
        "n_fake_evaluated": len(fake_paths),
        "fid_torchvision_inception": fid_from_features(real_features, fake_features),
        "kid": kid_from_features(real_features, fake_features, args.kid_subset, args.kid_repeats, args.seed),
        "lpips_pooled": lpips_pairs(fake_paths, device, args.lpips_pairs, args.seed, args.lpips_batch),
        "lpips_within_condition": within_condition_lpips(
            fake_root, device, args.lpips_pairs, args.seed, args.lpips_batch
        ),
        "mses_discrepancy": structure_discrepancy(real_struct, fake_struct),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_fake", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lpips_batch", type=int, default=16)
    parser.add_argument("--lpips_pairs", type=int, default=1000)
    parser.add_argument("--kid_subset", type=int, default=100)
    parser.add_argument("--kid_repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    configs = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config in configs:
        name = config["name"]
        if name in existing and "error" not in existing[name]:
            print(f"[SKIP] {name}")
            continue
        try:
            existing[name] = evaluate(config, args, device)
        except Exception as error:
            existing[name] = {"name": name, "error": repr(error)}
            print(f"[ERROR] {name}: {error!r}")
        output_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]", output_path)


if __name__ == "__main__":
    main()
