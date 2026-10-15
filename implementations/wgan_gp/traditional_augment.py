"""
traditional_augment.py — 传统数据增强：对真实缺陷图做翻转/旋转/颜色扰动

用法:
    python traditional_augment.py --input ../../data/cable_defect --output experiments/E5_traditional_aug/generated --n_aug 5000
"""

import argparse
import os
import random
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to cable_defect dataset (ImageFolder structure)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for augmented images")
    parser.add_argument("--n_aug", type=int, default=5000,
                        help="Number of augmented images to generate")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 收集所有源图片路径（递归，兼容类别子目录结构）
    src_dir = Path(args.input)
    if (src_dir / "defect").exists():
        img_dir = src_dir / "defect"
    else:
        img_dir = src_dir

    exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
    img_paths = sorted(p for e in exts for p in img_dir.rglob(e))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"[augment] Found {len(img_paths)} source images, generating {args.n_aug} augmented.")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 增强管线（随机组合）
    aug_pipeline = T.Compose([
        T.Resize(args.img_size),
        T.CenterCrop(args.img_size),
        T.RandomApply([T.RandomHorizontalFlip(p=1.0)], p=0.5),
        T.RandomApply([T.RandomVerticalFlip(p=1.0)], p=0.5),
        T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.7),
        T.ToTensor(),
    ])

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for i in range(args.n_aug):
        # 随机选一张源图
        src_path = random.choice(img_paths)
        img = Image.open(str(src_path)).convert("RGB")
        img_tensor = aug_pipeline(img)

        save_path = output_dir / f"{i+1:06d}.png"
        save_image(img_tensor, str(save_path))

        if (i + 1) % 500 == 0:
            print(f"[augment] Progress: {i+1}/{args.n_aug}")

    print(f"[augment] Done! {args.n_aug} images saved to {output_dir}")


if __name__ == "__main__":
    main()
