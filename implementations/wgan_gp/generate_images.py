"""
generate_images.py — 从训练好的 WGAN-GP / DCGAN checkpoint 批量生成图片

用法:
    # WGAN-GP 生成
    python generate_images.py --model wgan_gp --ckpt checkpoint_final.pth --n_images 5000 --output experiments/E1/generated

    # DCGAN 生成
    python generate_images.py --model dcgan --ckpt dcgan_checkpoint.pth --n_images 5000 --output experiments/E4/generated --img_size 256
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# ============================================================
#  WGAN-GP Generator (复制 wgan_gp.py 中的架构)
# ============================================================
class WGAN_GP_Generator(nn.Module):
    """512x512 Generator with 6 upsampling stages (8->16->32->64->128->256->512)"""
    def __init__(self, latent_dim=128, img_size=512, channels=3):
        super().__init__()
        self.init_size = img_size // 64  # 8

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),       # 8 -> 16
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),       # 16 -> 32
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),       # 32 -> 64
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),       # 64 -> 128
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),       # 128 -> 256
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),       # 256 -> 512
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)


# ============================================================
#  DCGAN Generator（适配 128/256 分辨率）
# ============================================================
class DCGAN_Generator(nn.Module):
    """DCGAN Generator, 输出分辨率由 img_size 控制（64/128/256）"""
    def __init__(self, latent_dim=100, img_size=256, channels=3):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        layers = []
        in_ch = 128
        scale = self.init_size
        while scale < img_size:
            layers += [
                nn.BatchNorm2d(in_ch),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, max(in_ch // 2, 64), 3, stride=1, padding=1),
                nn.BatchNorm2d(max(in_ch // 2, 64), 0.8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = max(in_ch // 2, 64)
            scale *= 2

        layers += [nn.Conv2d(in_ch, channels, 3, stride=1, padding=1), nn.Tanh()]
        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


# ============================================================
#  生成核心逻辑
# ============================================================
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] device={device}, model={args.model}, n_images={args.n_images}")

    # ---- 构建模型 ----
    if args.model == "wgan_gp":
        netG = WGAN_GP_Generator(
            latent_dim=args.latent_dim, img_size=args.img_size, channels=args.channels
        )
    elif args.model == "dcgan":
        netG = DCGAN_Generator(
            latent_dim=args.latent_dim, img_size=args.img_size, channels=args.channels
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # ---- 加载权重 ----
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location=device)
    if "generator_state_dict" in checkpoint:
        netG.load_state_dict(checkpoint["generator_state_dict"])
        print(f"[generate] Loaded from '{args.ckpt}', epoch={checkpoint.get('epoch', '?')}")
    else:
        # 直接是 state_dict
        netG.load_state_dict(checkpoint)
        print(f"[generate] Loaded raw state_dict from '{args.ckpt}'")

    netG.to(device)
    netG.eval()

    # ---- 输出目录 ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 检查已生成数量（断点续传） ----
    existing = list(output_dir.glob("*.png"))
    start_idx = len(existing)
    if start_idx >= args.n_images:
        print(f"[generate] Already have {start_idx} images >= {args.n_images}, skipping.")
        return str(output_dir)
    print(f"[generate] Resuming from index {start_idx}, target {args.n_images}")

    # ---- 批量生成 ----
    batch_size = args.batch_size
    total = args.n_images
    z_dim = args.latent_dim

    with torch.no_grad():
        for idx in range(start_idx, total, batch_size):
            cur_batch = min(batch_size, total - idx)
            z = torch.randn(cur_batch, z_dim, device=device)
            imgs = netG(z)

            for j in range(cur_batch):
                img_path = output_dir / f"{idx + j + 1:06d}.png"
                save_image(imgs[j:j+1].data, str(img_path), normalize=True)

            if (idx + cur_batch) % 500 == 0 or (idx + cur_batch) >= total:
                print(f"[generate] Progress: {idx + cur_batch}/{total}")

    print(f"[generate] Done! {total} images saved to {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate images from trained GAN checkpoint")
    parser.add_argument("--model", type=str, default="wgan_gp",
                        choices=["wgan_gp", "dcgan"],
                        help="Model architecture type")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--n_images", type=int, default=5000,
                        help="Total number of images to generate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for generated images")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image resolution (512 for wgan_gp, 64/128/256 for dcgan)")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent dim: 128 for wgan_gp, 100 for dcgan")

    args = parser.parse_args()
    output = generate(args)
    print(f"OUTPUT_DIR={output}")

if __name__ == "__main__":
    main()
