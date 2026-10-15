"""Generate conditional samples from trained NEU-DET E6 cWGAN-GP checkpoint.
Class mapping (ImageFolder sorted): 0:crazing 1:inclusion 2:patches 3:pitted_surface 4:rolled-in_scale 5:scratches
Output: data/neu_det/yolo/images/gen_E6_neu/<cls>_<idx>.jpg (1000 imgs, ~167/class)
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import argparse
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=str(_release_root / 'data/neu_det/ckpt_E6_neu/ckpt_latest.pth'))
parser.add_argument("--out_dir", type=str, default=str(_release_root / 'data/neu_det/yolo/images/gen_E6_neu'))
parser.add_argument("--n_total", type=int, default=1000)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=6)
parser.add_argument("--class_embed_dim", type=int, default=64)
parser.add_argument("--batch", type=int, default=50)
opt = parser.parse_args()

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
os.makedirs(opt.out_dir, exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = opt.img_size // 64
        self.class_embedding = nn.Embedding(opt.num_classes, opt.class_embed_dim)
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.class_embed_dim, 512 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512, 0.8), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256, 0.8), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128, 0.8), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64, 0.8), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32, 0.8), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16, 0.8), nn.ReLU(True),
            nn.Conv2d(16, opt.channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z, labels):
        ce = self.class_embedding(labels)
        out = self.l1(torch.cat([z, ce], dim=1))
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)


device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator().to(device)
ck = torch.load(opt.ckpt, map_location=device)
G.load_state_dict(ck["generator_state_dict"])
G.eval()
print(f"Loaded ckpt @ epoch {ck['epoch']}, batches {ck.get('batches_done')}")

# per-class counts
per = opt.n_total // opt.num_classes
counts = {c: per for c in range(opt.num_classes)}
for i in range(opt.n_total - per * opt.num_classes):
    counts[i] += 1

idx = 0
with torch.no_grad():
    for c, n in counts.items():
        done = 0
        while done < n:
            b = min(opt.batch, n - done)
            z = torch.randn(b, opt.latent_dim, device=device)
            lab = torch.full((b,), c, dtype=torch.long, device=device)
            imgs = G(z, lab)  # [-1,1]
            imgs = ((imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            for k in range(b):
                arr = np.transpose(imgs[k], (1, 2, 0))
                # resize to 200x200 to match NEU-DET resolution
                im = Image.fromarray(arr).resize((200, 200), Image.BILINEAR)
                im.save(os.path.join(opt.out_dir, f"{CLASSES[c]}_{idx:05d}.jpg"), quality=95)
                idx += 1
            done += b
        print(f"[GEN] {CLASSES[c]}: {n}")
print(f"Total generated: {idx} -> {opt.out_dir}")
