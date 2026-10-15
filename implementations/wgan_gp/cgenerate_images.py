"""
Conditional image generation script.

Generates images by defect class using a trained conditional SP-WGAN-GP.

Usage:
  # Generate 1250 per class (5000 total) from final checkpoint
  python cgenerate_images.py --checkpoint checkpoint_final.pth --n_per_class 1250

  # Generate 250 per class (1000 total)
  python cgenerate_images.py --checkpoint checkpoint_final.pth --n_per_class 250

  # Generate specific class only
  python cgenerate_images.py --checkpoint checkpoint_final.pth --n_per_class 1000 --class_idx 0
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="generator checkpoint path")
parser.add_argument("--n_per_class", type=int, default=1250, help="number of images to generate per class")
parser.add_argument("--class_idx", type=int, default=-1, help="generate only this class (-1 = all classes)")
parser.add_argument("--output", type=str, default="generated_by_class", help="output directory")
parser.add_argument("--latent_dim", type=int, default=128, help="latent dimension (must match training)")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes (must match training)")
parser.add_argument("--class_embed_dim", type=int, default=64, help="class embedding dim (must match training)")
parser.add_argument("--img_size", type=int, default=512, help="image size (must match training)")
parser.add_argument("--channels", type=int, default=3, help="image channels")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for generation")
parser.add_argument("--dataset", choices=["cable", "neu"], default="cable",
                    help="class-name set used for output subdirectories")
args = parser.parse_args()


# ============================================================
#  Generator (must match cwgan_gp.py architecture exactly)
# ============================================================
class Generator(nn.Module):
    """Conditional 512x512 Generator.

    Input: noise z [B, latent_dim] + class labels [B]
    The class label is embedded and concatenated with z before the linear layer.
    """
    def __init__(self, latent_dim=128, num_classes=4, class_embed_dim=64, img_size=512, channels=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 64  # 512//64 = 8
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        # latent_dim + class_embed_dim -> feature map
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + class_embed_dim, 512 * self.init_size ** 2)
        )

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

    def forward(self, z, labels):
        # Class embedding
        class_emb = self.class_embedding(labels)  # [B, class_embed_dim]
        # Concatenate noise + class embedding
        z_concat = torch.cat([z, class_emb], dim=1)  # [B, latent_dim + class_embed_dim]
        out = self.l1(z_concat)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Class names must match ImageFolder's alphabetical class order.
if args.dataset == "neu":
    CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    CLASS_NAMES_CN = ["龟裂", "夹杂", "斑块", "麻面", "辊印", "划痕"]
else:
    CLASS_NAMES = ["dent", "scratch", "deformation", "bulge"]
    CLASS_NAMES_CN = ["凹点", "划痕", "形变", "鼓包"]

# Setup
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Initialize generator
generator = Generator(
    latent_dim=args.latent_dim,
    num_classes=args.num_classes,
    class_embed_dim=args.class_embed_dim,
    img_size=args.img_size,
    channels=args.channels,
)
generator.to(device)

# Load checkpoint
print("Loading checkpoint: %s" % args.checkpoint)
checkpoint = torch.load(args.checkpoint, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()
print("Generator loaded successfully!")

# Create output directories
os.makedirs(args.output, exist_ok=True)
classes_to_generate = range(args.num_classes) if args.class_idx < 0 else [args.class_idx]

total_generated = 0
for cls_idx in classes_to_generate:
    cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "class_%d" % cls_idx
    cls_dir = os.path.join(args.output, cls_name)
    os.makedirs(cls_dir, exist_ok=True)

    n_generated = 0
    while n_generated < args.n_per_class:
        batch = min(args.batch_size, args.n_per_class - n_generated)
        z = torch.randn(batch, args.latent_dim, device=device)
        labels = torch.full((batch,), cls_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            gen_imgs = generator(z, labels)

        for j in range(batch):
            img_path = os.path.join(cls_dir, "%s_%05d.png" % (cls_name, n_generated + j))
            save_image(gen_imgs[j], img_path, normalize=True)

        n_generated += batch
        total_generated += batch

    print("  %s (%s): %d images -> %s" % (cls_name, CLASS_NAMES_CN[cls_idx], n_generated, cls_dir))

print("\nGeneration complete!")
print("Total: %d images in %d classes" % (total_generated, len(list(classes_to_generate))))
print("Output: %s" % args.output)
