"""
Generate images with text-guided conditional SP-WGAN-GP (E7).

Supports two modes:
  1. By class: --by_class --n_per_class 1250
     Generates images for all 4 defect classes using predefined descriptions.
  2. By text: --text "a cable surface with a red scratch defect" --n 100
     Generates images from arbitrary text descriptions.

Usage:
  python cgenerate_images_clip.py --checkpoint checkpoint_e7_final.pth --by_class --n_per_class 1250 --output ../experiments/E7/generated
  python cgenerate_images_clip.py --checkpoint checkpoint_e7_final.pth --text "a cable surface with a large dent defect" --n 100 --output ../experiments/E7/custom
"""

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image
import clip

# ------------
#  Args
# ------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="E7 checkpoint path")
parser.add_argument("--output", type=str, default="../experiments/E7/generated", help="output directory")
parser.add_argument("--latent_dim", type=int, default=128, help="latent dimension")
parser.add_argument("--img_size", type=int, default=512, help="image size")
parser.add_argument("--channels", type=int, default=3, help="image channels")
parser.add_argument("--text_emb_dim", type=int, default=512, help="CLIP text embedding dim")
parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant")
parser.add_argument("--n_per_class", type=int, default=1250, help="images per class (by_class mode)")
parser.add_argument("--n", type=int, default=100, help="total images (text mode)")
parser.add_argument("--by_class", action="store_true", help="generate by class")
parser.add_argument("--text", type=str, default="", help="custom text description")
parser.add_argument("--batch_size", type=int, default=25, help="batch size for generation")
parser.add_argument("--seed", type=int, default=42, help="fixed latent sampling seed")
parser.add_argument("--dataset", choices=["cable", "neu", "gc10"], default="cable",
                    help="predefined class-description set used by --by_class")
opt = parser.parse_args()
print(opt)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Inline Generator (no import from training script)
# ------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 64

        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.text_emb_dim, 512 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, text_emb):
        z_concat = torch.cat([z, text_emb], dim=1)
        out = self.l1(z_concat)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ------------
#  Load CLIP
# ------------
print("Loading CLIP %s..." % opt.clip_model)
clip_model, _ = clip.load(opt.clip_model, device="cuda" if cuda else "cpu")
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# ------------
#  Load Generator
# ------------
generator = Generator()
if cuda:
    generator.cuda()

print("Loading checkpoint: %s" % opt.checkpoint)
checkpoint = torch.load(opt.checkpoint, map_location="cuda" if cuda else "cpu")
generator.load_state_dict(checkpoint["generator_state_dict"])
generator.eval()
print("Generator loaded. Starting epoch was %d" % checkpoint.get("epoch", -1))

# ------------
#  Class descriptions
# ------------
if opt.dataset == "gc10":
    CLASS_NAMES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
                   "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]
    CLASS_LABELS_CN = CLASS_NAMES
    CLASS_DESCRIPTIONS = {
        "crease": "a steel surface with a crease defect",
        "crescent_gap": "a steel surface with a crescent-shaped gap defect",
        "inclusion": "a steel surface with a non-metallic inclusion defect",
        "oil_spot": "a steel surface with an oil spot defect",
        "punching_hole": "a steel surface with a punching hole defect",
        "rolled_pit": "a steel surface with a rolled pit defect",
        "silk_spot": "a steel surface with a silk spot defect",
        "waist_folding": "a steel surface with a waist folding defect",
        "water_spot": "a steel surface with a water spot defect",
        "welding_line": "a steel surface with a welding line defect",
    }
elif opt.dataset == "neu":
    CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    CLASS_LABELS_CN = ["龟裂", "夹杂", "斑块", "麻面", "辊印", "划痕"]
    CLASS_DESCRIPTIONS = {
        "crazing": "a steel surface with a fine network of crazing cracks",
        "inclusion": "a steel surface with a non-metallic inclusion defect",
        "patches": "a steel surface with irregular patch-like defects",
        "pitted_surface": "a steel surface with distributed pitting defects",
        "rolled-in_scale": "a steel surface with a rolled-in scale defect",
        "scratches": "a steel surface with elongated scratch defects",
    }
else:
    CLASS_NAMES = ["dent", "scratch", "deformation", "bulge"]
    CLASS_LABELS_CN = ["凹点", "划痕", "形变", "鼓包"]
    CLASS_DESCRIPTIONS = {
        "dent": "a cable surface with a dent depression defect",
        "scratch": "a cable surface with a scratch line defect",
        "deformation": "a cable surface with deformation bending defect",
        "bulge": "a cable surface with a bulge swelling defect",
    }


def encode_text(text):
    """Encode text description to CLIP embedding."""
    tokens = clip.tokenize([text])
    if cuda:
        tokens = tokens.cuda()
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return emb.squeeze(0)  # [512]


def generate_batch(generator, text_emb, n_images, output_dir, prefix, batch_size=25):
    """Generate n_images with given text embedding."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    while count < n_images:
        batch_n = min(batch_size, n_images - count)
        z = Tensor(np.random.normal(0, 1, (batch_n, opt.latent_dim)))
        text_batch = text_emb.unsqueeze(0).repeat(batch_n, 1)  # [batch_n, 512]

        with torch.no_grad():
            gen_imgs = generator(z, text_batch)

        for j in range(batch_n):
            save_path = "%s/%s_%05d.png" % (output_dir, prefix, count + j)
            save_image(gen_imgs[j].data, save_path, normalize=True)

        count += batch_n
        if count % 100 == 0 or count == n_images:
            print("  Generated %d/%d (%s)" % (count, n_images, prefix))

    return count


# ------------
#  Generation
# ------------
total = 0

if opt.by_class:
    print("\n=== Generating by class ===")
    for cls_name, cls_cn in zip(CLASS_NAMES, CLASS_LABELS_CN):
        desc = CLASS_DESCRIPTIONS[cls_name]
        text_emb = encode_text(desc)
        if cuda:
            text_emb = text_emb.cuda()

        cls_dir = "%s/%s" % (opt.output, cls_name)
        n = generate_batch(generator, text_emb, opt.n_per_class, cls_dir, cls_name, opt.batch_size)
        total += n
        print("Class %s (%s): %d images, desc: \"%s\"" % (cls_name, cls_cn, n, desc))

elif opt.text:
    print("\n=== Generating from custom text ===")
    print("Text: \"%s\"" % opt.text)
    text_emb = encode_text(opt.text)
    if cuda:
        text_emb = text_emb.cuda()

    # Create a safe prefix from text
    safe_prefix = "".join(c if c.isalnum() else "_" for c in opt.text[:30]).strip("_")
    n = generate_batch(generator, text_emb, opt.n, opt.output, safe_prefix, opt.batch_size)
    total = n
    print("Generated %d images from text: \"%s\"" % (n, opt.text))

else:
    print("Error: must specify --by_class or --text")
    exit(1)

print("\n=== Done! Total %d images generated in %s ===" % (total, opt.output))
