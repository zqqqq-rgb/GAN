"""
dcgan_train.py — 独立的 DCGAN 训练脚本，支持 256×256 分辨率 + checkpoint 保存

用法:
    python dcgan_train.py --n_epochs 500 --img_size 256 --batch_size 4 --output_ckpt experiments/E4/dcgan_checkpoint.pth
"""

import argparse
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--sample_interval", type=int, default=400)
parser.add_argument("--output_ckpt", type=str, default="dcgan_checkpoint.pth")
parser.add_argument("--data_path", type=str, default="../../data/cable_defect", help="image folder path (ImageFolder layout)")
parser.add_argument("--img_dir", type=str, default="dcgan_images")
parser.add_argument("--resume", type=str, default=None)
opt = parser.parse_args()

os.makedirs(opt.img_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DCGAN] device={device}, img_size={opt.img_size}, n_epochs={opt.n_epochs}")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=256, channels=3):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        layers = []
        in_ch = 128
        scale = self.init_size
        while scale < img_size:
            out_ch = max(in_ch // 2, 64)
            layers += [
                nn.BatchNorm2d(in_ch),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
            scale *= 2
        layers += [nn.Conv2d(in_ch, channels, 3, stride=1, padding=1), nn.Tanh()]
        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self, img_size=256, channels=3):
        super().__init__()
        layers = []
        in_ch = channels
        out_ch = 32
        scale = img_size
        while scale > 4:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
            scale //= 2
        self.model = nn.Sequential(*layers)
        self.adv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_ch * 16, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        feat = self.model(img)
        return self.adv_layer(feat)


adversarial_loss = nn.BCELoss()
generator = Generator(latent_dim=opt.latent_dim, img_size=opt.img_size).to(device)
discriminator = Discriminator(img_size=opt.img_size).to(device)
adversarial_loss.to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Data
dataset_path = opt.data_path
dataloader = DataLoader(
    datasets.ImageFolder(dataset_path, transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])),
    batch_size=opt.batch_size, shuffle=True,
)

opt_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

start_epoch = 0
if opt.resume:
    ckpt = torch.load(opt.resume, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    if "opt_G" in ckpt:
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from epoch {ckpt['epoch']}, starting at {start_epoch}")

for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real = imgs.to(device)
        b = real.size(0)
        valid = torch.ones(b, 1, device=device)
        fake_label = torch.zeros(b, 1, device=device)

        # Train Generator
        opt_G.zero_grad()
        z = torch.randn(b, opt.latent_dim, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        opt_G.step()

        # Train Discriminator
        opt_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs[:min(4,b)], f"{opt.img_dir}/{batches_done:06d}.png", normalize=True)
            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}] [D: {d_loss.item():.4f}] [G: {g_loss.item():.4f}]")

    if (epoch + 1) % 50 == 0:
        torch.save({
            "epoch": epoch, "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        }, opt.output_ckpt)
        print(f">>> Saved: {opt.output_ckpt}")

torch.save({
    "epoch": opt.n_epochs - 1,
    "generator_state_dict": generator.state_dict(),
    "discriminator_state_dict": discriminator.state_dict(),
    "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
}, opt.output_ckpt)
print(f"Done. Final checkpoint: {opt.output_ckpt}")