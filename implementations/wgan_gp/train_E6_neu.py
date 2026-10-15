"""
E6 (类别条件 SP-WGAN-GP) for NEU-DET steel surface defect dataset.
  - num_classes: 6 (crazing/inclusion/patches/pitted_surface/rolled-in_scale/scratches)
  - img_size: 256 (NEU-DET 200x200 resize to 256, init_size=4, 6 up-blocks)
  - grayscale -> 3-channel via .convert("RGB")
  - same SP (lambda_sp=5.0) + FM (lambda_fm=50) + GP (lambda_gp=50) as cable
  - Training set only (1440 imgs), val set untouched (test only on 360 val)
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[1]

import argparse
import os
import numpy as np
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.nn.utils import spectral_norm

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument("--b1", type=float, default=0.0)
parser.add_argument("--b2", type=float, default=0.9)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--n_critic", type=int, default=1)
parser.add_argument("--sample_interval", type=int, default=400)
parser.add_argument("--lambda_gp", type=float, default=50.0)
parser.add_argument("--lambda_fm", type=float, default=50.0)
parser.add_argument("--lambda_sp", type=float, default=5.0)
parser.add_argument("--num_classes", type=int, default=6)
parser.add_argument("--class_embed_dim", type=int, default=64)
parser.add_argument("--data_path", type=str, default=str(_release_root / 'data/neu_det/NEU-DET/train/images'))
parser.add_argument("--img_dir", type=str, default=str(_release_root / 'data/neu_det/gen_E6_neu'))
parser.add_argument("--ckpt_dir", type=str, default=str(_release_root / 'data/neu_det/ckpt_E6_neu'))
opt = parser.parse_args()
print(opt)

os.makedirs(opt.img_dir, exist_ok=True)
os.makedirs(opt.ckpt_dir, exist_ok=True)

cuda = torch.cuda.is_available()


def weights_init_normal(m):
    if m.__class__.__name__.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif m.__class__.__name__.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def d_block(in_f, out_f, bn=True):
            blk = [spectral_norm(nn.Conv2d(in_f, out_f, 4, 2, 1)),
                   nn.LeakyReLU(0.2, True)]
            if bn:
                blk.append(nn.InstanceNorm2d(out_f))
            return blk

        self.model = nn.Sequential(
            *d_block(opt.channels, 32, bn=False),
            *d_block(32, 64),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            *d_block(512, 512),
        )
        self.feature_layer = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)))
        self.adv_layer = nn.Sequential(nn.Linear(512 * 4 * 4, 1))
        self.class_embedding = nn.Embedding(opt.num_classes, opt.class_embed_dim)
        self.class_proj = nn.Linear(opt.class_embed_dim, 1)

    def forward(self, img, labels, return_features=False):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        out = feat_pool.view(feat_pool.shape[0], -1)
        validity = self.adv_layer(out)
        ce = self.class_embedding(labels)
        validity = validity + self.class_proj(ce)
        if return_features:
            return validity, feat_pool
        return validity

    def extract_features(self, img):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        return feat_pool.view(feat_pool.shape[0], -1)


generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda(); discriminator.cuda()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Data: ImageFolder + grayscale->RGB + Resize(256)
dataloader = DataLoader(
    datasets.ImageFolder(
        opt.data_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),  # grayscale to 3-channel
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]),
    ),
    batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True,
)
class_to_idx = dataloader.dataset.class_to_idx
print("Class mapping:")
for n, i in sorted(class_to_idx.items(), key=lambda x: x[1]):
    print("  %d: %s" % (i, n))
print("Dataset: %d images, %d classes" % (len(dataloader.dataset), opt.num_classes))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

start_epoch = 0
batches_done = 0
if opt.resume and os.path.isfile(opt.resume):
    print("Resuming from %s" % opt.resume)
    ck = torch.load(opt.resume, map_location='cuda' if cuda else 'cpu')
    generator.load_state_dict(ck['generator_state_dict'])
    discriminator.load_state_dict(ck['discriminator_state_dict'])
    optimizer_G.load_state_dict(ck['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(ck['optimizer_D_state_dict'])
    start_epoch = ck['epoch'] + 1
    batches_done = ck.get('batches_done', start_epoch * len(dataloader))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real, fake, labels):
    alpha = Tensor(np.random.random((real.size(0), 1, 1, 1)))
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d = D(interp, labels)
    fake_one = Variable(Tensor(real.shape[0], 1).fill_(1.0), requires_grad=False)
    g = autograd.grad(outputs=d, inputs=interp, grad_outputs=fake_one,
                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    g = g.view(g.size(0), -1)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def sobel_filters(x):
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = gray.unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8), gx, gy


def structure_preserving_loss(fake, real):
    fe, fgx, fgy = sobel_filters(fake)
    re, rgx, rgy = sobel_filters(real.detach())
    return F.l1_loss(fe, re) + F.l1_loss(fgx, rgx) + F.l1_loss(fgy, rgy)


for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        optimizer_D.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        fake_imgs = generator(z, labels)
        real_v = discriminator(real_imgs, labels)
        fake_v = discriminator(fake_imgs.detach(), labels)
        gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels)
        d_loss = -real_v.mean() + fake_v.mean() + opt.lambda_gp * gp
        d_loss.backward()
        optimizer_D.step()

        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()
            gen_imgs = generator(z, labels)
            fake_v = discriminator(gen_imgs, labels)
            g_adv = -fake_v.mean()
            g_fm = F.mse_loss(discriminator.extract_features(gen_imgs),
                              discriminator.extract_features(real_imgs).detach())
            g_sp = structure_preserving_loss(gen_imgs, real_imgs) if opt.lambda_sp > 0 else 0
            g_loss = g_adv + opt.lambda_fm * g_fm + opt.lambda_sp * g_sp
            g_loss.backward()
            optimizer_G.step()

        batches_done += 1
        if batches_done % opt.sample_interval == 0:
            # 8 类的 grid (取 8 张示例)
            with torch.no_grad():
                fix_z = Tensor(np.random.normal(0, 1, (8, opt.latent_dim)))
                fix_lab = LongTensor([j % opt.num_classes for j in range(8)])
                sample = generator(fix_z, fix_lab)
                sample = (sample + 1) / 2  # [0,1]
                save_image(sample, os.path.join(opt.img_dir, "%06d.png" % batches_done), nrow=4)
            print("[Epoch %d/%d] [Batches %d] [D loss: %.4f] [G loss: %.4f]"
                  % (epoch, opt.n_epochs, batches_done, d_loss.item(), g_loss.item()))

    # 定期保存checkpoint
    if (epoch + 1) % 50 == 0 or epoch == opt.n_epochs - 1:
        ck_path = os.path.join(opt.ckpt_dir, "ckpt_latest.pth")
        torch.save({
            'epoch': epoch, 'batches_done': batches_done,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, ck_path)
        print("[CKPT] saved %s at epoch %d" % (ck_path, epoch))

print("Training complete.")
