import argparse
import os
import numpy as np
import math
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
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: beta1")
parser.add_argument("--b2", type=float, default=0.9, help="adam: beta2")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches (small for 512px)")
parser.add_argument("--resume", type=str, default="", help="resume from checkpoint file path (e.g. checkpoint_epoch050.pth)")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="D steps per G step")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples (batches)")
parser.add_argument("--lambda_gp", type=float, default=50, help="gradient penalty weight")
parser.add_argument("--lambda_fm", type=float, default=50, help="feature matching loss weight")
parser.add_argument("--lambda_sp", type=float, default=10.0, help="structure-preserving loss weight (set 0 = vanilla WGAN-GP baseline for ablation)")
parser.add_argument("--data_path", type=str, default="../../data/cable_defect", help="image folder path (ImageFolder layout)")
parser.add_argument("--tag", type=str, default="", help="checkpoint filename tag (e.g. e1leak -> latest_checkpoint_e1leak.pth)")
parser.add_argument("--diff_aug", action="store_true", help="enable DiffAugment (color,translation,cutout) on discriminator inputs")
parser.add_argument("--img_dir", type=str, default="", help="output image directory (auto timestamp if empty)")
opt = parser.parse_args()

if opt.diff_aug:
    from diffaugment import DiffAugment, POLICY as DA_POLICY
else:
    DA_POLICY = ""
print(opt)

# Timestamped image directory to avoid overwriting previous runs
if opt.img_dir:
    img_dir = opt.img_dir
else:
    img_dir = "images/" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(img_dir, exist_ok=True)
print("Image output: %s" % img_dir)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """512x512 Generator with 6 upsampling stages (8->16->32->64->128->256->512)"""
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 64  # 512//64 = 8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 512 * self.init_size ** 2))

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

            nn.Conv2d(16, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """Discriminator with intermediate feature extraction for feature matching loss"""
    def __init__(self):
        super(Discriminator, self).__init__()

        def d_block(in_filters, out_filters, bn=True):
            block = [
                spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *d_block(opt.channels, 32, bn=False),   # 512 -> 256
            *d_block(32, 64),                        # 256 -> 128
            *d_block(64, 128),                       # 128 -> 64
            *d_block(128, 256),                      # 64 -> 32
            *d_block(256, 512),                      # 32 -> 16
            *d_block(512, 512),                      # 16 -> 8
        )

        # Feature layers for feature matching (extract from intermediate layers)
        self.feature_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.adv_layer = nn.Sequential(nn.Linear(512 * 4 * 4, 1))

    def forward(self, img, return_features=False):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        out = feat_pool.view(feat_pool.shape[0], -1)
        validity = self.adv_layer(out)
        if return_features:
            return validity, feat_pool
        return validity

    def extract_features(self, img):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        return feat_pool.view(feat_pool.shape[0], -1)


# Loss weight for gradient penalty
lambda_gp = opt.lambda_gp
lambda_fm = opt.lambda_fm
lambda_sp = opt.lambda_sp

# Initialize
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Data loader
dataset_path = opt.data_path
os.makedirs(dataset_path, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ------------
#  Resume
# ------------
start_epoch = 0
batches_done = 0

if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading checkpoint: %s" % opt.resume)
        checkpoint = torch.load(opt.resume, map_location='cuda' if cuda else 'cpu')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        batches_done = checkpoint.get('batches_done', start_epoch * len(dataloader))
        # Restore img_dir from checkpoint if not manually specified
        if not opt.img_dir and 'img_dir' in checkpoint:
            img_dir = checkpoint['img_dir']
            print("Restored img_dir from checkpoint: %s" % img_dir)
        print("Resumed from epoch %d, batches_done=%d" % (start_epoch, batches_done))
    else:
        print("Checkpoint not found: %s, starting from scratch" % opt.resume)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sobel_filters(x):
    """Compute Sobel edge magnitude and gradient components.
    x: [B, C, H, W] image tensor (works with any normalization).
    Returns: (edge_mag [B,1,H,W], gx [B,1,H,W], gy [B,1,H,W])
    """
    # RGB -> grayscale
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = gray.unsqueeze(1)  # [B, 1, H, W]
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return edge_mag, gx, gy


def structure_preserving_loss(fake, real):
    """Structure-Preserving Loss (SP-Loss).

    Enforces the generator to preserve the edge magnitude and gradient
    direction of real defect structures. This structural prior regularizes
    generation and mitigates mode collapse, which is critical for industrial
    defect images where defect boundaries/textures carry diagnostic meaning.

    L_sp = L1(edge_fake, edge_real) + L1(gx_fake, gx_real) + L1(gy_fake, gy_real)

    Set lambda_sp = 0 to recover vanilla WGAN-GP (used as ablation baseline).
    """
    fake_edge, fake_gx, fake_gy = sobel_filters(fake)
    real_edge, real_gx, real_gy = sobel_filters(real.detach())
    l_edge = F.l1_loss(fake_edge, real_edge)
    l_grad = F.l1_loss(fake_gx, real_gx) + F.l1_loss(fake_gy, real_gy)
    return l_edge + l_grad


# ----------
#  Training
# ----------

for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        fake_imgs = generator(z)

        # DiffAugment (Zhao et al. 2020): differentiable aug on D inputs only
        if opt.diff_aug:
            real_in = DiffAugment(real_imgs, policy=DA_POLICY)
            fake_in = DiffAugment(fake_imgs.detach(), policy=DA_POLICY)
        else:
            real_in, fake_in = real_imgs, fake_imgs.detach()

        real_validity = discriminator(real_in)
        fake_validity = discriminator(fake_in)
        gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()

            gen_imgs = generator(z)

            # WGAN adversarial loss (with DiffAugment if enabled)
            g_in = DiffAugment(gen_imgs, policy=DA_POLICY) if opt.diff_aug else gen_imgs
            fake_validity = discriminator(g_in)
            g_adv = -torch.mean(fake_validity)

            # Feature matching loss: match intermediate features of real and fake
            real_features = discriminator.extract_features(real_imgs)
            fake_features = discriminator.extract_features(gen_imgs)
            g_fm = F.mse_loss(fake_features, real_features.detach())

            # Structure-preserving loss: enforce edge/gradient consistency
            # (set lambda_sp=0 to disable -> vanilla WGAN-GP baseline)
            if lambda_sp > 0:
                g_sp = structure_preserving_loss(gen_imgs, real_imgs)
            else:
                g_sp = torch.zeros(1, device=gen_imgs.device)

            g_loss = g_adv + lambda_fm * g_fm + lambda_sp * g_sp

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D: %.3f] [G: %.3f adv=%.3f fm=%.3f sp=%.3f] [GP: %.3f]"
                % (epoch, opt.n_epochs, i, len(dataloader),
                   d_loss.item(), g_loss.item(), g_adv.item(), g_fm.item(), g_sp.item(), gp.item())
            )

            if batches_done % opt.sample_interval == 0:
                # Save 1 individual image (512x512)
                save_image(gen_imgs.data[:1], "%s/%05d.png" % (img_dir, batches_done), normalize=True)

            batches_done += 1

    # Save full checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        ckpt_path = "checkpoint%s_epoch%03d.pth" % ("_" + opt.tag if opt.tag else "", epoch + 1)
        torch.save({
            'epoch': epoch,
            'batches_done': batches_done,
            'img_dir': img_dir,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, ckpt_path)
        print(">>> Saved full checkpoint: %s" % ckpt_path)

    # Save latest checkpoint for easy resume (overwrites each epoch)
    torch.save({
        'epoch': epoch,
        'batches_done': batches_done,
        'img_dir': img_dir,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, "latest_checkpoint%s.pth" % ("_" + opt.tag if opt.tag else ""))

print("Training complete!")
torch.save({
    'epoch': opt.n_epochs - 1,
    'batches_done': batches_done,
    'img_dir': img_dir,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, "checkpoint%s_final.pth" % ("_" + opt.tag if opt.tag else ""))
print("Saved final checkpoint: checkpoint%s_final.pth" % ("_" + opt.tag if opt.tag else ""))
