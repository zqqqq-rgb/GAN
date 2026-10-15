"""
Class-conditional SP-WGAN-GP with a true projection discriminator.

Extends the unconditional SP-WGAN-GP with class-conditional generation:
  - Generator: noise z + class embedding -> image
  - Discriminator: D(x,y) = u(h(x)) + <h(x), e(y)>
  - Uses permutation-invariant multi-scale Sobel statistics for L_sp
  - Uses batch-mean discriminator features for classical feature matching

Usage:
  python cwgan_gp_proj.py --n_epochs 500 --lambda_sp 5.0 --lambda_fm 50
  python cwgan_gp_proj.py --resume latest_checkpoint_e6proj.pth --n_epochs 500
"""

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
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: beta1")
parser.add_argument("--b2", type=float, default=0.9, help="adam: beta2")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches (small for 512px)")
parser.add_argument("--resume", type=str, default="", help="resume from checkpoint file path")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="D steps per G step")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples (batches)")
parser.add_argument("--lambda_gp", type=float, default=50, help="gradient penalty weight")
parser.add_argument("--lambda_fm", type=float, default=50, help="feature matching loss weight")
parser.add_argument("--lambda_sp", type=float, default=5.0, help="structure-preserving loss weight")
parser.add_argument("--structure_scales", type=int, nargs="+", default=[32, 16, 8],
                    help="output grids used by multi-scale Sobel statistics")
parser.add_argument("--structure_std_weight", type=float, default=0.5,
                    help="relative weight of the batch standard-deviation term")
parser.add_argument("--num_classes", type=int, default=4, help="number of defect classes")
parser.add_argument("--class_embed_dim", type=int, default=64, help="class embedding dimension")
parser.add_argument("--data_path", type=str, default='../../data/cable_leakfix_train', help="classified dataset path")
parser.add_argument("--tag", type=str, default="", help="checkpoint filename tag (e.g. e6leak -> latest_checkpoint_e6leak.pth)")
parser.add_argument("--img_dir", type=str, default="", help="output image directory (auto timestamp if empty)")
parser.add_argument("--seed", type=int, default=42, help="random seed")
opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Timestamped image directory
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
    """Conditional 512x512 Generator.

    Input: noise z [B, latent_dim] + class labels [B]
    The class label is embedded and concatenated with z before the linear layer.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 64  # 512//64 = 8
        self.class_embedding = nn.Embedding(opt.num_classes, opt.class_embed_dim)

        # latent_dim + class_embed_dim -> feature map
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.class_embed_dim, 512 * self.init_size ** 2)
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

            nn.Conv2d(16, opt.channels, 3, stride=1, padding=1),
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


class Discriminator(nn.Module):
    """Conditional Discriminator with Projection Discriminator.

    Uses projection-based conditioning (Miyato & Koyama 2018):
      D(x, y) = u(h(x)) + <h(x), e(y)>.

    This is more stable than concatenation-based conditioning (AC-GAN).
    Feature extraction for feature matching loss is unconditional.
    """
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

        self.feature_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.adv_layer = nn.Sequential(nn.Linear(512 * 4 * 4, 1))

        # Projection discriminator components
        self.class_embedding = nn.Embedding(opt.num_classes, 512)

    def forward(self, img, labels, return_features=False):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        out = feat_pool.view(feat_pool.shape[0], -1)
        validity = self.adv_layer(out)  # [B, 1]

        # True image-condition projection term.
        image_feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # [B, 512]
        class_emb = self.class_embedding(labels)  # [B, 512]
        class_score = torch.sum(image_feat * class_emb, dim=1, keepdim=True) / (512.0 ** 0.5)
        validity = validity + class_score

        if return_features:
            return validity, feat_pool
        return validity

    def extract_features(self, img):
        """Extract features for feature matching (unconditional)."""
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        return feat_pool.view(feat_pool.shape[0], -1)


# Loss weights
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

# Data loader - ImageFolder with classified subdirectories
dataset_path = opt.data_path
os.makedirs(dataset_path, exist_ok=True)
dataloader = DataLoader(
    datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
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

# Print class-to-index mapping
class_to_idx = dataloader.dataset.class_to_idx
print("Class mapping (ImageFolder auto-sorted):")
for cls_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
    print("  %d: %s" % (idx, cls_name))
print("Dataset: %d images, %d classes" % (len(dataloader.dataset), opt.num_classes))

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
        if not opt.img_dir and 'img_dir' in checkpoint:
            img_dir = checkpoint['img_dir']
            print("Restored img_dir from checkpoint: %s" % img_dir)
        print("Resumed from epoch %d, batches_done=%d" % (start_epoch, batches_done))
    else:
        print("Checkpoint not found: %s, starting from scratch" % opt.resume)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Gradient penalty for conditional discriminator."""
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sobel_filters(x):
    """Compute Sobel edge magnitude and gradient components."""
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
    """Permutation-invariant multi-scale edge-statistics matching."""
    fake_edge, fake_gx, fake_gy = sobel_filters(fake)
    real_edge, real_gx, real_gy = sobel_filters(real.detach())
    fake_struct = torch.cat([fake_edge, fake_gx.abs(), fake_gy.abs()], dim=1)
    real_struct = torch.cat([real_edge, real_gx.abs(), real_gy.abs()], dim=1)

    loss = fake.new_zeros(())
    for scale in opt.structure_scales:
        fake_s = F.adaptive_avg_pool2d(fake_struct, (scale, scale))
        real_s = F.adaptive_avg_pool2d(real_struct, (scale, scale))
        fake_mean = fake_s.mean(dim=0)
        real_mean = real_s.mean(dim=0)
        fake_std = fake_s.std(dim=0, unbiased=False)
        real_std = real_s.std(dim=0, unbiased=False)
        loss = loss + F.l1_loss(fake_mean, real_mean)
        loss = loss + opt.structure_std_weight * F.l1_loss(fake_std, real_std)
    return loss / float(len(opt.structure_scales))


# ----------
#  Training
# ----------

for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        fake_imgs = generator(z, labels)

        real_validity = discriminator(real_imgs, labels)
        fake_validity = discriminator(fake_imgs.detach(), labels)
        gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()

            gen_imgs = generator(z, labels)

            # WGAN adversarial loss (conditional)
            fake_validity = discriminator(gen_imgs, labels)
            g_adv = -torch.mean(fake_validity)

            # Match expected discriminator features across the batch.
            real_features = discriminator.extract_features(real_imgs)
            fake_features = discriminator.extract_features(gen_imgs)
            g_fm = F.mse_loss(
                fake_features.mean(dim=0),
                real_features.detach().mean(dim=0),
            )

            # Structure-preserving loss
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
                # Save 1 image per class (up to 4) for visual monitoring
                sample_labels = LongTensor(np.arange(min(4, opt.num_classes)))
                sample_z = Tensor(np.random.normal(0, 1, (len(sample_labels), opt.latent_dim)))
                sample_imgs = generator(sample_z, sample_labels)
                save_image(sample_imgs.data, "%s/%05d.png" % (img_dir, batches_done), nrow=4, normalize=True)

            batches_done += 1

    # Save full checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        ckpt_path = "checkpoint%s_epoch%03d.pth" % ("_" + opt.tag if opt.tag else "", epoch + 1)
        torch.save({
            'epoch': epoch,
            'batches_done': batches_done,
            'img_dir': img_dir,
            'method_version': 'projection-mses-v1',
            'config': vars(opt),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, ckpt_path)
        print(">>> Saved full checkpoint: %s" % ckpt_path)

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'batches_done': batches_done,
        'img_dir': img_dir,
        'method_version': 'projection-mses-v1',
        'config': vars(opt),
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
    'method_version': 'projection-mses-v1',
    'config': vars(opt),
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
}, "checkpoint%s_final.pth" % ("_" + opt.tag if opt.tag else ""))
print("Saved final checkpoint: checkpoint%s_final.pth" % ("_" + opt.tag if opt.tag else ""))
