"""
Text-Guided SP-WGAN-GP with a true projection discriminator.

Upgrades E6 (class-label conditional) to true text-description conditional:
  - CLIP ViT-B/32 text encoder (FROZEN) encodes defect descriptions -> 512-dim
  - Generator: noise z(128) + text_emb(512) -> 640 -> 512x512 image
  - Discriminator: D(x,c) = u(h(x)) + <h(x), V e(c)>
  - Uses permutation-invariant multi-scale Sobel statistics for L_sp
  - Uses batch-mean discriminator features for classical feature matching

Usage:
  python cwgan_gp_clip_proj.py --n_epochs 500 --lambda_sp 5.0 --lambda_fm 50
  python cwgan_gp_clip_proj.py --resume latest_checkpoint_e7proj.pth --n_epochs 500
"""

import argparse
import os
import numpy as np
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.nn.utils import spectral_norm

import clip

# ------------
#  Args
# ------------
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
parser.add_argument("--text_emb_dim", type=int, default=512, help="CLIP text embedding dimension")
parser.add_argument("--data_path", type=str, default='../../data/cable_leakfix_train', help="classified dataset path")
parser.add_argument("--tag", type=str, default="e7", help="checkpoint filename tag (e.g. e7leak -> latest_checkpoint_e7leak.pth)")
parser.add_argument("--img_dir", type=str, default="", help="output image directory (auto timestamp if empty)")
parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant")
parser.add_argument("--seed", type=int, default=42, help="random seed")
opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
# Fixed RNG seeds make runs repeatable at the experimental level.  cuDNN
# benchmarking is retained because deterministic 512-pixel convolutions are
# prohibitively slow on the 8-GB laptop GPU; exact bitwise reproducibility is
# therefore not claimed.
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

# ------------
#  CLIP Text Encoder (FROZEN)
# ------------
print("Loading CLIP %s text encoder..." % opt.clip_model)
clip_model, clip_preprocess = clip.load(opt.clip_model, device="cuda" if cuda else "cpu")
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
print("CLIP loaded and frozen.")

# Defect class -> text description mapping
# ImageFolder sorts class names alphabetically: dent, scratch, deformation, bulge
# But actual order depends on folder names - will be printed at runtime
CLASS_DESCRIPTIONS = {
    "dent": "a cable surface with a dent depression defect",
    "scratch": "a cable surface with a scratch line defect",
    "deformation": "a cable surface with deformation bending defect",
    "bulge": "a cable surface with a bulge swelling defect",
    # Fallbacks for Chinese folder names or variations
    "凹点": "a cable surface with a dent depression defect",
    "划痕": "a cable surface with a scratch line defect",
    "形变": "a cable surface with deformation bending defect",
    "鼓包": "a cable surface with a bulge swelling defect",
    "crazing": "a steel surface with a fine network of crazing cracks",
    "inclusion": "a steel surface with a non-metallic inclusion defect",
    "patches": "a steel surface with irregular patch-like defects",
    "pitted_surface": "a steel surface with distributed pitting defects",
    "rolled-in_scale": "a steel surface with a rolled-in scale defect",
    "scratches": "a steel surface with elongated scratch defects",
    "crease": "a steel surface with a crease defect",
    "crescent_gap": "a steel surface with a crescent-shaped gap defect",
    "oil_spot": "a steel surface with an oil spot defect",
    "punching_hole": "a steel surface with a punching hole defect",
    "rolled_pit": "a steel surface with a rolled pit defect",
    "silk_spot": "a steel surface with a silk spot defect",
    "waist_folding": "a steel surface with a waist folding defect",
    "water_spot": "a steel surface with a water spot defect",
    "welding_line": "a steel surface with a welding line defect",
}

def get_class_description(class_name):
    """Get text description for a class name, with fallback."""
    class_lower = class_name.lower()
    if class_name in CLASS_DESCRIPTIONS:
        return CLASS_DESCRIPTIONS[class_name]
    if class_lower in CLASS_DESCRIPTIONS:
        return CLASS_DESCRIPTIONS[class_lower]
    # Generic fallback
    return "a cable surface with %s defect" % class_name


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ------------
#  Dataset with text descriptions
# ------------
class ConditionalImageDataset(Dataset):
    """Wraps ImageFolder to also return CLIP-encoded text descriptions."""

    def __init__(self, root, transform, clip_model, clip_tokenizer, cuda):
        self.base = datasets.ImageFolder(root, transform=transform)
        self.class_to_idx = self.base.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Pre-compute CLIP text embeddings for each class
        self.class_text_embeddings = {}
        for class_name, idx in self.class_to_idx.items():
            desc = get_class_description(class_name)
            text_tokens = clip_tokenizer([desc])
            if cuda:
                text_tokens = text_tokens.cuda()
            with torch.no_grad():
                text_emb = clip_model.encode_text(text_tokens)
            self.class_text_embeddings[idx] = text_emb.squeeze(0)  # [512]
            print("  Class %d (%s): \"%s\" -> emb shape %s" % (idx, class_name, desc, text_emb.shape))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        text_emb = self.class_text_embeddings[label]
        return img, label, text_emb


# ------------
#  Generator
# ------------
class Generator(nn.Module):
    """Text-conditional 512x512 Generator.

    Input: noise z [B, latent_dim] + text_emb [B, text_emb_dim]
    The text embedding is concatenated with z before the linear layer.
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 64  # 512//64 = 8

        # latent_dim + text_emb_dim -> feature map
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.text_emb_dim, 512 * self.init_size ** 2)
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

    def forward(self, z, text_emb):
        # Concatenate noise + text embedding
        z_concat = torch.cat([z, text_emb], dim=1)  # [B, latent_dim + text_emb_dim]
        out = self.l1(z_concat)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# ------------
#  Discriminator (Projection with text embedding)
# ------------
class Discriminator(nn.Module):
    """Text-conditional Discriminator with Projection.

    Uses projection-based conditioning (Miyato & Koyama 2018):
      D(x, c) = u(h(x)) + <h(x), V e(c)>.

    The condition therefore interacts with image features.  Mapping the
    condition directly to a scalar would cancel between the real and fake
    terms of the Wasserstein critic and would not test image-condition
    compatibility.

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

        # Project the frozen CLIP embedding into the 512-D image-feature space.
        # Bias is omitted because the projection term is an inner product.
        self.text_proj = nn.Linear(opt.text_emb_dim, 512, bias=False)

    def forward(self, img, text_emb, return_features=False):
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        out = feat_pool.view(feat_pool.shape[0], -1)
        validity = self.adv_layer(out)  # [B, 1]

        # True image-condition projection term.  The 1/sqrt(d) factor keeps
        # its scale comparable to the unconditional critic head.
        image_feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # [B, 512]
        text_feat = self.text_proj(F.normalize(text_emb.float(), dim=1))
        text_score = torch.sum(image_feat * text_feat, dim=1, keepdim=True) / (512.0 ** 0.5)
        validity = validity + text_score

        if return_features:
            return validity, feat_pool
        return validity

    def extract_features(self, img):
        """Extract features for feature matching (unconditional)."""
        feat = self.model(img)
        feat_pool = self.feature_layer(feat)
        return feat_pool.view(feat_pool.shape[0], -1)


# ------------
#  Init
# ------------
lambda_gp = opt.lambda_gp
lambda_fm = opt.lambda_fm
lambda_sp = opt.lambda_sp

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Data loader with text descriptions
dataset_path = opt.data_path
os.makedirs(dataset_path, exist_ok=True)

print("Building dataset with CLIP text descriptions:")
dataset = ConditionalImageDataset(
    root=dataset_path,
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
    clip_model=clip_model,
    clip_tokenizer=clip.tokenize,
    cuda=cuda,
)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

print("Dataset: %d images, %d classes" % (len(dataset), len(dataset.class_to_idx)))

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


def compute_gradient_penalty(D, real_samples, fake_samples, text_emb):
    """Gradient penalty for text-conditional discriminator."""
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, text_emb)
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
    """Permutation-invariant multi-scale edge-statistics matching.

    GAN training data are unpaired, so a pixel-wise distance between the
    b-th generated and b-th real image has no correspondence interpretation.
    We instead compare first- and second-order batch statistics of Sobel edge
    magnitude and absolute x/y gradients at several spatial scales.  The loss
    is unchanged by any permutation of the real or generated batch.
    """
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


# ------------
#  Training
# ------------
for epoch in range(start_epoch, opt.n_epochs):
    for i, (imgs, labels, text_embs) in enumerate(dataloader):

        real_imgs = Variable(imgs.type(Tensor))
        text_embs = Variable(text_embs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        fake_imgs = generator(z, text_embs)

        real_validity = discriminator(real_imgs, text_embs)
        fake_validity = discriminator(fake_imgs.detach(), text_embs)
        gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, text_embs)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        if i % opt.n_critic == 0:
            optimizer_G.zero_grad()

            gen_imgs = generator(z, text_embs)

            # WGAN adversarial loss (text-conditional)
            fake_validity = discriminator(gen_imgs, text_embs)
            g_adv = -torch.mean(fake_validity)

            # Classical feature matching compares expected feature statistics;
            # it does not impose an artificial sample-to-sample correspondence.
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
                sample_classes = list(range(min(4, len(dataset.class_to_idx))))
                if sample_classes:
                    sample_text_embs = torch.stack([dataset.class_text_embeddings[c] for c in sample_classes])
                    sample_z = Tensor(np.random.normal(0, 1, (len(sample_classes), opt.latent_dim)))
                    sample_imgs = generator(sample_z, sample_text_embs)
                    save_image(sample_imgs.data, "%s/%05d.png" % (img_dir, batches_done), nrow=4, normalize=True)

            batches_done += 1

    # Save full checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        ckpt_path = "checkpoint_%s_epoch%03d.pth" % (opt.tag, epoch + 1)
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
    }, "latest_checkpoint_%s.pth" % opt.tag)

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
}, "checkpoint_%s_final.pth" % opt.tag)
print("Saved final checkpoint: checkpoint_%s_final.pth" % opt.tag)
