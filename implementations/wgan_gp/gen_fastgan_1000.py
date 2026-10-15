"""FastGAN (lightweight-gan) 生成 N 张 — 用 EMA 权重 (v2: 加 spawn 保护)
用法: python gen_fastgan_1000.py [--n 1000] [--out <dir>] [--name fastgan_leak]
"""
import os, sys, argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import torchvision
from tqdm import tqdm
from lightweight_gan import Trainer

RESULTS = '../experiments/fastgan_leak'
MODELS = '../experiments/fastgan_leak/models'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--name', default='fastgan_leak')
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--out', default='../../data/yolo_dataset/images/leakfix/gen_FastGAN_leak')
    args = ap.parse_args()

    model = Trainer(
        name=args.name,
        results_dir=RESULTS,
        models_dir=MODELS,
        image_size=args.image_size,
        batch_size=10,
        attn_res_layers=[32],
        num_workers=0,
        use_aim=False,
    )
    model.load(-1)  # 最新 checkpoint (model_*.pt)
    model.GAN.eval()

    latent_dim = model.GAN.latent_dim
    os.makedirs(args.out, exist_ok=True)
    ext = 'jpg' if not model.transparent else 'png'

    with torch.no_grad():
        for i in tqdm(range(args.n), desc='Generating FastGAN'):
            latents = torch.randn((1, latent_dim)).cuda(0)
            img = model.generate_(model.GAN.GE, latents)  # EMA 权重
            torchvision.utils.save_image(img[0], os.path.join(args.out, f'{i:04d}.{ext}'), nrow=1)

    print(f'[FastGAN] generated {args.n} images -> {args.out}', flush=True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
