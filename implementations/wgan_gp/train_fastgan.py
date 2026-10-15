"""FastGAN (lightweight-gan) leakfix 训练 — 修复版 v2
修复1: 入口从 `python -m lightweight_gan` 改为直接调用 train_from_folder
修复2: Windows spawn 需 if __name__=='__main__' 保护 + num_workers=0 (避免 DataLoader worker 递归 import)
用法: python train_fastgan.py [--steps N] [--name fastgan_leak]
"""
import os, sys, argparse

# cd 到本脚本所在目录 (implementations/wgan_gp)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from lightweight_gan.cli import train_from_folder

DATA = '../../data/cable_leakfix_train-flat'
RESULTS = '../experiments/fastgan_leak'
MODELS = '../experiments/fastgan_leak/models'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--name', default='fastgan_leak')
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--accumulate', type=int, default=2)
    ap.add_argument('--amp', action='store_true', default=False)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f'[FastGAN] data={DATA} name={args.name} steps={args.steps} size={args.image_size} '
          f'batch={args.batch_size} accum={args.accumulate} amp={args.amp}', flush=True)

    train_from_folder(
        data=DATA,
        results_dir=RESULTS,
        models_dir=MODELS,
        name=args.name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        gradient_accumulate_every=args.accumulate,
        num_train_steps=args.steps,
        save_every=1000,
        evaluate_every=1000,
        seed=args.seed,
        attn_res_layers=[32],
        num_workers=0,  # Windows spawn 安全
        amp=args.amp,
    )
    print('[FastGAN] training done', flush=True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
