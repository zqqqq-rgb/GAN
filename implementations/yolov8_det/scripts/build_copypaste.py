# -*- coding: utf-8 -*-
"""CopyPaste 数据增强基线（电缆数据集）。
协议：196张真实训练图作为背景池与源图池，每张生成图 = 背景图(保留原缺陷)
     + 从其他图随机裁剪的1~2个缺陷patch（随机缩放0.9~1.1，粘贴位置按真实缺陷
     位置统计分布采样：x~N(0.434,0.071) clip[0.25,0.62]，y~U(0.08,0.92)），
     新bbox精确已知。生成1000张增强图，与GAN增强组（1000张）完全对齐。
参考：Ghiasi et al. Simple Copy-Paste is a Strong Data Augmentation Method, 2021
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="splits/train_baseline.txt",
                    help="train split txt (relative to YOLO_ROOT)")
parser.add_argument("--suffix", type=str, default="",
                    help="output dir suffix (e.g. _leak -> gen_copypaste_leak)")
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()

random.seed(42)
np.random.seed(42)

YOLO_ROOT = Path(str(_release_root / 'data/yolo_dataset'))
IMG_DIR = YOLO_ROOT / "images" / "all_real"
LBL_DIR = YOLO_ROOT / "labels" / "all_real"
OUT_IMG = YOLO_ROOT / "images" / ("gen_copypaste" + args.suffix)
OUT_LBL = YOLO_ROOT / "labels" / ("gen_copypaste" + args.suffix)
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

N_GEN = args.n
PAD_RATIO = 0.05          # patch边缘padding
SCALE_RANGE = (0.9, 1.1)  # 随机缩放
X_MEAN, X_STD, X_LO, X_HI = 0.434, 0.071, 0.25, 0.62
Y_LO, Y_HI = 0.08, 0.92
MAX_OVERLAP = 0.40        # 与已有框最大IoU
MAX_ATTEMPT = 8


def read_img(path: Path):
    """兼容中文/空格路径"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def read_boxes(lbl_path: Path):
    """返回 [(xc, yc, w, h) 归一化]"""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def box_xywh_norm_to_pix(box, H, W):
    xc, yc, w, h = box
    return (xc * W, yc * H, w * W, h * H)


def iou_pix(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    x2, y2 = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def main():
    # 加载真实训练图（split 参数化）
    lines = (YOLO_ROOT / args.split).read_text(encoding="utf-8").splitlines()
    imgs = []  # (path, img_array, boxes_pix)
    for line in lines:
        p = Path(line.strip())
        if not p.exists():
            continue
        img = read_img(p)
        if img is None:
            print(f"[WARN] cannot read {p.name}")
            continue
        boxes = read_boxes(LBL_DIR / (p.stem + ".txt"))
        boxes_pix = [box_xywh_norm_to_pix(b, img.shape[0], img.shape[1]) for b in boxes]
        imgs.append((p, img, boxes_pix))
    print(f"[LOAD] {len(imgs)} real train images")

    H, W = imgs[0][1].shape[:2]
    written = 0
    for i in range(N_GEN):
        bg_idx = random.randrange(len(imgs))
        bg_path, bg_img, bg_boxes = imgs[bg_idx]
        canvas = bg_img.copy()
        new_boxes = list(bg_boxes)

        n_paste = random.randint(1, 2)
        for _ in range(n_paste):
            # 随机选源图（避免与背景相同）
            src_idx = random.choice([j for j in range(len(imgs)) if j != bg_idx])
            _, src_img, src_boxes = imgs[src_idx]
            if not src_boxes:
                continue
            src_box = random.choice(src_boxes)
            sx, sy, sw, sh = src_box
            # 裁剪patch（含padding）
            pad_x = sw * PAD_RATIO
            pad_y = sh * PAD_RATIO
            x1 = max(0, int(sx - sw / 2 - pad_x))
            y1 = max(0, int(sy - sh / 2 - pad_y))
            x2 = min(W, int(sx + sw / 2 + pad_x))
            y2 = min(H, int(sy + sh / 2 + pad_y))
            patch = src_img[y1:y2, x1:x2].copy()
            if patch.size == 0:
                continue
            # 随机缩放
            scale = random.uniform(*SCALE_RANGE)
            pw, ph = patch.shape[1], patch.shape[0]
            if scale != 1.0:
                patch = cv2.resize(patch, (max(2, int(pw * scale)), max(2, int(ph * scale))),
                                   interpolation=cv2.INTER_LINEAR)
            pw, ph = patch.shape[1], patch.shape[0]
            # 采样粘贴位置（语义合理：x在中轴带，y沿电缆长度均匀）
            for _ in range(MAX_ATTEMPT):
                xc = float(np.clip(np.random.normal(X_MEAN, X_STD), X_LO, X_HI)) * W
                yc = random.uniform(Y_LO, Y_HI) * H
                # 保证patch完全在图像内
                l = xc - pw / 2
                t = yc - ph / 2
                if l < 0 or l + pw > W or t < 0 or t + ph > H:
                    continue
                new_box = (xc, yc, pw, ph)
                # 重叠检查
                if any(iou_pix(new_box, b) > MAX_OVERLAP for b in new_boxes):
                    continue
                # 粘贴
                canvas[int(t):int(t + ph), int(l):int(l + pw)] = patch
                new_boxes.append(new_box)
                break

        out_name = f"cp_{i:05d}.png"
        cv2.imencode(".png", canvas)[1].tofile(str(OUT_IMG / out_name))
        # 写标签（归一化）
        lbl_lines = [f"0 {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}" for xc, yc, w, h in new_boxes]
        (OUT_LBL / (out_name[:-4] + ".txt")).write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")
        written += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N_GEN}")

    print(f"[DONE] {written} CopyPaste images written to {OUT_IMG.name}")


if __name__ == "__main__":
    main()
