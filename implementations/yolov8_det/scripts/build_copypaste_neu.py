# -*- coding: utf-8 -*-
"""NEU-DET CopyPaste 数据增强：复用真实图的缺陷实例拼到其他图。
NEU-DET 6类多实例：每张真实图有1-4个bbox，粘贴时随机选实例复制到另一张。
输出 1000 张 200x200 灰度增强图 + 精确 YOLO 标注。
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

NEU_YOLO = Path(str(_release_root / 'data/neu_det/yolo'))
SRC_IMG = NEU_YOLO / "images" / "train"
SRC_LBL = NEU_YOLO / "labels" / "train"
OUT_IMG = NEU_YOLO / "images" / "gen_copypaste"
OUT_LBL = NEU_YOLO / "labels" / "gen_copypaste"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

N_GEN = 1000
SCALE_RANGE = (0.85, 1.0)  # 仅缩小防止patch超出画布
MAX_OVERLAP = 0.40
MAX_ATTEMPT = 8


def read_img(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def read_boxes(lbl):
    boxes = []
    if not lbl.exists():
        return boxes
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        c, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((c, xc, yc, w, h))
    return boxes


def iou(a, b):
    # a/b: (xc, yc, w, h) normalized
    ax, ay, aw, ah = a[1], a[2], a[3], a[4]
    bx, by, bw, bh = b[1], b[2], b[3], b[4]
    x1, y1 = max(ax - aw/2, bx - bw/2), max(ay - ah/2, by - ah/2)
    x2, y2 = min(ax + aw/2, bx + bw/2), min(ay + ah/2, by + bh/2)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0


def main():
    imgs = []  # (path, img, boxes_norm)
    for p in sorted(SRC_IMG.glob("*.jpg")):
        img = read_img(p)
        if img is None: continue
        boxes = read_boxes(SRC_LBL / (p.stem + ".txt"))
        imgs.append((p, img, boxes))
    print(f"[LOAD] {len(imgs)} neu train imgs")
    H, W = imgs[0][1].shape[:2]

    written = 0
    for i in range(N_GEN):
        bg_idx = random.randrange(len(imgs))
        bg_path, bg_img, bg_boxes = imgs[bg_idx]
        canvas = bg_img.copy()
        new_boxes = list(bg_boxes)

        n_paste = random.randint(1, 2)
        for _ in range(n_paste):
            src_idx = random.choice([j for j in range(len(imgs)) if j != bg_idx])
            _, src_img, src_boxes = imgs[src_idx]
            if not src_boxes:
                continue
            src_box = random.choice(src_boxes)
            c, xc, yc, w, h = src_box
            x1, y1 = max(0, int((xc - w/2)*W)), max(0, int((yc - h/2)*H))
            x2, y2 = min(W, int((xc + w/2)*W)), min(H, int((yc + h/2)*H))
            patch = src_img[y1:y2, x1:x2].copy()
            if patch.size == 0: continue
            scale = random.uniform(*SCALE_RANGE)
            if scale != 1.0:
                patch = cv2.resize(patch, (max(2, int(patch.shape[1]*scale)), max(2, int(patch.shape[0]*scale))),
                                   interpolation=cv2.INTER_LINEAR)
            pw, ph = patch.shape[1], patch.shape[0]
            # 跳过超界patch (NEU-DET部分bbox横跨全图)
            if pw >= W or ph >= H:
                continue
            placed = False
            for _ in range(MAX_ATTEMPT):
                nxc = random.uniform(pw/W/2, 1 - pw/W/2)
                nyc = random.uniform(ph/H/2, 1 - ph/H/2)
                new_box = (c, nxc, nyc, pw/W, ph/H)
                if any(iou(new_box, b) > MAX_OVERLAP for b in new_boxes):
                    continue
                l, t = int((nxc - pw/W/2)*W), int((nyc - ph/H/2)*H)
                canvas[t:t+ph, l:l+pw] = patch
                new_boxes.append(new_box)
                placed = True
                break
            # 即便没成功放置也无所谓

        out_name = f"cp_{i:05d}.jpg"
        cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(OUT_IMG / out_name))
        lbl_lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in new_boxes]
        (OUT_LBL / (out_name[:-4] + ".txt")).write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")
        written += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_GEN}")
    print(f"[DONE] {written} NEU CopyPaste images")


if __name__ == "__main__":
    main()
