"""
build_dataset.py
================
一站式构建 4.3 检测实验数据集：
1) 245张真实图按 80/20 随机划分（固定种子，可复现）
2) CV 自动标注：行级方差法找缺陷垂直范围 -> YOLO 格式 bbox
3) 复制/软链接真实图与各实验生成图到统一 YOLO 目录
4) 生成每个实验组的 train.txt / val.txt 列表
5) 写 data.yaml

Usage:
  python build_dataset.py
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
import cv2

# ===== 配置 =====
PROJECT_ROOT = Path(str(_release_root))
REAL_DIR = PROJECT_ROOT / "data" / "cable_defect" / "defect"
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
REAL_OUT = YOLO_ROOT / "images" / "all_real"
GEN_BASE = PROJECT_ROOT / "implementations" / "wgan_gp" / "experiments"
GEN_BASE2 = PROJECT_ROOT / "implementations" / "experiments"

# 实验组 -> 生成图目录
EXPERIMENTS = {
    "baseline":   {"gen_dir": None,                       "n_gen": 0,    "label_strategy": "real_only"},
    "E5_tradit":  {"gen_dir": GEN_BASE / "traditional_aug",       "n_gen": 1000, "label_strategy": "from_generated"},
    "E4_dcgan":   {"gen_dir": GEN_BASE / "E4" / "images",         "n_gen": 1000, "label_strategy": "from_generated"},
    "E2_wgan_gp": {"gen_dir": GEN_BASE / "vanilla_wgan_gp",       "n_gen": 1000, "label_strategy": "from_generated"},
    "E1_sp":      {"gen_dir": GEN_BASE / "E1" / "generated",      "n_gen": 1000, "label_strategy": "from_generated"},
    "E7_clip":    {"gen_dir": GEN_BASE2 / "E7" / "generated_flat", "n_gen": 1000, "label_strategy": "from_generated"},
}

SPLIT_SEED = 42
TRAIN_RATIO = 0.8
CLASSES = ["defect"]
NUM_CLASSES = len(CLASSES)

# ===== 步骤 1：划分 train/test =====
def split_real_files(seed=SPLIT_SEED, ratio=TRAIN_RATIO):
    files = sorted([f for f in REAL_DIR.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    random.seed(seed)
    random.shuffle(files)
    n_train = int(len(files) * ratio)
    return files[:n_train], files[n_train:]

# ===== 步骤 2：CV 自动标注（行级方差法） =====
def annotate_image(img_path, debug=False):
    """
    算法（针对电缆缺陷：垂直电缆+水平带状缺陷）：
      1) Otsu 找电缆（最大连通暗区域）
      2) 重腐蚀电缆 mask 排除边缘（沿宽度方向腐蚀 25%，沿长度 5%）
      3) 每行在腐蚀后的电缆内计算灰度方差
      4) 平滑 row_std 后找高方差行段
      5) 最长高方差段 = 缺陷垂直范围；该段内电缆水平范围 = 缺陷水平范围
      6) 加 padding 输出 YOLO bbox
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    h, w = img.shape

    # 1) 找电缆
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n_labels <= 1:
        return _fallback_bbox(0, 0, w, h, w, h)
    # 找最瘦长（aspect ratio 最高）的暗区域 = 电缆
    # 排除太小的和高宽比 < 2 的（不像电缆）
    best_idx = -1
    best_score = 0  # aspect ratio * sqrt(area)
    for i in range(1, n_labels):
        x, y, ww, hh, area = stats[i]
        if area < h * w * 0.005:  # 太小
            continue
        if hh < h * 0.3:  # 高度不足
            continue
        aspect = hh / (ww + 1e-6)
        if aspect < 2:  # 不够瘦长
            continue
        # 评分：瘦长度 × 面积开方
        score = aspect * (area ** 0.5)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0:
        # 回退：取最大面积
        best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cable_idx = best_idx
    cx, cy, cw, ch, _ = stats[cable_idx]
    if cw < 20 or ch < 50:
        return _fallback_bbox(cx, cy, max(cw, 100), max(ch, 300), w, h)
    cable_mask = (labels == cable_idx)

    # 2) 重腐蚀：宽度方向腐蚀 25%，长度方向腐蚀 5%
    erode_w = max(11, cw // 4)
    erode_h = max(5, ch // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_w, erode_h))
    eroded_bool = cv2.erode(cable_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    if eroded_bool.sum() < 200:
        return ""

    # 3) 每行在腐蚀后的电缆内计算灰度方差
    row_std = np.zeros(ch, dtype=np.float32)
    for r in range(ch):
        row_mask = eroded_bool[cy + r, cx:cx + cw]  # 限定到电缆水平范围
        if row_mask.sum() < 5:
            row_std[r] = 0
            continue
        row_pixels = img[cy + r, cx:cx + cw][row_mask]
        row_std[r] = row_pixels.std()
    if not np.any(row_std > 0):
        return ""

    # 4) 平滑 row_std 后找高方差段
    if row_std.max() < 1:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    k = 7
    smooth_std = np.convolve(row_std, np.ones(k) / k, mode='same')
    nz = smooth_std[smooth_std > 0]
    if len(nz) < 10:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    med = np.median(nz)
    q75, q25 = np.percentile(nz, [75, 25])
    iqr = q75 - q25
    # 阈值放宽：median + 0.3*iqr (只要比典型方差略高就算)
    th = med + 0.3 * iqr
    high_var = smooth_std > th

    # 5) 找高方差连续段；优先选峰值最高的那段，不是最长
    runs = []
    start = None
    for i, v in enumerate(high_var):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(high_var) - 1, len(high_var) - start))
    if not runs:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    # 限制缺陷高度不超过电缆长度的 70%
    max_allowed = int(ch * 0.7)
    valid = [(s, e, ln) for s, e, ln in runs if ln <= max_allowed and ln >= 5]
    if not valid:
        valid = [(s, e, ln) for s, e, ln in runs if ln >= 5]
    if not valid:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    # 优先选含最高峰的段（用段内最大smooth_std作为评分）
    def seg_score(s, e, ln):
        return float(smooth_std[s:e+1].max())
    best = max(valid, key=lambda x: seg_score(x[0], x[1], x[2]))
    dy, d_end = best[0], best[1]
    # 垂直 padding（更慷慨）
    pad_v = max(20, ch // 30)
    dy_p = max(0, dy - pad_v)
    d_end_p = min(ch - 1, d_end + pad_v)
    defect_h = d_end_p - dy_p + 1

    # 6) 水平方向：在缺陷垂直范围内，取腐蚀后mask的列范围
    defect_band = eroded_bool[cy + dy_p:cy + d_end_p + 1, cx:cx + cw]
    if defect_band.sum() < 50:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    col_sum = defect_band.sum(axis=0)
    cols_with_cable = np.where(col_sum > 0)[0]
    if len(cols_with_cable) < 5:
        return _fallback_bbox(cx, cy, cw, ch, w, h)
    dx = int(cols_with_cable.min())
    d_w_end = int(cols_with_cable.max())
    pad_h = max(20, cw // 8)
    dx_p = max(0, dx - pad_h)
    d_w_end_p = min(cw - 1, d_w_end + pad_h)
    defect_w = d_w_end_p - dx_p + 1

    # 7) 转回原图坐标 + 归一化
    abs_x = cx + dx_p
    abs_y = cy + dy_p
    x_c = (abs_x + defect_w / 2) / w
    y_c = (abs_y + defect_h / 2) / h
    bw_norm = defect_w / w
    bh_norm = defect_h / h
    if debug:
        print(f"  cable=({cx},{cy},{cw}x{ch})  defect_y=[{dy},{d_end}] defect_h={defect_h}  norm=({x_c:.3f},{y_c:.3f},{bw_norm:.3f},{bh_norm:.3f})")
    return f"0 {x_c:.6f} {y_c:.6f} {bw_norm:.6f} {bh_norm:.6f}\n"

def _fallback_bbox(cx, cy, cw, ch, img_w, img_h):
    """
    兜底 bbox：当无法精确检测缺陷时，在电缆中央放一个合理大小的 bbox。
    大小：宽=cable_width, 高=0.3*cable_height（占电缆的30%）。
    """
    bw = cw
    bh = int(ch * 0.3)
    # 居中放在电缆中段
    bx = cx
    by = cy + (ch - bh) // 2
    x_c = (bx + bw / 2) / img_w
    y_c = (by + bh / 2) / img_h
    bw_norm = bw / img_w
    bh_norm = bh / img_h
    return f"0 {x_c:.6f} {y_c:.6f} {bw_norm:.6f} {bh_norm:.6f}\n"

# ===== 步骤 2.5：生成图中央弱监督 =====
# GAN 生成图无 ground-truth 缺陷位置，采用"电缆中央区域"作为弱监督 bbox
# 基于先验：512x512 生成图中电缆通常在 x=[0.40, 0.60], y=[0.20, 0.80] 的中央竖条区域
GEN_CENTRAL_BBOX = (0.50, 0.50, 0.20, 0.60)  # (x_c, y_c, w_norm, h_norm)

def annotate_central(img_path, debug=False):
    """
    对生成图统一标一个覆盖中央电缆区域的 bbox（弱监督）。
    所有生成图都标相同位置，避免噪声干扰。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    x_c, y_c, bw_n, bh_n = GEN_CENTRAL_BBOX
    if debug:
        print(f"  central bbox: x_c={x_c}, y_c={y_c}, w={bw_n}, h={bh_n}  (img {w}x{h})")
    return f"0 {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}\n"

# ===== 步骤 3：组织真实图到统一目录 =====
def stage_real_images(train_files, test_files):
    real_dir = REAL_OUT
    real_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = YOLO_ROOT / "labels" / "all_real"
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STAGE] 复制 {len(train_files)} train + {len(test_files)} val 真实图...")
    for f in train_files + test_files:
        dst = real_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    print(f"[ANNOTATE] 标注所有真实图...")
    annotated = 0
    for f in train_files + test_files:
        label_path = labels_dir / (f.stem + ".txt")
        if not label_path.exists() or label_path.stat().st_size == 0:
            ann = annotate_image(f, debug=False)
            label_path.write_text(ann, encoding="utf-8")
        if label_path.stat().st_size > 0:
            annotated += 1
    print(f"[ANNOTATE] {annotated}/{len(train_files)+len(test_files)} 张成功标注")
    return annotated

# ===== 步骤 4：处理每个实验组 =====
def build_experiment_groups(train_files, test_files):
    """为每个实验组生成 train.txt + (可选)生成图目录"""
    images_all = YOLO_ROOT / "images"
    labels_all = YOLO_ROOT / "labels"
    images_all.mkdir(parents=True, exist_ok=True)
    labels_all.mkdir(parents=True, exist_ok=True)
    val_set = {f.name for f in test_files}
    train_set = {f.name for f in train_files}

    splits_dir = YOLO_ROOT / "splits"
    splits_dir.mkdir(exist_ok=True)

    # 写 val.txt (所有组共用)
    val_list = sorted(REAL_OUT.glob("*.jpg")) + sorted(REAL_OUT.glob("*.png"))
    val_list = [p for p in val_list if p.name in val_set]
    (splits_dir / "val.txt").write_text(
        "\n".join(str(p.relative_to(YOLO_ROOT)).replace("\\", "/") for p in val_list),
        encoding="utf-8"
    )
    print(f"[SPLIT] val.txt: {len(val_list)} 张")

    for grp_name, cfg in EXPERIMENTS.items():
        train_imgs = []
        for f in train_files:
            rel = REAL_OUT.relative_to(YOLO_ROOT) / f.name
            train_imgs.append(str(rel).replace("\\", "/"))
        if cfg["gen_dir"] is not None and Path(cfg["gen_dir"]).exists():
            gen_files = sorted(Path(cfg["gen_dir"]).glob("*.png")) + sorted(Path(cfg["gen_dir"]).glob("*.jpg"))
            if cfg["n_gen"] > 0 and len(gen_files) > cfg["n_gen"]:
                gen_files = gen_files[:cfg["n_gen"]]
            grp_img_dir = images_all / f"gen_{grp_name}"
            grp_lbl_dir = labels_all / f"gen_{grp_name}"
            grp_img_dir.mkdir(parents=True, exist_ok=True)
            grp_lbl_dir.mkdir(parents=True, exist_ok=True)
            print(f"[STAGE] {grp_name}: 处理 {len(gen_files)} 张生成图 (中央弱监督)...")
            n_ann = 0
            for gf in gen_files:
                dst_img = grp_img_dir / gf.name
                if not dst_img.exists():
                    shutil.copy2(gf, dst_img)
                dst_lbl = grp_lbl_dir / (gf.stem + ".txt")
                if not dst_lbl.exists() or dst_lbl.stat().st_size == 0:
                    # 生成图用中央弱监督（不调用真实图算法）
                    ann = annotate_central(gf, debug=False)
                    dst_lbl.write_text(ann, encoding="utf-8")
                if dst_lbl.stat().st_size > 0:
                    n_ann += 1
                    rel = grp_img_dir.relative_to(YOLO_ROOT) / gf.name
                    train_imgs.append(str(rel).replace("\\", "/"))
            print(f"  {grp_name}: {n_ann}/{len(gen_files)} 生成图弱监督标注完成")
        (splits_dir / f"train_{grp_name}.txt").write_text(
            "\n".join(train_imgs), encoding="utf-8"
        )
        print(f"[SPLIT] train_{grp_name}.txt: {len(train_imgs)} 张")

def write_data_yaml():
    yaml_path = YOLO_ROOT / "data.yaml"
    content = f"""# Cable Defect Detection - YOLOv8 Dataset
path: {str(YOLO_ROOT).replace(chr(92), '/')}
train: splits/train_baseline.txt
val: splits/val.txt
nc: {NUM_CLASSES}
names: {CLASSES}
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[YAML] {yaml_path}")

# ===== 主流程 =====
def main():
    print("=" * 60)
    print("电缆缺陷 YOLO 数据集构建（行级方差法）")
    print("=" * 60)
    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    train_files, test_files = split_real_files()
    print(f"[SPLIT] train={len(train_files)}  test={len(test_files)}")

    (YOLO_ROOT / "splits").mkdir(exist_ok=True)
    split_info = {
        "seed": SPLIT_SEED, "ratio": TRAIN_RATIO,
        "train": [f.name for f in train_files],
        "val": [f.name for f in test_files],
    }
    (YOLO_ROOT / "splits" / "split_info.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stage_real_images(train_files, test_files)
    build_experiment_groups(train_files, test_files)
    write_data_yaml()

    print("\n" + "=" * 60)
    print("✅ 数据集构建完成")
    print(f"   YOLO 根目录: {YOLO_ROOT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
