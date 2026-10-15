"""
Leak-fix data preparation (投稿方案硬伤1修复 · 第0步)

背景:
  原 245 张 all_real 中有 4 张是分类集图片的字节级重复副本(MD5相同),
  其中 72163236*(train) 与 凹点/151721-825(val) 构成跨训练/测试集重复,
  导致检测评估本身存在数据泄漏(与GAN无关的第二重泄漏)。

本脚本:
  1. 以 data/cable_defect_classified (241张唯一图, 全部带类别标签) 为唯一真源
  2. 按 8:2 分层划分 (seed=42): train ~193 / val ~48
  3. 输出:
     - data/yolo_dataset/splits_leakfix/train.txt  (YOLO用, all_real绝对路径)
     - data/yolo_dataset/splits_leakfix/val.txt
     - data/yolo_dataset/splits_leakfix/split_info.json
     - data/cable_leakfix_train/<类别>/...  (GAN训练用 ImageFolder, 仅train子集)
  4. 校验: MD5级无重复、train/val无交集、类别分布报告
"""
import os
import json
import random
import shutil
import hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CLS_DIR = os.path.join(ROOT, "data/cable_defect_classified")
ALL_REAL = os.path.join(ROOT, "data/yolo_dataset/images/all_real")
OUT_SPLIT = os.path.join(ROOT, "data/yolo_dataset/splits_leakfix")
OUT_GAN = os.path.join(ROOT, "data/cable_leakfix_train")

SEED = 42
RATIO = 0.8

CLASSES = ["凹点", "划痕", "形变", "鼓包"]  # ImageFolder 字母序无关, 这里固定中文目录名


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_SPLIT, exist_ok=True)

    # 1. 收集唯一图 (分类集即唯一真源)
    per_class = {}
    seen_md5 = {}
    dup_dropped = []
    for c in CLASSES:
        d = os.path.join(CLS_DIR, c)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        kept = []
        for f in files:
            p = os.path.join(d, f)
            h = md5(p)
            if h in seen_md5:
                dup_dropped.append((os.path.join(c, f), seen_md5[h]))
                continue
            seen_md5[h] = os.path.join(c, f)
            kept.append(f)
        per_class[c] = kept

    total = sum(len(v) for v in per_class.values())
    print(f"唯一图总数: {total} (分类集内去重剔除 {len(dup_dropped)} 张)")
    for c, fs in per_class.items():
        print(f"  {c}: {len(fs)}")

    # 2. 分层 8:2
    random.seed(SEED)
    train_map, val_map = {}, []  # train_map: class->files
    for c in CLASSES:
        fs = sorted(per_class[c])
        random.shuffle(fs)
        n_tr = round(len(fs) * RATIO)
        train_map[c] = sorted(fs[:n_tr])
        val_map.extend((c, f) for f in sorted(fs[n_tr:]))

    n_tr = sum(len(v) for v in train_map.values())
    n_va = len(val_map)
    print(f"划分: train {n_tr} / val {n_va} (ratio {RATIO}, seed {SEED})")

    # 3. 写 splits (YOLO 用 all_real 绝对路径)
    with open(os.path.join(OUT_SPLIT, "train.txt"), "w", encoding="utf-8") as f:
        for c in CLASSES:
            for name in train_map[c]:
                f.write(os.path.join(ALL_REAL, name).replace("\\", "/") + "\n")
    with open(os.path.join(OUT_SPLIT, "val.txt"), "w", encoding="utf-8") as f:
        for c, name in sorted(val_map):
            f.write(os.path.join(ALL_REAL, name).replace("\\", "/") + "\n")

    # 4. 建 GAN ImageFolder (仅train, 拷贝)
    if os.path.isdir(OUT_GAN):
        shutil.rmtree(OUT_GAN)
    for c in CLASSES:
        dst_dir = os.path.join(OUT_GAN, c)
        os.makedirs(dst_dir, exist_ok=True)
        for name in train_map[c]:
            shutil.copy2(os.path.join(CLS_DIR, c, name), os.path.join(dst_dir, name))

    # 5. 校验
    tr_names = set()
    for c in CLASSES:
        for n in train_map[c]:
            tr_names.add(n)
    va_names = set(n for _, n in val_map)
    assert not (tr_names & va_names), "train/val 交集非空!"
    all_md5 = set()
    for c in CLASSES:
        for n in train_map[c] + [n for cc, n in val_map if cc == c]:
            h = md5(os.path.join(CLS_DIR, c, n))
            assert h not in all_md5, f"MD5重复: {c}/{n}"
            all_md5.add(h)

    info = {
        "seed": SEED,
        "ratio": RATIO,
        "unique_total": total,
        "train": {c: train_map[c] for c in CLASSES},
        "val": {c: [n for cc, n in val_map if cc == c] for c in CLASSES},
        "note": "4张哈希副本已剔除; 72163236/8b07a2ce/8a054ace/80d182b4 不再使用",
    }
    with open(os.path.join(OUT_SPLIT, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n[校验通过] train/val 无交集; 全集 MD5 无重复")
    print("输出:")
    print(" ", OUT_SPLIT)
    print(" ", OUT_GAN)


if __name__ == "__main__":
    main()
