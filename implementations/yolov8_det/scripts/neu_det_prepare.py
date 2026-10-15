# -*- coding: utf-8 -*-
"""NEU-DET 数据集准备：VOC XML -> YOLO txt，构建 train/val splits。
官方划分：train 1440 (6类x240) / validation 360 (6类x60)。
协议与电缆一致：GAN 只在 train 1440 上训练（第二数据集修复信息泄露）。
"""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

NEU_ROOT = Path(str(_release_root / 'data/neu_det/NEU-DET'))
YOLO_ROOT = Path(str(_release_root / 'data/neu_det/yolo'))

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}


def parse_xml(xml_path: Path):
    """返回 [(cls_id, x_center, y_center, w, h)]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W, H = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        xc = (xmin + xmax) / 2 / W
        yc = (ymin + ymax) / 2 / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        boxes.append((CLS2ID[name], xc, yc, w, h))
    return boxes


def convert(split: str):
    """split: 'train' | 'validation'"""
    src_img_dir = NEU_ROOT / split / "images"
    src_xml_dir = NEU_ROOT / split / "annotations"
    out_img_dir = YOLO_ROOT / "images" / ("train" if split == "train" else "val")
    out_lbl_dir = YOLO_ROOT / "labels" / ("train" if split == "train" else "val")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    split_lines = []
    n_img = n_box = 0
    for cls_dir in sorted(src_img_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.glob("*.jpg")):
            xml = src_xml_dir / (img.stem + ".xml")
            if not xml.exists():
                print(f"[WARN] missing xml: {xml}")
                continue
            boxes = parse_xml(xml)
            if not boxes:
                print(f"[WARN] no boxes: {img.name}")
                continue
            # 复制图片（平铺，保留原名）
            shutil.copy(img, out_img_dir / img.name)
            # 写 YOLO label
            lbl = out_lbl_dir / (img.stem + ".txt")
            lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
            lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            split_lines.append(str(out_img_dir / img.name))
            n_img += 1
            n_box += len(boxes)
    out_split = YOLO_ROOT / "splits"
    out_split.mkdir(parents=True, exist_ok=True)
    tag = "train" if split == "train" else "val"
    (out_split / f"neu_{tag}.txt").write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    print(f"[{split}] images={n_img}, boxes={n_box}")


def build_yaml():
    names = {i: c for i, c in enumerate(CLASSES)}
    yaml_text = (
        "# NEU-DET Steel Surface Defect Detection (6 classes)\n"
        f"train: {YOLO_ROOT / 'splits' / 'neu_train.txt'}\n"
        f"val: {YOLO_ROOT / 'splits' / 'neu_val.txt'}\n"
        "nc: 6\n"
        "names:\n" + "\n".join(f"  {i}: {c}" for i, c in names.items()) + "\n"
    )
    (YOLO_ROOT / "neu_det.yaml").write_text(yaml_text, encoding="utf-8")
    print("[yaml] neu_det.yaml written")


if __name__ == "__main__":
    convert("train")
    convert("validation")
    build_yaml()
