# -*- coding: utf-8 -*-
"""Prepare a leakage-controlled GC10-DET external validation split.

Input is the Dataset Ninja Supervisely-format export. The script:
1. validates every rectangle and class;
2. removes pixel-identical images, including rotations/reflections;
3. creates a deterministic 70/30 multilabel-aware split;
4. hard-links images and writes YOLO labels/split files;
5. records a complete audit manifest.

The held-out split must not be used by GANs, teachers, sample selection, or
detector training.
"""

from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(str(_release_root))
RAW = ROOT / "data/gc10_det_raw/extracted/ds"
OUT = ROOT / "data/gc10_det/yolo"
SEED = 20260922
TEST_FRACTION = 0.30

CLASSES = [
    "crease",
    "crescent_gap",
    "inclusion",
    "oil_spot",
    "punching_hole",
    "rolled_pit",
    "silk_spot",
    "waist folding",
    "water_spot",
    "welding_line",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def image_transforms(array: np.ndarray):
    """Yield the eight dihedral transforms used for exact duplicate checks."""
    for k in range(4):
        rotated = np.rot90(array, k)
        yield rotated
        yield np.fliplr(rotated)


def canonical_thumbnail(path: Path) -> str:
    """Fast candidate hash invariant to right-angle rotations/reflections."""
    with Image.open(path) as image:
        gray = image.convert("L")
        variants = []
        for transpose in (
            None,
            Image.Transpose.FLIP_LEFT_RIGHT,
            Image.Transpose.ROTATE_90,
            Image.Transpose.TRANSVERSE,
            Image.Transpose.ROTATE_180,
            Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_270,
            Image.Transpose.TRANSPOSE,
        ):
            transformed = (gray if transpose is None else gray.transpose(transpose)).resize(
                (64, 64), Image.Resampling.BILINEAR)
            variants.append(hashlib.sha256(transformed.tobytes()).hexdigest())
    return min(variants)


def exact_equivalent(left: Path, right: Path) -> bool:
    with Image.open(left) as image:
        a = np.asarray(image.convert("L"))
    with Image.open(right) as image:
        b = np.asarray(image.convert("L"))
    return any(candidate.shape == b.shape and np.array_equal(candidate, b)
               for candidate in image_transforms(a))


def load_record(image_path: Path) -> dict:
    annotation_path = RAW / "ann" / f"{image_path.name}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    width = int(annotation["size"]["width"])
    height = int(annotation["size"]["height"])
    boxes = []
    for obj in annotation.get("objects", []):
        class_name = obj["classTitle"]
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {class_name!r} in {annotation_path}")
        (x1, y1), (x2, y2) = obj["points"]["exterior"]
        x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
        y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box in {annotation_path}: {(x1, y1, x2, y2)}")
        boxes.append({
            "class_name": class_name,
            "class_id": CLASS_TO_ID[class_name],
            "xywhn": [
                (x1 + x2) / (2.0 * width),
                (y1 + y2) / (2.0 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ],
        })
    return {
        "image": image_path,
        "annotation": annotation_path,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": sorted({box["class_id"] for box in boxes}),
    }


def remove_duplicates(records: list[dict]) -> tuple[list[dict], list[dict]]:
    candidate_groups: dict[str, list[dict]] = defaultdict(list)
    for index, record in enumerate(records, 1):
        candidate_groups[canonical_thumbnail(record["image"])].append(record)
        if index % 250 == 0:
            print(f"[HASH] {index}/{len(records)}", flush=True)

    kept = []
    duplicates = []
    for group in candidate_groups.values():
        representatives = []
        for record in sorted(group, key=lambda item: item["image"].name):
            match = next((item for item in representatives
                          if exact_equivalent(item["image"], record["image"])), None)
            if match is None:
                representatives.append(record)
                kept.append(record)
            else:
                duplicates.append({
                    "kept": match["image"].name,
                    "removed": record["image"].name,
                    "kept_boxes": len(match["boxes"]),
                    "removed_boxes": len(record["boxes"]),
                })
    return sorted(kept, key=lambda item: item["image"].name), duplicates


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratify by each image's rarest label, then use deterministic shuffling."""
    image_class_counts = Counter(label for record in records for label in record["labels"])
    strata: dict[int, list[dict]] = defaultdict(list)
    empty = []
    for record in records:
        if not record["labels"]:
            empty.append(record)
            continue
        rarest = min(record["labels"], key=lambda label: (image_class_counts[label], label))
        strata[rarest].append(record)

    rng = random.Random(SEED)
    train, test = [], []
    for label in sorted(strata):
        group = sorted(strata[label], key=lambda item: item["image"].name)
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_FRACTION))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(empty)
    n_empty_test = round(len(empty) * TEST_FRACTION)
    test.extend(empty[:n_empty_test])
    train.extend(empty[n_empty_test:])
    return sorted(train, key=lambda item: item["image"].name), sorted(
        test, key=lambda item: item["image"].name)


def class_summary(records: list[dict]) -> dict:
    images = Counter()
    boxes = Counter()
    for record in records:
        for label in record["labels"]:
            images[CLASSES[label]] += 1
        for box in record["boxes"]:
            boxes[box["class_name"]] += 1
    return {
        name: {"images": images[name], "boxes": boxes[name]}
        for name in CLASSES
    }


def write_dataset(train: list[dict], test: list[dict], duplicates: list[dict],
                  excluded_unannotated: list[dict]) -> None:
    image_dir = OUT / "images/all"
    label_dir = OUT / "labels/all"
    split_dir = OUT / "splits"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    all_records = train + test
    for record in all_records:
        destination = image_dir / record["image"].name
        if not destination.exists():
            try:
                os.link(record["image"], destination)
            except OSError:
                shutil.copy2(record["image"], destination)
        lines = [
            f"{box['class_id']} " + " ".join(f"{value:.8f}" for value in box["xywhn"])
            for box in record["boxes"]
        ]
        (label_dir / f"{record['image'].stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def paths(records_for_split: list[dict]) -> list[str]:
        return [str((image_dir / record["image"].name).resolve())
                for record in records_for_split]

    train_paths = paths(train)
    test_paths = paths(test)
    (split_dir / "gc10_train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    (split_dir / "gc10_test.txt").write_text("\n".join(test_paths) + "\n", encoding="utf-8")

    yaml = (
        f"train: {(split_dir / 'gc10_train.txt').as_posix()}\n"
        f"val: {(split_dir / 'gc10_test.txt').as_posix()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES!r}\n"
    )
    (OUT / "gc10.yaml").write_text(yaml, encoding="utf-8")

    train_names = {record["image"].name for record in train}
    test_names = {record["image"].name for record in test}
    if train_names & test_names:
        raise RuntimeError("Train/test filename overlap")

    manifest = {
        "source": "Dataset Ninja GC10-DET Supervisely export",
        "source_url": "https://datasetninja.com/gc10-det",
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "raw_images": 2300,
        "kept_images": len(all_records),
        "removed_duplicates": len(duplicates),
        "duplicate_details": duplicates,
        "removed_unannotated": len(excluded_unannotated),
        "unannotated_details": [record["image"].name for record in excluded_unannotated],
        "unannotated_kept": 0,
        "train_images": len(train),
        "test_images": len(test),
        "train": class_summary(train),
        "test": class_summary(test),
        "classes": CLASSES,
        "split_overlap": 0,
        "held_out_policy": (
            "Test images are excluded from GAN training, teacher training, "
            "pseudo-labeling, sample selection, and detector training."
        ),
    }
    (OUT / "gc10_split_audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    image_paths = sorted((RAW / "img").glob("*.jpg"))
    if len(image_paths) != 2300:
        raise ValueError(f"Expected 2300 source images, found {len(image_paths)}")
    records = [load_record(path) for path in image_paths]
    kept, duplicates = remove_duplicates(records)
    excluded_unannotated = [record for record in kept if not record["boxes"]]
    annotated = [record for record in kept if record["boxes"]]
    train, test = split_records(annotated)
    write_dataset(train, test, duplicates, excluded_unannotated)


if __name__ == "__main__":
    main()
