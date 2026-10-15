"""Prepare single-condition GC10 GAN data from the fixed detection train split.

The existing ImageFolder GAN has one class condition per image. For multi-label
images, use the least prevalent class in the fixed train split (ties: class
index). Each of the 1,604 training images occurs exactly once. Hard links avoid
copies and preserve source bytes. Held-out images are excluded by assertion.
"""

from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[1]


import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(str(_release_root))
DROOT = ROOT / "data/gc10_det/yolo"
TRAIN = DROOT / "splits/gc10_train.txt"
TEST = DROOT / "splits/gc10_test.txt"
OUT = ROOT / "data/gc10_det/gan_train_single_condition"
MANIFEST = ROOT / "data/gc10_det/gan_train_single_condition_manifest.json"
CLASSES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
           "rolled_pit", "silk_spot", "waist_folding", "water_spot", "welding_line"]


def read_split(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path(image: Path) -> Path:
    return Path(image.as_posix().replace("/images/", "/labels/")).with_suffix(".txt")


def main() -> None:
    train, test = read_split(TRAIN), read_split(TEST)
    if len(train) != 1604 or len(test) != 688 or len(set(train)) != 1604 or set(train) & set(test):
        raise ValueError("Unexpected GC10 split or train/test overlap")
    image_classes: dict[Path, list[int]] = {}
    for image in train:
        if not image.is_file():
            raise FileNotFoundError(image)
        classes = sorted({int(row.split()[0]) for row in label_path(image).read_text(encoding="utf-8").splitlines() if row.strip()})
        if not classes or any(c < 0 or c >= len(CLASSES) for c in classes):
            raise ValueError(f"Invalid labels for {image}")
        image_classes[image] = classes
    prevalence = Counter(c for classes in image_classes.values() for c in classes)
    assignments = {image: min(classes, key=lambda c: (prevalence[c], c)) for image, classes in image_classes.items()}
    if len(assignments) != 1604:
        raise AssertionError("Image assignment is not one-to-one")
    per_class = Counter(assignments.values())
    if any(per_class[c] == 0 for c in range(len(CLASSES))):
        raise ValueError("Some GC10 class has no GAN training image")

    for image, cls in assignments.items():
        dest = OUT / CLASSES[cls] / image.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if not os.path.samefile(image, dest):
                raise FileExistsError(f"Conflicting prepared image: {dest}")
        else:
            os.link(image, dest)
    prepared = sorted(path for path in OUT.glob("*/*") if path.is_file())
    if len(prepared) != 1604 or {path.name for path in prepared} != {path.name for path in train}:
        raise AssertionError("Prepared GAN images do not match the fixed train split")
    if any(path.name in {image.name for image in test} for path in prepared):
        raise AssertionError("Held-out image found in GAN training directory")

    manifest = {
        "source_split": str(TRAIN), "held_out_split": str(TEST),
        "source_train_images": len(train), "held_out_test_images": len(test),
        "prepared_images": len(prepared),
        "single_condition_rule": "least prevalent class among the image's train labels, tie by class index",
        "original_class_prevalence": {CLASSES[c]: prevalence[c] for c in range(len(CLASSES))},
        "assigned_condition_counts": {CLASSES[c]: per_class[c] for c in range(len(CLASSES))},
        "multilabel_image_count": sum(len(classes) > 1 for classes in image_classes.values()),
        "assignments": [{"image": str(image), "condition": CLASSES[assignments[image]],
                         "original_classes": [CLASSES[c] for c in classes]}
                        for image, classes in sorted(image_classes.items())],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "assignments"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
