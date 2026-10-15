# -*- coding: utf-8 -*-
"""Leakage-controlled GC10-DET box-guided CopyPaste baseline.

Only the fixed 1,604-image training split supplies donors and backgrounds.
The 688-image held-out split is referenced only by the evaluation YAML used
after each fixed 30-epoch detector run.  Each synthetic image contains one
class-balanced rectangular defect crop and preserves all background labels.
"""

from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, _release_sysconfig.get_paths()['purelib'])
from scipy.stats import wilcoxon
from ultralytics import YOLO


ROOT = Path(str(_release_root))
DROOT = ROOT / "data/gc10_det/yolo"
TRAIN_SPLIT = DROOT / "splits/gc10_train.txt"
TEST_SPLIT = DROOT / "splits/gc10_test.txt"
OUT_IMAGES = DROOT / "images/gen_copypaste_q300"
OUT_LABELS = DROOT / "labels/gen_copypaste_q300"
CP_SPLIT = DROOT / "splits/gc10_copypaste_q300.txt"
TRAIN_YAML = DROOT / "gc10_copypaste_q300_train.yaml"
EVAL_YAML = DROOT / "gc10_copypaste_q300_eval.yaml"
MANIFEST = DROOT / "gc10_copypaste_manifest.json"
RESULTS = DROOT / "gc10_copypaste_results.json"
STATISTICS = DROOT / "gc10_copypaste_statistics.json"
PHASE1 = DROOT / "gc10_phase1_results.json"
RUNS = ROOT / "implementations/yolov8_det/runs_gc10_copypaste_strict_b4"
PYTHON = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']

CLASSES = [
    "crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
    "rolled_pit", "silk_spot", "waist folding", "water_spot", "welding_line",
]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTION_SEED = 20260922
Q = 300
EPOCHS = 30
BATCH = 4
IMGSZ = 320


def read_split(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path(image_path: Path) -> Path:
    text = image_path.as_posix().replace("/images/", "/labels/")
    return Path(text).with_suffix(".txt")


def read_boxes(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in label_path(image_path).read_text(encoding="utf-8").splitlines():
        c, x, y, w, h = line.split()[:5]
        rows.append((int(c), float(x), float(y), float(w), float(h)))
    return rows


def xyxy(box: tuple[int, float, float, float, float]) -> tuple[float, float, float, float]:
    _, x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(a: tuple[int, float, float, float, float], b: tuple[int, float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def prepare() -> None:
    rng = random.Random(SELECTION_SEED)
    train, test = read_split(TRAIN_SPLIT), read_split(TEST_SPLIT)
    if (len(train), len(test)) != (1604, 688) or set(train) & set(test):
        raise ValueError("Unexpected or leaking GC10 split")
    records = [(path, read_boxes(path)) for path in train]
    donors: dict[int, list[tuple[Path, tuple[int, float, float, float, float]]]] = {i: [] for i in range(10)}
    for path, boxes in records:
        for box in boxes:
            donors[box[0]].append((path, box))
    if any(not rows for rows in donors.values()):
        raise ValueError("At least one class has no training donor")

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    targets = [c for c in range(10) for _ in range(Q // 10)]
    rng.shuffle(targets)
    generated: list[str] = []
    class_counts = {name: 0 for name in CLASSES}
    provenance = []
    for index, target_class in enumerate(targets):
        success = False
        for _ in range(400):
            donor_path, donor_box = rng.choice(donors[target_class])
            bg_path, bg_boxes = rng.choice(records)
            if bg_path == donor_path:
                continue
            donor = cv2.imdecode(np.fromfile(str(donor_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            background = cv2.imdecode(np.fromfile(str(bg_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if donor is None or background is None:
                continue
            dh, dw = donor.shape[:2]
            bh, bw = background.shape[:2]
            _, x, y, w, h = donor_box
            x1, y1 = max(0, int(round((x - w / 2) * dw))), max(0, int(round((y - h / 2) * dh)))
            x2, y2 = min(dw, int(round((x + w / 2) * dw))), min(dh, int(round((y + h / 2) * dh)))
            patch = donor[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            scale = rng.uniform(0.75, 1.0)
            pw, ph = max(3, int(patch.shape[1] * scale)), max(3, int(patch.shape[0] * scale))
            if pw >= 0.8 * bw or ph >= 0.8 * bh:
                continue
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
            for _ in range(100):
                left = rng.randint(0, bw - pw)
                top = rng.randint(0, bh - ph)
                pasted = (target_class, (left + pw / 2) / bw, (top + ph / 2) / bh, pw / bw, ph / bh)
                if any(iou(pasted, existing) > 0.05 for existing in bg_boxes):
                    continue
                canvas = background.copy()
                canvas[top:top + ph, left:left + pw] = patch
                name = f"gc10_cp_{index:04d}.jpg"
                out_image = OUT_IMAGES / name
                out_label = OUT_LABELS / f"gc10_cp_{index:04d}.txt"
                ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                encoded.tofile(str(out_image))
                boxes = [*bg_boxes, pasted]
                out_label.write_text("\n".join(
                    f"{c} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}" for c, xc, yc, ww, hh in boxes
                ) + "\n", encoding="utf-8")
                generated.append(str(out_image))
                class_counts[CLASSES[target_class]] += 1
                provenance.append({"output": name, "class": CLASSES[target_class],
                                   "donor": str(donor_path), "background": str(bg_path)})
                success = True
                break
            if success:
                break
        if not success:
            raise RuntimeError(f"Could not place CopyPaste sample {index}")

    CP_SPLIT.write_text("\n".join([*(str(path) for path in train), *generated]) + "\n", encoding="utf-8")
    names = repr(CLASSES)
    TRAIN_YAML.write_text(f"train: {CP_SPLIT.as_posix()}\nval: {CP_SPLIT.as_posix()}\nnc: 10\nnames: {names}\n", encoding="utf-8")
    EVAL_YAML.write_text(f"train: {CP_SPLIT.as_posix()}\nval: {TEST_SPLIT.as_posix()}\nnc: 10\nnames: {names}\n", encoding="utf-8")
    manifest = {"method": "box-guided rectangular CopyPaste with transferred exact bounding boxes",
                "selection_seed": SELECTION_SEED, "train_real": len(train), "held_out_test": len(test),
                "q": len(generated), "pasted_class_counts": class_counts, "max_iou_with_background_boxes": 0.05,
                "test_used_for_generation": False, "provenance": provenance}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "provenance"}, ensure_ascii=False, indent=2), flush=True)


def complete(run: Path) -> bool:
    path = run / "results.csv"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return len(list(csv.DictReader(handle))) >= EPOCHS


def train_eval(seed: int) -> dict:
    run = RUNS / f"copypaste_q300_s{seed}"
    checkpoint = run / "weights/last.pt"
    if not (checkpoint.exists() and complete(run)):
        if checkpoint.exists():
            code = f"import sys;sys.path.insert(0,r'{SITE}');from ultralytics import YOLO;YOLO(r'{checkpoint}').train(resume=True)"
        else:
            code = (f"import sys;sys.path.insert(0,r'{SITE}');from ultralytics import YOLO;"
                    f"YOLO(r'{ROOT / 'yolov8n.pt'}').train(data=r'{TRAIN_YAML}',imgsz={IMGSZ},batch={BATCH},"
                    f"epochs={EPOCHS},seed={seed},workers=0,device=0,project=r'{RUNS}',name='{run.name}',"
                    "exist_ok=True,plots=False,verbose=False,cache=False,val=False,patience=0,deterministic=True)")
        result = subprocess.run([PYTHON, "-S", "-c", code], cwd=ROOT)
        time.sleep(20)
        if result.returncode or not (checkpoint.exists() and complete(run)):
            raise RuntimeError(f"Incomplete CopyPaste run seed={seed}")
    metrics = YOLO(str(checkpoint)).val(data=str(EVAL_YAML), imgsz=IMGSZ, batch=BATCH, workers=0,
                                        device=0, plots=False, save_json=False, verbose=False)
    return {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "ap50_per_class": [float(x) for x in metrics.box.ap50], "seed": seed,
            "checkpoint": "last.pt", "fixed_epochs": EPOCHS, "detector_batch": BATCH}


def detect() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for seed in SEEDS:
        key = f"copypaste_q300_s{seed}"
        if key not in results or "error" in results[key]:
            try:
                results[key] = train_eval(seed)
            except Exception as error:
                results[key] = {"error": repr(error)}
                RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
                raise
            RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(key, json.dumps(results[key]), flush=True)
    analyze(results)


def analyze(results: dict) -> None:
    phase1 = json.loads(PHASE1.read_text(encoding="utf-8"))
    baseline = np.array([phase1[f"real_only_s{s}"]["mAP50"] for s in SEEDS]) * 100
    method = np.array([results[f"copypaste_q300_s{s}"]["mAP50"] for s in SEEDS]) * 100
    delta = method - baseline
    rng = np.random.default_rng(SELECTION_SEED)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(100000)])
    stats = {"unit": "percentage points", "seeds": SEEDS,
             "real_only_mean_sd": [float(baseline.mean()), float(baseline.std(ddof=1))],
             "copypaste_mean_sd": [float(method.mean()), float(method.std(ddof=1))],
             "paired_delta_mean": float(delta.mean()),
             "paired_delta_descriptive_bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
             "wins_losses_ties": [int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())],
             "wilcoxon_raw_p": float(wilcoxon(delta).pvalue),
             "warning": "Descriptive unadjusted interval; multiplicity correction is deferred until all external methods are complete."}
    STATISTICS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "detect", "all"])
    args = parser.parse_args()
    if args.mode in {"prepare", "all"}:
        prepare()
    if args.mode in {"detect", "all"}:
        if not MANIFEST.exists():
            prepare()
        detect()


if __name__ == "__main__":
    main()
