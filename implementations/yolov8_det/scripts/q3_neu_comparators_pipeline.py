# -*- coding: utf-8 -*-
"""Complete fixed-Q NEU-DET comparators under the current leak-free protocol.

The held-out 360-image split is never used for GAN training, pseudo-labeling,
sample selection, or detector training. GAN conditions are not treated as
labels: a detector trained only on the 1439 real training images supplies both
class and box pseudo-labels. CopyPaste keeps its exact labels and is sampled to
the same Q=300 budget.
"""

from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, _release_sysconfig.get_paths()['purelib'])
from ultralytics import YOLO

ROOT = Path(str(_release_root))
NROOT = ROOT / "data/neu_det/yolo"
REAL_SPLIT = NROOT / "splits/neu_train.txt"
VAL_SPLIT = NROOT / "splits/neu_val.txt"
BASE_MODEL = ROOT / "implementations/yolov8_det/runs/yolov8n_neu_baseline/weights/last.pt"
RUNS = ROOT / "implementations/yolov8_det/runs_neu_q3_comparators_strict_b4"
RESULTS = NROOT / "q3_neu_comparator_results.json"
PYTHON = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
DETECTOR_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Q = 300
CONF = 0.10
SELECTION_SEED = 20260912
BATCH = 4
GPU_COOLDOWN_SECONDS = 20

GROUPS = {
    "text_nom": {
        "images": NROOT / "images/gen_E7proj_mses_nosp_neu_s42",
        "labels": NROOT / "labels/gen_E7proj_mses_nosp_neu_s42",
    },
    "class_nom": {
        "images": NROOT / "images/gen_E6proj_mses_nosp_neu_s42",
        "labels": NROOT / "labels/gen_E6proj_mses_nosp_neu_s42",
    },
}


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_real_split() -> tuple[list[str], list[str]]:
    real, val = read_lines(REAL_SPLIT), read_lines(VAL_SPLIT)
    if (len(real), len(val)) != (1439, 360):
        raise ValueError(f"Unexpected NEU split: train={len(real)}, val={len(val)}")
    if set(real) & set(val):
        raise ValueError("Train/validation leakage detected")
    return real, val


def generated_images(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*.png"))
    if len(paths) != 3000:
        raise ValueError(f"Expected 3000 generated images in {root}, found {len(paths)}")
    return paths


def predict_group(name: str) -> None:
    cfg = GROUPS[name]
    paths = generated_images(cfg["images"])
    output = NROOT / f"q3_{name}_predictions.json"
    model = YOLO(str(BASE_MODEL))
    predictions: dict[str, list[list[float]]] = {}
    for start in range(0, len(paths), 16):
        chunk = paths[start:start + 16]
        for result in model.predict(source=[str(x) for x in chunk], conf=0.05,
                                    imgsz=320, batch=16, stream=True,
                                    verbose=False, save=False):
            rows = []
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    cls = int(box.cls.item())
                    x, y, w, h = [float(v) for v in box.xywhn[0].tolist()]
                    rows.append([confidence, cls, x, y, w, h])
            rel = Path(result.path).relative_to(cfg["images"]).as_posix()
            predictions[rel] = rows
    output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    accepted = sum(any(row[0] >= CONF for row in rows) for rows in predictions.values())
    print(f"[PSEUDO] {name}: accepted={accepted}/{len(predictions)} at conf={CONF}", flush=True)


def prepare_gan_group(name: str, real: list[str]) -> dict:
    cfg = GROUPS[name]
    pred_path = NROOT / f"q3_{name}_predictions.json"
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    cfg["labels"].mkdir(parents=True, exist_ok=True)
    accepted = []
    class_counts = {class_name: 0 for class_name in CLASSES}
    for rel, rows in predictions.items():
        kept = [row for row in rows if row[0] >= CONF]
        if not kept:
            continue
        image_path = cfg["images"] / rel
        label_path = cfg["labels"] / Path(rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(
            f"{int(row[1])} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f}"
            for row in kept
        ) + "\n", encoding="utf-8")
        accepted.append(str(image_path))
        for row in kept:
            class_counts[CLASSES[int(row[1])]] += 1
    if len(accepted) < Q:
        raise ValueError(f"{name}: only {len(accepted)} accepted images; Q={Q}")
    selected = sorted(random.Random(SELECTION_SEED).sample(sorted(accepted), Q))
    split = NROOT / f"splits/neu_q3_{name}_q300.txt"
    split.write_text("\n".join(real + selected) + "\n", encoding="utf-8")
    return {"source": "teacher-predicted class and xywhn", "accepted_pool": len(accepted),
            "q": Q, "class_counts": class_counts, "split": str(split)}


def prepare_copypaste(real: list[str]) -> dict:
    image_root = NROOT / "images/gen_copypaste"
    label_root = NROOT / "labels/gen_copypaste"
    candidates = sorted(path for path in image_root.iterdir()
                        if path.is_file() and (label_root / f"{path.stem}.txt").exists())
    if len(candidates) < Q:
        raise ValueError(f"CopyPaste candidates={len(candidates)} < Q={Q}")
    selected = sorted(random.Random(SELECTION_SEED).sample(candidates, Q))
    split = NROOT / "splits/neu_q3_copypaste_q300.txt"
    split.write_text("\n".join(real + [str(path) for path in selected]) + "\n", encoding="utf-8")
    return {"source": "exact CopyPaste labels", "accepted_pool": len(candidates),
            "q": Q, "split": str(split)}


def prepare() -> None:
    real, _ = validate_real_split()
    manifest = {name: prepare_gan_group(name, real) for name in GROUPS}
    manifest["copypaste"] = prepare_copypaste(real)
    manifest["protocol"] = {"train_real": 1439, "held_out_val": 360,
                            "q": Q, "confidence": CONF,
                            "selection_seed": SELECTION_SEED,
                            "condition_as_label": False}
    (NROOT / "q3_neu_comparator_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in [*GROUPS, "copypaste"]:
        split = NROOT / f"splits/neu_q3_{name}_q300.txt"
        yaml = NROOT / f"neu_q3_{name}_q300.yaml"
        yaml.write_text(
            f"train: {split.as_posix()}\nval: {VAL_SPLIT.as_posix()}\nnc: 6\n"
            f"names: {CLASSES!r}\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def train_eval(group: str, seed: int) -> dict:
    yaml = NROOT / f"neu_q3_{group}_q300.yaml"
    run = RUNS / f"{group}_q300_s{seed}"
    checkpoint = run / "weights/last.pt"
    if not checkpoint.exists():
        code = (
            f"import sys; sys.path.insert(0, r'{SITE}'); from ultralytics import YOLO; "
            f"YOLO(r'{ROOT / 'yolov8n.pt'}').train(data=r'{yaml}', imgsz=320, batch={BATCH}, "
            f"epochs=30, seed={seed}, workers=0, project=r'{RUNS}', name='{run.name}', "
            "exist_ok=True, plots=False, verbose=False, cache=False, val=False, patience=0)"
        )
        result = subprocess.run([PYTHON, "-S", "-c", code], cwd=ROOT)
        time.sleep(GPU_COOLDOWN_SECONDS)
        if result.returncode:
            raise RuntimeError(f"Training failed: {group} seed={seed}")
    model = YOLO(str(checkpoint))
    metrics = model.val(data=str(yaml), imgsz=320, batch=BATCH, workers=0,
                        plots=False, save_json=False, verbose=False)
    return {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "ap50_per_class": [float(x) for x in metrics.box.ap50], "seed": seed,
            "checkpoint": "last.pt", "train_val": False, "fixed_epochs": 30,
            "detector_batch": BATCH}


def detect(recompute: bool = False) -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for seed in DETECTOR_SEEDS:
        for group in [*GROUPS, "copypaste"]:
            key = f"{group}_q300_s{seed}"
            if recompute or key not in results or "error" in results[key]:
                try:
                    results[key] = train_eval(group, seed)
                except Exception as error:
                    results[key] = {"error": repr(error)}
                    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    raise
                RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(key, results[key], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["predict", "prepare", "detect", "all"])
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.mode in {"predict", "all"}:
        for name in GROUPS:
            pred_path = NROOT / f"q3_{name}_predictions.json"
            if pred_path.exists():
                print(f"[SKIP] predictions {name}", flush=True)
            else:
                predict_group(name)
    if args.mode in {"prepare", "all"}:
        prepare()
    if args.mode in {"detect", "all"}:
        detect(recompute=args.recompute)


if __name__ == "__main__":
    main()
