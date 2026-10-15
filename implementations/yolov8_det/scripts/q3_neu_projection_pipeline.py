# -*- coding: utf-8 -*-
"""Leakage-free NEU-DET validation for the current text-projection + MSES model."""

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
GEN = NROOT / "images/gen_E7proj_mses_sp_neu_s42"
LABELS = NROOT / "labels/gen_E7proj_mses_sp_neu_s42"
REAL_SPLIT = NROOT / "splits/neu_train.txt"
VAL_SPLIT = NROOT / "splits/neu_val.txt"
POOL_SPLIT = NROOT / "splits/neu_q3_textproj_pool.txt"
FIXED_SPLIT = NROOT / "splits/neu_q3_textproj_q300.txt"
STEPMATCHED_SPLIT = NROOT / "splits/neu_q3_baseline_stepmatched_q300.txt"
PRED_JSON = NROOT / "q3_textproj_predictions.json"
MANIFEST = NROOT / "q3_textproj_manifest.json"
RESULTS = NROOT / "q3_textproj_results.json"
BASE_MODEL = ROOT / "implementations/yolov8_det/runs/yolov8n_neu_baseline/weights/last.pt"
RUNS = ROOT / "implementations/yolov8_det/runs_neu_q3_strict_b4"
PYTHON = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
Q = 300
CONF = 0.10
SELECTION_SEED = 20260912
DETECTOR_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BATCH = 4
GPU_COOLDOWN_SECONDS = 20


def images() -> list[Path]:
    return sorted(GEN.rglob("*.png"))


def predict() -> None:
    paths = images()
    if len(paths) != 3000:
        raise ValueError(f"Expected 3000 generated images, found {len(paths)}")
    model = YOLO(str(BASE_MODEL))
    predictions = {}
    # A 3000-path list makes Ultralytics warm up with a 3000-image tensor.
    # Chunking keeps inference within the 8 GB GPU budget without changing predictions.
    for start in range(0, len(paths), 16):
        chunk = paths[start:start + 16]
        for result in model.predict(source=[str(x) for x in chunk], conf=0.05, imgsz=320,
                                    batch=16, stream=True, verbose=False, save=False):
            rows = []
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    cls = int(box.cls.item())
                    x, y, w, h = [float(v) for v in box.xywhn[0].tolist()]
                    rows.append([confidence, cls, x, y, w, h])
            rel = Path(result.path).relative_to(GEN).as_posix()
            predictions[rel] = rows
    PRED_JSON.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    accepted = sum(any(row[0] >= CONF for row in rows) for rows in predictions.values())
    print(f"[PSEUDO] accepted={accepted}/{len(predictions)} at conf={CONF}")


def prepare() -> None:
    predictions = json.loads(PRED_JSON.read_text(encoding="utf-8"))
    LABELS.mkdir(parents=True, exist_ok=True)
    real = [x for x in REAL_SPLIT.read_text(encoding="utf-8").splitlines() if x]
    val = [x for x in VAL_SPLIT.read_text(encoding="utf-8").splitlines() if x]
    if len(real) != 1439 or len(val) != 360:
        raise ValueError(f"Unexpected NEU split: train={len(real)}, val={len(val)}")
    if set(real) & set(val):
        raise ValueError("Train/validation leakage detected")
    accepted = []
    class_counts = {name: 0 for name in CLASSES}
    for rel, rows in predictions.items():
        kept = [row for row in rows if row[0] >= CONF]
        if not kept:
            continue
        image_path = GEN / rel
        label_path = LABELS / Path(rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(
            f"{int(row[1])} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f}" for row in kept
        ) + "\n", encoding="utf-8")
        accepted.append(str(image_path))
        for row in kept:
            class_counts[CLASSES[int(row[1])]] += 1
    if len(accepted) < Q:
        raise ValueError(f"Only {len(accepted)} accepted images; Q={Q}")
    accepted = sorted(accepted)
    POOL_SPLIT.write_text("\n".join(real + accepted) + "\n", encoding="utf-8")
    selected = sorted(random.Random(SELECTION_SEED).sample(accepted, Q))
    FIXED_SPLIT.write_text("\n".join(real + selected) + "\n", encoding="utf-8")
    repeated_real = random.Random(SELECTION_SEED).choices(real, k=Q)
    STEPMATCHED_SPLIT.write_text(
        "\n".join(real + repeated_real) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol": "teacher-predicted class and xywhn; generated image condition is not used as a label",
        "train_real": len(real), "held_out_val": len(val), "accepted_pool": len(accepted),
        "q": Q, "confidence": CONF, "selection_seed": SELECTION_SEED,
        "step_matched_control": {
            "real_unique": len(real),
            "repeated_real_entries": Q,
            "total_entries_per_epoch": len(real) + Q,
            "purpose": "match sample count and optimizer steps of Q=300 augmentation groups",
            "split": str(STEPMATCHED_SPLIT),
        },
        "teacher_box_class_counts": class_counts,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, split in [
        ("baseline", REAL_SPLIT),
        ("baseline_stepmatched", STEPMATCHED_SPLIT),
        ("textproj_q300", FIXED_SPLIT),
    ]:
        yaml = NROOT / f"neu_q3_{name}.yaml"
        yaml.write_text(
            f"train: {split.as_posix()}\nval: {VAL_SPLIT.as_posix()}\nnc: 6\n"
            f"names: {CLASSES!r}\n", encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False))


def train_eval(group: str, seed: int) -> dict:
    yaml = NROOT / f"neu_q3_{group}.yaml"
    run = RUNS / f"{group}_s{seed}"
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


def detect(recompute: bool = False, groups: list[str] | None = None) -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    groups = groups or ["baseline", "baseline_stepmatched", "textproj_q300"]
    for seed in DETECTOR_SEEDS:
        for group in groups:
            key = f"{group}_s{seed}"
            if recompute or key not in results or "error" in results[key]:
                try:
                    results[key] = train_eval(group, seed)
                except Exception as error:
                    results[key] = {"error": repr(error)}
                    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    raise
                RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(key, results[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["predict", "prepare", "detect", "all"])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--groups", nargs="+",
        choices=["baseline", "baseline_stepmatched", "textproj_q300"],
    )
    args = parser.parse_args()
    if args.mode in {"predict", "all"}: predict()
    if args.mode in {"prepare", "all"}: prepare()
    if args.mode in {"detect", "all"}: detect(recompute=args.recompute, groups=args.groups)


if __name__ == "__main__":
    main()
