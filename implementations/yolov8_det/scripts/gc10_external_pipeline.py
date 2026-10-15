# -*- coding: utf-8 -*-
"""GC10-DET leakage-controlled external validation pipeline.

Phase 1 establishes the two indispensable controls before any generated data
are admitted: Real-only and a fixed with-replacement real-image resampling
control with Q=300 extra records. Detector training is serial on one GPU.
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
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, _release_sysconfig.get_paths()['purelib'])
from scipy.stats import wilcoxon
from ultralytics import YOLO


ROOT = Path(str(_release_root))
DROOT = ROOT / "data/gc10_det/yolo"
TRAIN_SPLIT = DROOT / "splits/gc10_train.txt"
TEST_SPLIT = DROOT / "splits/gc10_test.txt"
RUNS = ROOT / "implementations/yolov8_det/runs_gc10_q3_strict_b4"
RESULTS = DROOT / "gc10_phase1_results.json"
STATISTICS = DROOT / "gc10_phase1_statistics.json"
MANIFEST = DROOT / "gc10_phase1_manifest.json"
PYTHON = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']

CLASSES = [
    "crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
    "rolled_pit", "silk_spot", "waist folding", "water_spot", "welding_line",
]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTION_SEED = 20260922
Q = 300
BATCH = 4
EPOCHS = 30
IMGSZ = 320
COOLDOWN_SECONDS = 20


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def yaml_for(name: str, train_split: Path) -> tuple[Path, Path]:
    train_path = DROOT / f"gc10_{name}_train.yaml"
    eval_path = DROOT / f"gc10_{name}_eval.yaml"
    # Ultralytics may scan the val entry even when val=False. Point the training
    # YAML at training data only so held-out labels are not read until model.val.
    train_path.write_text(
        f"train: {train_split.as_posix()}\n"
        f"val: {train_split.as_posix()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES!r}\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        f"train: {train_split.as_posix()}\n"
        f"val: {TEST_SPLIT.as_posix()}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES!r}\n",
        encoding="utf-8",
    )
    return train_path, eval_path


def run_complete(run: Path) -> bool:
    """Require all fixed epochs in results.csv before accepting last.pt."""
    results_csv = run / "results.csv"
    if not results_csv.exists():
        return False
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) >= EPOCHS


def prepare() -> None:
    train = read_lines(TRAIN_SPLIT)
    test = read_lines(TEST_SPLIT)
    if (len(train), len(test)) != (1604, 688):
        raise ValueError(f"Unexpected split sizes: train={len(train)}, test={len(test)}")
    if set(train) & set(test):
        raise ValueError("GC10 train/test leakage detected")

    extras = random.Random(SELECTION_SEED).choices(train, k=Q)
    resampled = train + extras
    resampled_split = DROOT / "splits/gc10_real_resampled_q300.txt"
    resampled_split.write_text("\n".join(resampled) + "\n", encoding="utf-8")
    counts = {path: extras.count(path) for path in set(extras)}

    yaml_for("real_only", TRAIN_SPLIT)
    yaml_for("real_resampled_q300", resampled_split)
    manifest = {
        "dataset": "GC10-DET Dataset Ninja export after structural audit",
        "train_images": len(train),
        "held_out_test_images": len(test),
        "q": Q,
        "selection_seed": SELECTION_SEED,
        "resampling": "one fixed with-replacement draw from training images",
        "resampled_records_per_epoch": len(resampled),
        "extra_unique_images": len(counts),
        "max_extra_repetitions_per_image": max(counts.values()),
        "detector": {
            "architecture": "YOLOv8n",
            "imgsz": IMGSZ,
            "batch": BATCH,
            "epochs": EPOCHS,
            "train_time_val": False,
            "patience": 0,
            "checkpoint": "last.pt",
            "seeds": SEEDS,
        },
        "interpretation": (
            "The resampling control matches training-record and iteration budgets but also "
            "changes empirical image weights; it is not a pure optimizer-step intervention."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def train_eval(group: str, seed: int) -> dict:
    train_yaml = DROOT / f"gc10_{group}_train.yaml"
    eval_yaml = DROOT / f"gc10_{group}_eval.yaml"
    run_name = f"{group}_s{seed}"
    run = RUNS / run_name
    checkpoint = run / "weights/last.pt"
    if not (checkpoint.exists() and run_complete(run)):
        if checkpoint.exists():
            code = (
                f"import sys; sys.path.insert(0, r'{SITE}'); from ultralytics import YOLO; "
                f"YOLO(r'{checkpoint}').train(resume=True)"
            )
        else:
            code = (
                f"import sys; sys.path.insert(0, r'{SITE}'); from ultralytics import YOLO; "
                f"YOLO(r'{ROOT / 'yolov8n.pt'}').train(data=r'{train_yaml}', imgsz={IMGSZ}, "
                f"batch={BATCH}, epochs={EPOCHS}, seed={seed}, workers=0, device=0, "
                f"project=r'{RUNS}', name='{run_name}', exist_ok=True, plots=False, "
                "verbose=False, cache=False, val=False, patience=0, deterministic=True)"
            )
        completed = subprocess.run([PYTHON, "-S", "-c", code], cwd=ROOT)
        time.sleep(COOLDOWN_SECONDS)
        if completed.returncode:
            raise RuntimeError(f"Training failed: {group} seed={seed}")
        if not (checkpoint.exists() and run_complete(run)):
            raise RuntimeError(f"Incomplete fixed-epoch run: {group} seed={seed}")
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(eval_yaml), imgsz=IMGSZ, batch=BATCH, workers=0, device=0,
        plots=False, save_json=False, verbose=False,
    )
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "ap50_per_class": [float(value) for value in metrics.box.ap50],
        "seed": seed,
        "checkpoint": "last.pt",
        "train_val": False,
        "fixed_epochs": EPOCHS,
        "detector_batch": BATCH,
    }


def detect() -> None:
    if not MANIFEST.exists():
        prepare()
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for seed in SEEDS:
        for group in ("real_only", "real_resampled_q300"):
            key = f"{group}_s{seed}"
            if key not in results or "error" in results[key]:
                try:
                    results[key] = train_eval(group, seed)
                except Exception as error:
                    results[key] = {"error": repr(error)}
                    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    raise
                RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(key, json.dumps(results[key]), flush=True)
    analyze(results)


def bootstrap_mean_ci(values: np.ndarray, repetitions: int = 100000) -> list[float]:
    rng = np.random.default_rng(20260922)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def analyze(results: dict | None = None) -> None:
    if results is None:
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
    real = np.array([results[f"real_only_s{seed}"]["mAP50"] for seed in SEEDS]) * 100.0
    control = np.array([results[f"real_resampled_q300_s{seed}"]["mAP50"] for seed in SEEDS]) * 100.0
    difference = control - real
    test = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    output = {
        "unit": "percentage points",
        "seeds": SEEDS,
        "real_only_mean_sd": [float(real.mean()), float(real.std(ddof=1))],
        "resampling_mean_sd": [float(control.mean()), float(control.std(ddof=1))],
        "paired_delta_mean": float(difference.mean()),
        "paired_delta_descriptive_bootstrap_95ci": bootstrap_mean_ci(difference),
        "wins_losses_ties": [int((difference > 0).sum()), int((difference < 0).sum()),
                              int((difference == 0).sum())],
        "wilcoxon_raw_p": float(test.pvalue),
        "cohens_dz": float(difference.mean() / difference.std(ddof=1))
        if not math.isclose(float(difference.std(ddof=1)), 0.0) else None,
        "warning": (
            "The interval is descriptive and unadjusted. The resampling control changes "
            "both optimization iterations and empirical image weights."
        ),
    }
    STATISTICS.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "detect", "analyze", "all"])
    args = parser.parse_args()
    if args.mode in {"prepare", "all"}:
        prepare()
    if args.mode in {"detect", "all"}:
        detect()
    if args.mode == "analyze":
        analyze()


if __name__ == "__main__":
    main()
