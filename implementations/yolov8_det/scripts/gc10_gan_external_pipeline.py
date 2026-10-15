"""GC10 external validation of the fixed text projection + MSES generator.

Generated condition folders are provenance only. A frozen real-only detector
provides the training class and box pseudo-labels. The held-out split is used
only by the final detector evaluation after each fixed 30-epoch training run.
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
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, _release_sysconfig.get_paths()['purelib'])
from scipy.stats import wilcoxon
from ultralytics import YOLO

ROOT = Path(str(_release_root))
DROOT = ROOT / "data/gc10_det/yolo"
IMAGES = DROOT / "images/gen_textproj_mses_s42_5k"
LABELS = DROOT / "labels/gen_textproj_mses_s42_5k"
TEACHER = ROOT / "implementations/yolov8_det/runs_gc10_q3_strict_b4/real_only_s42/weights/last.pt"
TRAIN_SPLIT = DROOT / "splits/gc10_train.txt"
TEST_SPLIT = DROOT / "splits/gc10_test.txt"
Q_SPLIT = DROOT / "splits/gc10_textproj_mses_s42_q300.txt"
TRAIN_YAML = DROOT / "gc10_textproj_mses_s42_q300_train.yaml"
EVAL_YAML = DROOT / "gc10_textproj_mses_s42_q300_eval.yaml"
PREDICTIONS = DROOT / "gc10_textproj_mses_s42_predictions.json"
MANIFEST = DROOT / "gc10_textproj_mses_s42_q300_manifest.json"
RESULTS = DROOT / "gc10_textproj_mses_s42_q300_results.json"
STATISTICS = DROOT / "gc10_textproj_mses_s42_q300_statistics.json"
PHASE1 = DROOT / "gc10_phase1_results.json"
COPYPASTE = DROOT / "gc10_copypaste_results.json"
RUNS = ROOT / "implementations/yolov8_det/runs_gc10_gan_strict_b4"
PYTHON = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']
CLASSES = ["crease", "crescent_gap", "inclusion", "oil_spot", "punching_hole",
           "rolled_pit", "silk_spot", "waist folding", "water_spot", "welding_line"]
FOLDERS = [name.replace(" ", "_") for name in CLASSES]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SELECTION_SEED = 20260922
Q = 300
CONF = 0.10
IMGSZ = 320
BATCH = 4
EPOCHS = 30


def read_split(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidates() -> list[Path]:
    by_class = {folder: sorted((IMAGES / folder).glob("*.png")) for folder in FOLDERS}
    counts = {folder: len(paths) for folder, paths in by_class.items()}
    if any(count != 500 for count in counts.values()):
        raise ValueError(f"Incomplete generated candidate set: {counts}")
    paths = [path for folder in FOLDERS for path in by_class[folder]]
    if len(set(paths)) != 5000:
        raise ValueError("Duplicate generated path")
    return paths


def validate_split() -> tuple[list[Path], list[Path]]:
    train, test = read_split(TRAIN_SPLIT), read_split(TEST_SPLIT)
    if (len(train), len(test)) != (1604, 688) or set(train) & set(test):
        raise ValueError("GC10 train/test split mismatch or overlap")
    return train, test


def predict() -> None:
    validate_split()
    if not TEACHER.is_file():
        raise FileNotFoundError(TEACHER)
    paths = candidates()
    model = YOLO(str(TEACHER))
    predictions: dict[str, list[list[float]]] = {}
    for start in range(0, len(paths), 8):
        chunk = paths[start:start + 8]
        for result in model.predict(source=[str(path) for path in chunk], conf=0.05,
                                    imgsz=IMGSZ, batch=8, device=0, stream=True,
                                    verbose=False, save=False):
            rows = []
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf.item())
                    cls = int(box.cls.item())
                    x, y, w, h = [float(value) for value in box.xywhn[0].tolist()]
                    if not (0 <= cls < 10 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError(f"Invalid teacher prediction for {result.path}")
                    rows.append([confidence, cls, x, y, w, h])
            key = Path(result.path).relative_to(IMAGES).as_posix()
            predictions[key] = rows
        if (start + len(chunk)) % 500 == 0:
            print(f"[PREDICT] {start + len(chunk)}/5000", flush=True)
    if len(predictions) != 5000:
        raise ValueError(f"Teacher predictions={len(predictions)}, expected=5000")
    PREDICTIONS.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    accepted = sum(any(row[0] >= CONF for row in rows) for rows in predictions.values())
    print(f"[PREDICT] accepted at conf={CONF}: {accepted}/5000", flush=True)


def prepare() -> None:
    train, test = validate_split()
    paths = candidates()
    expected = {path.relative_to(IMAGES).as_posix() for path in paths}
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if set(predictions) != expected:
        raise ValueError("Teacher predictions do not cover the exact candidate set")
    LABELS.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []
    predicted_classes = Counter()
    condition_classes = Counter()
    for rel in sorted(predictions):
        kept = [row for row in predictions[rel] if row[0] >= CONF]
        if not kept:
            continue
        label = LABELS / Path(rel).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(
            f"{int(row[1])} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f}" for row in kept
        ) + "\n", encoding="utf-8")
        accepted.append(IMAGES / rel)
        condition_classes[Path(rel).parts[0]] += 1
        predicted_classes.update(int(row[1]) for row in kept)
    if len(accepted) < Q:
        raise ValueError(f"Only {len(accepted)} teacher-accepted candidates; Q={Q}")
    selected = sorted(random.Random(SELECTION_SEED).sample(accepted, Q))
    if set(selected) & set(train) or set(selected) & set(test):
        raise ValueError("Synthetic image overlaps with real train or held-out test")
    Q_SPLIT.write_text("\n".join(str(path) for path in [*train, *selected]) + "\n", encoding="utf-8")
    names = repr(CLASSES)
    TRAIN_YAML.write_text(f"train: {Q_SPLIT.as_posix()}\nval: {Q_SPLIT.as_posix()}\nnc: 10\nnames: {names}\n", encoding="utf-8")
    EVAL_YAML.write_text(f"train: {Q_SPLIT.as_posix()}\nval: {TEST_SPLIT.as_posix()}\nnc: 10\nnames: {names}\n", encoding="utf-8")
    selected_predictions = Counter()
    for image in selected:
        rel = image.relative_to(IMAGES).as_posix()
        selected_predictions.update(int(row[1]) for row in predictions[rel] if row[0] >= CONF)
    manifest = {
        "generator_checkpoint": str(ROOT / "implementations/wgan_gp/checkpoint_gc10_textproj_mses_s42_final.pth"),
        "teacher_checkpoint": str(TEACHER), "teacher_role": "class and box pseudo-labels only",
        "generator_condition_used_as_label": False, "confidence_threshold": CONF,
        "selection_seed": SELECTION_SEED, "candidate_pool": len(paths),
        "teacher_accepted_pool": len(accepted), "q": Q,
        "real_train": len(train), "held_out_test": len(test),
        "accepted_by_generator_condition": dict(condition_classes),
        "accepted_predicted_box_classes": {CLASSES[c]: predicted_classes[c] for c in range(10)},
        "selected_predicted_box_classes": {CLASSES[c]: selected_predictions[c] for c in range(10)},
        "selected_paths": [str(path) for path in selected],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_paths"}, ensure_ascii=False, indent=2), flush=True)


def complete(run: Path) -> bool:
    path = run / "results.csv"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return len(list(csv.DictReader(handle))) >= EPOCHS


def train_eval(seed: int) -> dict:
    run = RUNS / f"textproj_mses_q300_s{seed}"
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
            raise RuntimeError(f"Incomplete GAN-augmented detector seed={seed}")
    metrics = YOLO(str(checkpoint)).val(data=str(EVAL_YAML), imgsz=IMGSZ, batch=BATCH,
                                        workers=0, device=0, plots=False, save_json=False, verbose=False)
    return {"mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "ap50_per_class": [float(value) for value in metrics.box.ap50],
            "seed": seed, "checkpoint": "last.pt", "fixed_epochs": EPOCHS, "detector_batch": BATCH}


def detect() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError("Run prepare and audit manifest before detector training")
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for seed in SEEDS:
        key = f"textproj_mses_q300_s{seed}"
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
    baseline = json.loads(PHASE1.read_text(encoding="utf-8"))
    copypaste = json.loads(COPYPASTE.read_text(encoding="utf-8"))
    method = np.array([results[f"textproj_mses_q300_s{s}"]["mAP50"] for s in SEEDS]) * 100
    comparators = {
        "real_only": np.array([baseline[f"real_only_s{s}"]["mAP50"] for s in SEEDS]) * 100,
        "real_resampled_q300": np.array([baseline[f"real_resampled_q300_s{s}"]["mAP50"] for s in SEEDS]) * 100,
        "copypaste_q300": np.array([copypaste[f"copypaste_q300_s{s}"]["mAP50"] for s in SEEDS]) * 100,
    }
    rng = np.random.default_rng(SELECTION_SEED)
    comparisons = {}
    for name, control in comparators.items():
        delta = method - control
        boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(100000)])
        comparisons[name] = {
            "control_mean_sd": [float(control.mean()), float(control.std(ddof=1))],
            "paired_delta_mean": float(delta.mean()),
            "paired_delta_descriptive_bootstrap_95ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "wins_losses_ties": [int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())],
            "wilcoxon_raw_p": float(wilcoxon(delta).pvalue),
        }
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["wilcoxon_raw_p"])
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * comparisons[name]["wilcoxon_raw_p"]))
        comparisons[name]["holm_p_within_three_method_comparisons"] = running
    stats = {"unit": "percentage points", "seeds": SEEDS,
             "method_mean_sd": [float(method.mean()), float(method.std(ddof=1))],
             "comparisons": comparisons,
             "warning": "Intervals are descriptive and unadjusted. Holm p-values adjust the three external comparisons within this family."}
    STATISTICS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["predict", "prepare", "detect", "all"])
    args = parser.parse_args()
    if args.mode in {"predict", "all"}:
        if PREDICTIONS.exists():
            print(f"[SKIP] {PREDICTIONS}", flush=True)
        else:
            predict()
    if args.mode in {"prepare", "all"}:
        prepare()
    if args.mode in {"detect", "all"}:
        detect()


if __name__ == "__main__":
    main()
