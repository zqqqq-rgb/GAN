# -*- coding: utf-8 -*-
"""Nested Q sensitivity for the current text-projection + MSES seed-42 generator."""
import json
import random
from pathlib import Path

import leakfix_det_pipeline as lf

QS = [100, 200, 300, 500]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GROUP = "E7proj_mses_sp_full"
SELECT_SEED = 20260912
RUNS = lf.PROJECT_ROOT / "implementations/yolov8_det/runs_q3_q_sensitivity"
RESULTS = lf.YOLO_ROOT / "q3_q_sensitivity_current.json"
MANIFEST = lf.YOLO_ROOT / "q3_q_sensitivity_manifest.json"


def prepare():
    real = (lf.SPLIT_DIR / "train_baseline.txt").read_text(encoding="utf-8").splitlines()
    source = (lf.SPLIT_DIR / f"train_{GROUP}.txt").read_text(encoding="utf-8").splitlines()
    if len(real) != 193:
        raise ValueError(f"Expected 193 real images, found {len(real)}")
    pool = sorted(set(source) - set(real))
    if len(pool) < max(QS):
        raise ValueError(f"Accepted pool {len(pool)} < Q={max(QS)}")
    permutation = pool[:]
    random.Random(SELECT_SEED).shuffle(permutation)
    payload = {"selection_seed": SELECT_SEED, "nested_prefix_design": True,
               "accepted_pool": len(pool), "real_count": len(real), "q": {}}
    for q in QS:
        selected = sorted(permutation[:q])
        split = lf.SPLIT_DIR / f"train_qsens_q{q}_{GROUP}.txt"
        split.write_text("\n".join(real + selected) + "\n", encoding="utf-8")
        yaml = lf.YOLO_ROOT / f"data_qsens_q{q}_{GROUP}.yaml"
        yaml.write_text(
            f"train: {split.as_posix()}\nval: {(lf.SPLIT_DIR / 'val.txt').as_posix()}\n"
            "nc: 1\nnames: ['defect']\n", encoding="utf-8")
        payload["q"][str(q)] = {"split": str(split), "selected": q}
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(payload)


def detect(force_train=False, strict_resume=False):
    old = lf.RUNS_DIR
    lf.RUNS_DIR = RUNS
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    try:
        for q in QS:
            yaml = lf.YOLO_ROOT / f"data_qsens_q{q}_{GROUP}.yaml"
            for seed in SEEDS:
                key = f"q{q}_s{seed}"
                run = RUNS / key
                checkpoint = run / "weights/last.pt"
                prior = results.get(key, {})
                strict_done = (prior.get("checkpoint") == "last.pt" and
                               prior.get("train_val") is False and
                               prior.get("fixed_epochs") == lf.EPOCHS and
                               prior.get("detector_batch") == lf.BATCH)
                if force_train or not checkpoint.exists() or (strict_resume and not strict_done):
                    if not lf.train_yolo(str(yaml), run, seed=seed):
                        results[key] = {"error": "train failed"}
                        RESULTS.write_text(json.dumps(results, indent=2))
                        raise RuntimeError(f"Training failed for {key}; stopping to protect the GPU and preserve protocol consistency")
                metrics = lf.eval_yolo(str(checkpoint), str(yaml))
                if metrics:
                    metrics.update({"q": q, "seed": seed, "nested_prefix_design": True,
                                    "checkpoint": "last.pt", "train_val": False,
                                    "fixed_epochs": lf.EPOCHS,
                                    "detector_batch": lf.BATCH})
                    results[key] = metrics
                    RESULTS.write_text(json.dumps(results, indent=2))
                    print(key, metrics)
    finally:
        lf.RUNS_DIR = old


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--strict-resume", action="store_true")
    args = parser.parse_args()
    prepare()
    detect(force_train=args.force_train, strict_resume=args.strict_resume)
