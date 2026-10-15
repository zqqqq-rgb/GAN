# -*- coding: utf-8 -*-
"""Build and evaluate fixed-quantity augmentation splits for the Q3 study.

Every augmented group contains the same 193 real training images and exactly
Q accepted synthetic images.  Selection is deterministic.  The held-out list
is never read when constructing training splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import leakfix_det_pipeline as lf


Q_DEFAULT = 300
SELECTION_SEED = 20260910
SPLIT_DIR = lf.SPLIT_DIR
RUNS_DIR = lf.PROJECT_ROOT / "implementations" / "yolov8_det" / "runs_q3_fixedq"
RESULTS_JSON = lf.YOLO_ROOT / "q3_fixedq_results.json"
MANIFEST_JSON = lf.YOLO_ROOT / "q3_fixedq_manifest.json"

GROUPS = [
    "baseline_stepmatched",
    "E5_tradit",
    "E4_dcgan",
    "E2_wgan_gp",
    "E2DA",
    "FastGAN",
    "copypaste",
    "E7proj_mses_nosp",
    "E6proj_mses_sp",
    "E6proj_mses_nosp",
    "E7proj_mses_sp",
    "E7proj_mses_nosp_s1",
    "E7proj_mses_nosp_s2",
    "E6proj_mses_nosp_s1",
    "E6proj_mses_nosp_s2",
    "E7proj_mses_sp_s1",
    "E7proj_mses_sp_s2",
    "Diffusion_LDM",
    "Diffusion_LDM_gs1",
    "Diffusion_LDM_gs2",
]


def condition_prefix(path_text: str) -> str:
    stem = Path(path_text).stem
    prefix = stem.split("_", 1)[0].lower()
    return prefix if prefix in {"dent", "scratch", "deformation", "bulge"} else "unavailable"


def prepare(q: int, groups: list[str]) -> None:
    real_lines = (SPLIT_DIR / "train_baseline.txt").read_text(encoding="utf-8").splitlines()
    if len(real_lines) != 193:
        raise ValueError(f"Expected 193 real training images, found {len(real_lines)}")

    if MANIFEST_JSON.exists():
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        if manifest.get("q") != q or manifest.get("selection_seed") != SELECTION_SEED:
            raise ValueError("Existing manifest uses a different fixed-Q protocol")
        if manifest.get("real_count") != len(real_lines):
            raise ValueError("Existing manifest uses a different real training split")
        manifest.setdefault("groups", {})
    else:
        manifest = {
            "q": q,
            "selection_seed": SELECTION_SEED,
            "real_count": len(real_lines),
            "groups": {},
        }
    for index, group in enumerate(groups):
        if group == "baseline_stepmatched":
            rng = random.Random(SELECTION_SEED)
            repeated = rng.choices(real_lines, k=q)
            output = SPLIT_DIR / f"train_q{q}_{group}.txt"
            output.write_text("\n".join(real_lines + repeated) + "\n", encoding="utf-8")
            manifest["groups"][group] = {
                "source_split": str(SPLIT_DIR / "train_baseline.txt"),
                "output_split": str(output),
                "real_unique": len(real_lines),
                "repeated_real_entries": q,
                "selected": q,
                "purpose": "match the per-epoch sample count and optimizer steps of Q=300 augmentation groups",
            }
            print(f"[FIXED-Q] {group}: {len(real_lines)} unique real + {q} repeated real entries")
            continue
        source = SPLIT_DIR / f"train_{group}.txt"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run leakfix_det_pipeline.py pseudo for this group first."
            )
        all_lines = source.read_text(encoding="utf-8").splitlines()
        real_set = set(real_lines)
        synthetic = [line for line in all_lines if line not in real_set]
        if len(synthetic) < q:
            raise ValueError(f"{group} has only {len(synthetic)} accepted synthetic images; Q={q}")

        rng = random.Random(SELECTION_SEED + index)
        selected = sorted(rng.sample(synthetic, q))
        output = SPLIT_DIR / f"train_q{q}_{group}.txt"
        output.write_text("\n".join(real_lines + selected) + "\n", encoding="utf-8")

        manifest["groups"][group] = {
            "source_split": str(source),
            "output_split": str(output),
            "accepted_pool": len(synthetic),
            "selected": q,
            "condition_prefix_counts": dict(sorted(Counter(condition_prefix(x) for x in selected).items())),
        }
        print(f"[FIXED-Q] {group}: pool={len(synthetic)}, selected={q} -> {output.name}")

    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[MANIFEST]", MANIFEST_JSON)


def build_yaml(group: str, q: int) -> str:
    path = lf.YOLO_ROOT / f"data_q3_q{q}_{group}.yaml"
    content = (
        f"# q3 fixed-Q={q} {group}\n"
        f"path: {str(lf.YOLO_ROOT).replace(chr(92), '/')}\n"
        f"train: splits_leakfix/train_q{q}_{group}.txt\n"
        "val: splits_leakfix/val.txt\n"
        "nc: 1\n"
        "names:\n  0: defect\n"
    )
    path.write_text(content, encoding="utf-8")
    return str(path)


def run_detection(q: int, seeds: list[int], groups: list[str], force_train: bool = False,
                  strict_resume: bool = False) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    old_runs = lf.RUNS_DIR
    lf.RUNS_DIR = RUNS_DIR
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    try:
        for seed in seeds:
            for group in groups:
                key = f"q{q}_{group}_s{seed}"
                run_dir = RUNS_DIR / key
                yaml_path = build_yaml(group, q)
                checkpoint_pt = run_dir / "weights" / "last.pt"
                prior = results.get(key, {})
                strict_done = (prior.get("checkpoint") == "last.pt" and
                               prior.get("train_val") is False and
                               prior.get("fixed_epochs") == lf.EPOCHS and
                               prior.get("detector_batch") == lf.BATCH)
                if force_train or not checkpoint_pt.exists() or (strict_resume and not strict_done):
                    print(f"[TRAIN] {key}")
                    if not lf.train_yolo(yaml_path, run_dir, seed=seed):
                        results[key] = {"error": "train failed"}
                        RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
                        raise RuntimeError(f"Training failed for {key}; stopping to protect the GPU and preserve protocol consistency")
                metrics = lf.eval_yolo(str(checkpoint_pt), yaml_path)
                if metrics:
                    metrics.update({"seed": seed, "q": q, "group": group,
                                    "checkpoint": "last.pt", "train_val": False,
                                    "fixed_epochs": lf.EPOCHS,
                                    "detector_batch": lf.BATCH})
                    results[key] = metrics
                    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    print(
                        f"[DET] {key}: mAP50={metrics['mAP50']:.4f} "
                        f"mAP50_95={metrics['mAP50_95']:.4f} "
                        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
                    )
    finally:
        lf.RUNS_DIR = old_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["prepare", "det", "all"])
    parser.add_argument("--q", type=int, default=Q_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--strict-resume", action="store_true",
                        help="仅重训尚无严格协议元数据的运行")
    args = parser.parse_args()

    if args.cmd in {"prepare", "all"}:
        prepare(args.q, args.groups)
    if args.cmd in {"det", "all"}:
        run_detection(args.q, args.seeds, args.groups,
                      force_train=args.force_train, strict_resume=args.strict_resume)


if __name__ == "__main__":
    main()
