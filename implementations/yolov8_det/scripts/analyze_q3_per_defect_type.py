# -*- coding: utf-8 -*-
"""Evaluate fixed-Q detectors on the four held-out defect-type strata.

The detector is class-agnostic (one YOLO class named ``defect``).  Therefore,
this script reports localization metrics after stratifying the fixed 48-image
test set by its image-level defect type; it does not reinterpret the detector
as a four-class classifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from ultralytics import YOLO

# The four defect-type strata share one label directory (labels/all_real) but
# contain different image subsets, so every stratum switch makes Ultralytics
# rewrite labels/all_real.cache: save_dataset_cache_file() unlinks the old
# cache before writing the new one, and ~50 unlinks trip the bulk-delete
# safety hook long before the 160 evaluations finish.  Neutralise cache
# persistence instead: labels are re-scanned per evaluation (a 12-13 image
# scan costs milliseconds) and the shared cache file is never written, never
# unlinked, and left exactly as it is on disk.
import ultralytics.data.dataset as _ul_dataset  # noqa: E402
import ultralytics.data.utils as _ul_utils  # noqa: E402


def _no_save_cache(prefix, path, x, version):
    # Keep the in-memory dict consistent with the original function
    # (get_labels pops the "version" key afterwards) while never
    # writing to or unlinking the cache file on disk.
    x["version"] = version
    return None


_ul_dataset.save_dataset_cache_file = _no_save_cache
_ul_utils.save_dataset_cache_file = _no_save_cache


ROOT = Path(__file__).resolve().parents[3]
YOLO_ROOT = ROOT / "data" / "yolo_dataset"
SPLIT_INFO = YOLO_ROOT / "splits_leakfix" / "split_info.json"
SUBSET_DIR = YOLO_ROOT / "splits_per_defect_type"
OUT_JSON = YOLO_ROOT / "q3_per_defect_type_results.json"
SUMMARY_JSON = YOLO_ROOT / "q3_per_defect_type_summary.json"
OUT_MD = ROOT / "deliverables" / "q3_per_defect_type_summary.md"
VAL_PROJECT = ROOT / "deliverables" / "_per_defect_type_val"

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TYPE_KEYS = {
    "凹点": "dent",
    "划痕": "scratch",
    "形变": "deformation",
    "鼓包": "bulge",
}
METHOD_LABELS = {
    "real_only": "Real-only",
    "text_nom": "文本投影无MSES",
    "text_mses": "文本投影+MSES",
    "copypaste": "CopyPaste（精确框）",
}


def model_path(method: str, seed: int) -> Path:
    if method == "real_only":
        suffix = "" if seed == 42 else f"_s{seed}"
        return (
            ROOT
            / "implementations"
            / "yolov8_det"
            / "runs_leakfix"
            / f"yolov8n_baseline_lf{suffix}"
            / "weights"
            / "last.pt"
        )
    run_group = {
        "text_nom": "E7proj_mses_nosp",
        "text_mses": "E7proj_mses_sp",
        "copypaste": "copypaste",
    }[method]
    return (
        ROOT
        / "implementations"
        / "yolov8_det"
        / "runs_q3_fixedq"
        / f"q300_{run_group}_s{seed}"
        / "weights"
        / "last.pt"
    )


def prepare_subsets() -> dict[str, dict[str, str | int]]:
    info = json.loads(SPLIT_INFO.read_text(encoding="utf-8"))
    all_real = YOLO_ROOT / "images" / "all_real"
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, str | int]] = {}
    for zh_name, key in TYPE_KEYS.items():
        names = info["val"][zh_name]
        split_path = SUBSET_DIR / f"val_{key}.txt"
        lines = [(all_real / name).as_posix() for name in names]
        split_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        yaml_path = YOLO_ROOT / f"data_lf_val_{key}.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {YOLO_ROOT.as_posix()}",
                    "train: splits_leakfix/train_baseline.txt",
                    f"val: splits_per_defect_type/val_{key}.txt",
                    "nc: 1",
                    "names:",
                    "  0: defect",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        result[key] = {"zh_name": zh_name, "n_images": len(names), "yaml": str(yaml_path)}
    return result


def evaluate(recompute: bool = False) -> tuple[dict, dict]:
    os.environ.setdefault("WANDB_DISABLED", "true")
    subsets = prepare_subsets()
    results = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    for method in METHOD_LABELS:
        results.setdefault(method, {})
        for seed in SEEDS:
            seed_key = str(seed)
            results[method].setdefault(seed_key, {})
            pt = model_path(method, seed)
            if not pt.exists():
                raise FileNotFoundError(pt)
            model = YOLO(str(pt))
            for type_key, meta in subsets.items():
                if not recompute and type_key in results[method][seed_key]:
                    continue
                metrics = model.val(
                    data=meta["yaml"],
                    imgsz=640,
                    batch=8,
                    device=0,
                    workers=0,
                    cache=False,
                    plots=True,
                    verbose=False,
                    project=str(VAL_PROJECT),
                    name=type_key,
                    exist_ok=True,
                )
                cm = np.asarray(metrics.confusion_matrix.matrix, dtype=float)
                tp = float(cm[0, 0]) if cm.shape == (2, 2) else None
                fp = float(cm[0, 1]) if cm.shape == (2, 2) else None
                fn = float(cm[1, 0]) if cm.shape == (2, 2) else None
                results[method][seed_key][type_key] = {
                    "n_images": int(meta["n_images"]),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "mAP50": float(metrics.box.map50),
                    "mAP50_95": float(metrics.box.map),
                    "tp_at_conf025": tp,
                    "fp_at_conf025": fp,
                    "fn_at_conf025": fn,
                }
                OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[{method} seed={seed} {type_key}] "
                    f"AP50={metrics.box.map50:.4f} R={metrics.box.mr:.4f}"
                )
    return results, subsets


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_bootstrap_ci(delta: np.ndarray, seed: int, n_boot: int = 100_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def summarize(results: dict, subsets: dict) -> dict:
    summary: dict = {"protocol": {"seeds": SEEDS, "q": 300, "detector_classes": 1}, "types": {}}
    for type_key, meta in subsets.items():
        entry = {"zh_name": meta["zh_name"], "n_images": meta["n_images"], "methods": {}}
        for method in METHOD_LABELS:
            rows = [results[method][str(seed)][type_key] for seed in SEEDS]
            method_stats = {}
            for metric in [
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
                "tp_at_conf025",
                "fp_at_conf025",
                "fn_at_conf025",
            ]:
                method_stats[metric] = dict(zip(["mean", "sd"], mean_sd([r[metric] for r in rows])))
            entry["methods"][method] = method_stats
        base = np.asarray([results["real_only"][str(seed)][type_key]["mAP50"] for seed in SEEDS])
        for method in ["text_nom", "text_mses", "copypaste"]:
            vals = np.asarray([results[method][str(seed)][type_key]["mAP50"] for seed in SEEDS])
            delta = vals - base
            entry["methods"][method]["paired_delta_mAP50"] = {
                "mean": float(delta.mean()),
                "sd": float(delta.std(ddof=1)),
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        summary["types"][type_key] = entry

    # The only type-wise inferential family is the four paired comparisons of
    # the best-performing GAN configuration (text projection without MSES)
    # against Real-only. Other method columns remain descriptive controls.
    raw_p = []
    keys = list(subsets)
    for type_key in keys:
        base = np.asarray([results["real_only"][str(seed)][type_key]["mAP50"] for seed in SEEDS])
        vals = np.asarray([results["text_nom"][str(seed)][type_key]["mAP50"] for seed in SEEDS])
        delta = vals - base
        p_value = float(wilcoxon(delta, alternative="two-sided", method="auto").pvalue)
        lo, hi = paired_bootstrap_ci(delta, seed=20260914 + keys.index(type_key))
        summary["types"][type_key]["text_nom_vs_real"] = {
            "mean_delta": float(delta.mean()),
            "bootstrap_ci95": [lo, hi],
            "wilcoxon_p": p_value,
        }
        raw_p.append(p_value)
    for type_key, adjusted in zip(keys, holm_adjust(raw_p)):
        summary["types"][type_key]["text_nom_vs_real"]["holm_p"] = adjusted

    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def pct(stat: dict) -> str:
    return f"{100 * stat['mean']:.2f}±{100 * stat['sd']:.2f}"


def write_markdown(summary: dict) -> None:
    lines = [
        "# Q3 per-defect-type localization summary",
        "",
        "The detector has one class (`defect`). Metrics are stratified by image-level defect type.",
        "Values are mean ± sample SD across 10 paired detector seeds.",
        "",
        "| Type (n) | Real-only AP50 | Text no-MSES AP50 | Delta (95% CI) | Holm p | Text+MSES AP50 | CopyPaste AP50 | Real-only recall | Text no-MSES recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in summary["types"].items():
        m = entry["methods"]
        comp = entry["text_nom_vs_real"]
        delta = 100 * comp["mean_delta"]
        lo, hi = [100 * x for x in comp["bootstrap_ci95"]]
        lines.append(
            f"| {entry['zh_name']} ({entry['n_images']}) | {pct(m['real_only']['mAP50'])} | "
            f"{pct(m['text_nom']['mAP50'])} | {delta:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
            f"{comp['holm_p']:.4f} | {pct(m['text_mses']['mAP50'])} | "
            f"{pct(m['copypaste']['mAP50'])} | {pct(m['real_only']['recall'])} | "
            f"{pct(m['text_nom']['recall'])} |"
        )
    lines += [
        "",
        "Confusion counts use the Ultralytics default confusion-matrix operating point (confidence 0.25).",
        "They are descriptive only because each stratum contains 10–13 images.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    results, subsets = evaluate(recompute=args.recompute)
    summary = summarize(results, subsets)
    write_markdown(summary)
    print(f"[DONE] {SUMMARY_JSON}")
    print(f"[DONE] {OUT_MD}")


if __name__ == "__main__":
    main()
