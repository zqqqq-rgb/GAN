# -*- coding: utf-8 -*-
"""Paired detector-seed analysis for the new NEU-DET projection protocol."""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import json
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel, wilcoxon

ROOT = Path(str(_release_root))
INPUT = ROOT / "data/neu_det/yolo/q3_textproj_results.json"
OUTPUT = ROOT / "data/neu_det/yolo/q3_textproj_statistics.json"
REPORT = ROOT / "deliverables/q3_neu_projection_statistics.md"
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


data = json.loads(INPUT.read_text(encoding="utf-8"))
results, raw_p = {}, []
for metric in METRICS:
    baseline = np.array([data[f"baseline_s{s}"][metric] for s in SEEDS], dtype=float)
    augmented = np.array([data[f"textproj_q300_s{s}"][metric] for s in SEEDS], dtype=float)
    delta = augmented - baseline
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    ci = t.interval(0.95, len(delta) - 1, loc=delta.mean(), scale=se)
    t_p = float(ttest_rel(augmented, baseline).pvalue)
    w_p = float(wilcoxon(delta).pvalue)
    raw_p.append(w_p)
    results[metric] = {
        "baseline_mean": float(baseline.mean()), "baseline_sd": float(baseline.std(ddof=1)),
        "augmented_mean": float(augmented.mean()), "augmented_sd": float(augmented.std(ddof=1)),
        "paired_delta_mean": float(delta.mean()), "paired_delta_sd": float(delta.std(ddof=1)),
        "paired_delta_95ci": [float(ci[0]), float(ci[1])],
        "relative_change_percent": float(100 * delta.mean() / baseline.mean()),
        "paired_t_p": t_p, "wilcoxon_p": w_p,
    }
for metric, adjusted in zip(METRICS, holm(raw_p)):
    results[metric]["wilcoxon_holm_p_across_four_metrics"] = adjusted

payload = {
    "protocol": {"detector_seeds": SEEDS, "paired_by_seed": True, "q": 300,
                 "split": "1439 train / 360 held-out validation", "multiple_testing": "Holm across four reported metrics"},
    "results": results,
}
OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# NEU-DET current-protocol paired results", "",
         "| Metric | Baseline | Text projection + MSES, Q=300 | Paired delta (95% CI) | Wilcoxon p | Holm p |",
         "|---|---:|---:|---:|---:|---:|"]
for metric in METRICS:
    r = results[metric]
    lines.append(
        f"| {metric} | {r['baseline_mean']:.4f} ± {r['baseline_sd']:.4f} | "
        f"{r['augmented_mean']:.4f} ± {r['augmented_sd']:.4f} | "
        f"{r['paired_delta_mean']:+.4f} [{r['paired_delta_95ci'][0]:+.4f}, {r['paired_delta_95ci'][1]:+.4f}] | "
        f"{r['wilcoxon_p']:.4f} | {r['wilcoxon_holm_p_across_four_metrics']:.4f} |"
    )
lines += ["", "> Unadjusted mAP tests are positive, but none of the four metrics remains significant after Holm correction; the manuscript should describe this as a small, protocol-dependent gain rather than decisive cross-domain superiority."]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(REPORT)
