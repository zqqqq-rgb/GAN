# -*- coding: utf-8 -*-
"""Paired, multiplicity-aware analysis of fixed-Q NEU-DET comparators."""

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(str(_release_root))
OLD = ROOT / "data/neu_det/yolo/q3_textproj_results.json"
NEW = ROOT / "data/neu_det/yolo/q3_neu_comparator_results.json"
OUT = ROOT / "data/neu_det/yolo/q3_neu_comparator_statistics.json"
REPORT = ROOT / "deliverables/q3_neu_comparator_statistics.md"
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
GROUPS = {
    "text_mses": ("Text projection + MSES", "textproj_q300"),
    "text_nom": ("Text projection, no MSES", "text_nom_q300"),
    "class_nom": ("Class projection, no MSES", "class_nom_q300"),
    "copypaste": ("CopyPaste (exact labels)", "copypaste_q300"),
}
STEPMATCHED_KEY = "baseline_stepmatched"


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def rank_biserial(delta):
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def bootstrap_ci(delta, seed=20260914, repeats=100000):
    rng = np.random.default_rng(seed)
    means = rng.choice(delta, size=(repeats, len(delta)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


old = json.loads(OLD.read_text(encoding="utf-8"))
new = json.loads(NEW.read_text(encoding="utf-8"))
baseline = {metric: np.array([old[f"baseline_s{s}"][metric] for s in SEEDS]) for metric in METRICS}

payload = {"protocol": {"paired_detector_seeds": SEEDS, "q": 300,
                         "primary_outcome": "mAP50",
                         "primary_multiplicity": "Holm across four augmentation methods",
                         "step_matched_control": (
                             "1439 unique real images plus 300 real entries sampled with "
                             "replacement using selection seed 20260912"
                         ),
                         "bootstrap_repeats": 100000}, "groups": {}}

stepmatched = {
    metric: np.array([old[f"{STEPMATCHED_KEY}_s{s}"][metric] for s in SEEDS])
    for metric in METRICS
}
payload["step_matched_control"] = {"label": "Real-only (step matched)", "metrics": {}}
for metric in METRICS:
    delta = stepmatched[metric] - baseline[metric]
    payload["step_matched_control"]["metrics"][metric] = {
        "baseline_mean": float(baseline[metric].mean()),
        "baseline_sd": float(baseline[metric].std(ddof=1)),
        "mean": float(stepmatched[metric].mean()),
        "sd": float(stepmatched[metric].std(ddof=1)),
        "delta_mean": float(delta.mean()),
        "delta_ci95": bootstrap_ci(delta),
        "wilcoxon_p_unadjusted_control": float(
            wilcoxon(delta, alternative="two-sided").pvalue
        ),
        "rank_biserial": rank_biserial(delta),
        "cohens_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else None,
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
    }

primary_raw = []
stepmatched_raw = []
for short, (label, key) in GROUPS.items():
    payload["groups"][short] = {"label": label, "metrics": {}}
    source = old if short == "text_mses" else new
    for metric in METRICS:
        augmented = np.array([source[f"{key}_s{s}"][metric] for s in SEEDS])
        delta = augmented - baseline[metric]
        p = float(wilcoxon(delta, alternative="two-sided").pvalue)
        result = {
            "baseline_mean": float(baseline[metric].mean()),
            "baseline_sd": float(baseline[metric].std(ddof=1)),
            "mean": float(augmented.mean()), "sd": float(augmented.std(ddof=1)),
            "delta_mean": float(delta.mean()), "delta_ci95": bootstrap_ci(delta),
            "wilcoxon_p": p, "rank_biserial": rank_biserial(delta),
            "cohens_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else None,
            "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
        }
        payload["groups"][short]["metrics"][metric] = result
        if metric == "mAP50":
            primary_raw.append(p)
            step_delta = augmented - stepmatched[metric]
            step_p = float(wilcoxon(step_delta, alternative="two-sided").pvalue)
            result["vs_step_matched"] = {
                "control_mean": float(stepmatched[metric].mean()),
                "delta_mean": float(step_delta.mean()),
                "delta_ci95": bootstrap_ci(step_delta, seed=20260915),
                "wilcoxon_p": step_p,
                "rank_biserial": rank_biserial(step_delta),
                "cohens_dz": (
                    float(step_delta.mean() / step_delta.std(ddof=1))
                    if step_delta.std(ddof=1) else None
                ),
                "wins": int((step_delta > 0).sum()),
                "losses": int((step_delta < 0).sum()),
            }
            stepmatched_raw.append(step_p)

for (short, _), adjusted in zip(GROUPS.items(), holm(primary_raw)):
    payload["groups"][short]["metrics"]["mAP50"]["holm_p_across_methods"] = adjusted
for (short, _), adjusted in zip(GROUPS.items(), holm(stepmatched_raw)):
    payload["groups"][short]["metrics"]["mAP50"]["vs_step_matched"][
        "holm_p_across_methods"
    ] = adjusted

matrix = [baseline["mAP50"], stepmatched["mAP50"]]
for short, (_, key) in GROUPS.items():
    source = old if short == "text_mses" else new
    matrix.append(np.array([source[f"{key}_s{s}"]["mAP50"] for s in SEEDS]))
friedman = friedmanchisquare(*matrix)
payload["overall_mAP50"] = {
    "friedman_chi2": float(friedman.statistic), "df": len(matrix) - 1,
    "p": float(friedman.pvalue),
    "kendalls_w": float(friedman.statistic / (len(SEEDS) * (len(matrix) - 1))),
}

OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# NEU-DET fixed-Q comparator analysis", "",
         "Primary outcome: mAP@0.5; paired by 10 detector seeds. The four augmentation comparisons form one Holm family against Real-only and a separate Holm family against the step-matched control. The control-vs-Real-only p-value is unadjusted.", "",
         "| Method | mAP@0.5 | Delta vs Real-only (95% CI) | p / Holm p | Delta vs step matched (95% CI) | Holm p |",
         "|---|---:|---:|---:|---:|---:|"]
s = payload["step_matched_control"]["metrics"]["mAP50"]
lines.append(
    f"| Real-only | {baseline['mAP50'].mean():.4f} ± {baseline['mAP50'].std(ddof=1):.4f} | — | — | — | — |"
)
lines.append(
    f"| Real-only (step matched) | {s['mean']:.4f} ± {s['sd']:.4f} | "
    f"{s['delta_mean']:+.4f} [{s['delta_ci95'][0]:+.4f}, {s['delta_ci95'][1]:+.4f}] | "
    f"{s['wilcoxon_p_unadjusted_control']:.4f} (unadjusted) | — | — |"
)
for short, (label, _) in GROUPS.items():
    r = payload["groups"][short]["metrics"]["mAP50"]
    rs = r["vs_step_matched"]
    lines.append(f"| {label} | {r['mean']:.4f} ± {r['sd']:.4f} | {r['delta_mean']:+.4f} "
                 f"[{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}] | "
                 f"{r['wilcoxon_p']:.4f} / {r['holm_p_across_methods']:.4f} | "
                 f"{rs['delta_mean']:+.4f} [{rs['delta_ci95'][0]:+.4f}, {rs['delta_ci95'][1]:+.4f}] | "
                 f"{rs['holm_p_across_methods']:.4f} |")
o = payload["overall_mAP50"]
lines += ["", f"Overall Friedman chi-square({o['df']})={o['friedman_chi2']:.3f}, "
          f"p={o['p']:.6g}, Kendall W={o['kendalls_w']:.3f}.", "",
          "> Interpretation must follow the Holm-adjusted primary comparisons; secondary metrics are descriptive and must not be used to rescue a non-significant primary result."]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(REPORT)
print(json.dumps(payload["overall_mAP50"], indent=2))
