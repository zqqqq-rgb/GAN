# -*- coding: utf-8 -*-
"""Statistical analysis for the fixed-Q=300 detection experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_JSON = PROJECT_ROOT / "data" / "yolo_dataset" / "results_leakfix.json"
FIXEDQ_JSON = PROJECT_ROOT / "data" / "yolo_dataset" / "q3_fixedq_results.json"
OUTPUT_JSON = PROJECT_ROOT / "data" / "yolo_dataset" / "q3_fixedq_statistics.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "yolo_dataset" / "q3_fixedq_statistics.csv"
OUTPUT_MD = PROJECT_ROOT / "deliverables" / "q3_fixedq_statistics.md"

Q = 300
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
GROUPS = [
    "baseline",
    "baseline_stepmatched",
    "E5_tradit",
    "FastGAN",
    "copypaste",
    "E7proj_mses_nosp",
    "E6proj_mses_sp",
    "E6proj_mses_nosp",
    "E7proj_mses_sp",
    "Diffusion_LDM",
]
# ``baseline_stepmatched`` is a Real-only control, not an augmentation method:
# it repeats real images to Q=300 so that epoch step count matches the
# augmentation runs. It is reported in the table and tested against
# ``baseline``, but its own ``_vs_baseline`` entry is not part of the
# augmentation multiplicity family and therefore carries no Holm-adjusted p.
CONTROL_GROUPS = {"baseline", "baseline_stepmatched"}

DISPLAY = {
    "baseline": "Real only",
    "baseline_stepmatched": "Real only (step matched)",
    "E5_tradit": "Traditional augmentation",
    "E4_dcgan": "DCGAN",
    "E2_wgan_gp": "WGAN-GP+FM",
    "E2DA": "WGAN-GP+DiffAugment",
    "FastGAN": "FastGAN",
    "copypaste": "CopyPaste (exact boxes)",
    "E7proj_mses_nosp": "Text projection, no MSES",
    "E6proj_mses_sp": "Class projection+MSES",
    "E6proj_mses_nosp": "Class projection, no MSES",
    "E7proj_mses_sp": "Text projection+MSES",
    "Diffusion_LDM": "Class-conditional latent DDPM",
}


def baseline_key(seed: int) -> str:
    return "lf_baseline" if seed == 42 else f"lf_baseline_s{seed}"


def fixed_key(group: str, seed: int) -> str:
    return f"q{Q}_{group}_s{seed}"


def bootstrap_ci(diff: np.ndarray, repeats: int = 100_000) -> list[float]:
    rng = np.random.default_rng(20260910)
    indices = rng.integers(0, len(diff), size=(repeats, len(diff)))
    return np.quantile(diff[indices].mean(axis=1), [0.025, 0.975]).tolist()


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summary(values: np.ndarray) -> dict:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "sd": sd, "ci95": [mean - half, mean + half],
            "values": values.tolist()}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    wilcoxon = stats.wilcoxon(diff, alternative="two-sided", method="auto")
    p_w = float(wilcoxon.pvalue)
    p_t = float(stats.ttest_rel(a, b).pvalue)
    sd_diff = float(diff.std(ddof=1))
    nonzero = diff[diff != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        w_plus = float(ranks[nonzero > 0].sum())
        w_minus = float(ranks[nonzero < 0].sum())
        rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
    else:
        w_plus = w_minus = 0.0
        rank_biserial = 0.0
    shapiro = stats.shapiro(diff)
    return {
        "delta_mean": float(diff.mean()),
        "delta_sd": sd_diff,
        "delta_ci95_bootstrap": bootstrap_ci(diff),
        "cohen_dz": float(diff.mean() / sd_diff) if sd_diff > 0 else None,
        "rank_biserial": float(rank_biserial),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "signed_rank_sums": {"positive": w_plus, "negative": w_minus},
        "p_wilcoxon_two_sided": p_w,
        "p_paired_t_two_sided": p_t,
        "difference_normality_shapiro": {"W": float(shapiro.statistic), "p": float(shapiro.pvalue)},
        "wins_ties_losses": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
    }


def main():
    baseline_raw = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    fixed_raw = json.loads(FIXEDQ_JSON.read_text(encoding="utf-8"))
    arrays = {}
    for group in GROUPS:
        if group == "Diffusion_LDM":
            generator_groups = ["Diffusion_LDM", "Diffusion_LDM_gs1", "Diffusion_LDM_gs2"]
            arrays[group] = {}
            for metric in METRICS:
                matrix = []
                for generator_group in generator_groups:
                    row = []
                    for seed in SEEDS:
                        key = fixed_key(generator_group, seed)
                        if key not in fixed_raw or metric not in fixed_raw[key]:
                            raise KeyError(f"Missing complete result: {key}")
                        if (fixed_raw[key].get("checkpoint") != "last.pt" or
                                fixed_raw[key].get("train_val") is not False or
                                fixed_raw[key].get("detector_batch") != 4):
                            raise ValueError(f"Result does not use the corrected fixed-epoch protocol: {key}")
                        row.append(fixed_raw[key][metric])
                    matrix.append(row)
                arrays[group][metric] = np.asarray(matrix, dtype=float).mean(axis=0)
            continue
        records = []
        for seed in SEEDS:
            key = baseline_key(seed) if group == "baseline" else fixed_key(group, seed)
            source = baseline_raw if group == "baseline" else fixed_raw
            if key not in source or any(metric not in source[key] for metric in METRICS):
                raise KeyError(f"Missing complete result: {key}")
            if (source[key].get("checkpoint") != "last.pt" or
                    source[key].get("train_val") is not False or
                    source[key].get("detector_batch") != 4):
                raise ValueError(f"Result does not use the corrected fixed-epoch protocol: {key}")
            records.append(source[key])
        arrays[group] = {metric: np.asarray([record[metric] for record in records]) for metric in METRICS}

    per_group = {group: {metric: summary(arrays[group][metric]) for metric in METRICS} for group in GROUPS}
    augmentation_groups = [group for group in GROUPS if group not in CONTROL_GROUPS]
    direct_pairs = [
        ("E7proj_mses_sp", "E7proj_mses_nosp"),
        ("E7proj_mses_sp", "E6proj_mses_sp"),
        ("E7proj_mses_nosp", "E6proj_mses_nosp"),
        ("E6proj_mses_sp", "E6proj_mses_nosp"),
        ("E7proj_mses_sp", "copypaste"),
    ]
    pairs = (
        [(group, "baseline") for group in GROUPS if group != "baseline"]
        + [(group, "baseline_stepmatched") for group in augmentation_groups]
        + direct_pairs
    )
    comparisons = {
        f"{a}_vs_{b}": {metric: compare(arrays[a][metric], arrays[b][metric]) for metric in METRICS}
        for a, b in pairs
    }

    primary = [f"{group}_vs_baseline" for group in augmentation_groups]
    corrected = holm([comparisons[key]["mAP50"]["p_wilcoxon_two_sided"] for key in primary])
    for key, value in zip(primary, corrected):
        comparisons[key]["mAP50"]["p_wilcoxon_holm"] = value

    stepmatched_keys = [f"{group}_vs_baseline_stepmatched" for group in augmentation_groups]
    stepmatched_corrected = holm([
        comparisons[key]["mAP50"]["p_wilcoxon_two_sided"] for key in stepmatched_keys
    ])
    for key, value in zip(stepmatched_keys, stepmatched_corrected):
        comparisons[key]["mAP50"]["p_wilcoxon_holm"] = value

    friedman = {}
    for metric in METRICS:
        test = stats.friedmanchisquare(*(arrays[group][metric] for group in GROUPS))
        friedman[metric] = {
            "chi_square": float(test.statistic),
            "df": len(GROUPS) - 1,
            "p": float(test.pvalue),
            "kendalls_W": float(test.statistic / (len(SEEDS) * (len(GROUPS) - 1))),
        }

    output = {
        "design": {
            "q": Q,
            "real_training_images": 193,
            "held_out_evaluation_images": 48,
            "detector_training_seeds": SEEDS,
            "pairing": "fixed 193/48 split and fixed-Q; paired by detector-training seed",
            "checkpoint_protocol": "validation disabled during optimization; fixed 30 epochs; batch 4; last.pt evaluated once",
            "excluded_from_q300": {
                "DCGAN": "3 accepted images at confidence 0.10",
                "WGAN-GP+FM": "142 accepted images at confidence 0.10",
                "WGAN-GP+DiffAugment": "98 accepted images at confidence 0.10",
            },
            "primary_test": "two-sided paired Wilcoxon signed-rank",
            "multiplicity": f"Holm correction across {len(primary)} augmentation-vs-real-only tests",
            "step_matched_control": (
                "Real-only images are repeated with replacement to 493 entries per epoch; "
                f"a separate Holm family covers {len(stepmatched_keys)} augmentation comparisons against this control"
            ),
            "global_test": "Friedman test across methods, with Kendall's W",
            "effect_sizes": "paired rank-biserial correlation (primary nonparametric) and Cohen dz (sensitivity)",
            "analysis_status": (
                "Original comparisons were planned before result inspection. Diffusion_LDM is a post-hoc "
                "protocol extension using three generator-training seeds. Its three results are averaged within "
                "each detector-training seed before paired inference, and the method is included once in the "
                "updated Holm family; all generator-seed outcomes are reported separately."
            ),
        },
        "per_group": per_group,
        "global_friedman": friedman,
        "comparisons": comparisons,
    }
    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "n", "mean", "sd", "ci95_low", "ci95_high"])
        for group in GROUPS:
            for metric in METRICS:
                values = per_group[group][metric]
                writer.writerow([group, metric, values["n"], values["mean"], values["sd"], *values["ci95"]])

    lines = [
        f"# Fixed-Q={Q}, fixed-split 10-seed detection statistics",
        "",
        "Rows are paired by detector-training seed against Real only. For augmentation methods the "
        "p-value is Holm-adjusted across the augmentation family; for the Real-only step-matched control "
        "it is the unadjusted two-sided Wilcoxon p-value, since that row is a control rather than a member "
        "of the augmentation family.",
        "",
        "| Method | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Δ mAP@0.5 vs real-only | p (Holm; unadjusted for control) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        cells = [f"{100*per_group[group][metric]['mean']:.2f} ± {100*per_group[group][metric]['sd']:.2f}" for metric in METRICS]
        if group == "baseline":
            delta, p_value = "—", "—"
        elif group == "baseline_stepmatched":
            result = comparisons["baseline_stepmatched_vs_baseline"]["mAP50"]
            delta = f"{100*result['delta_mean']:+.2f} pp"
            p_value = f"{result['p_wilcoxon_two_sided']:.4g}"
        else:
            result = comparisons[f"{group}_vs_baseline"]["mAP50"]
            delta = f"{100*result['delta_mean']:+.2f} pp"
            if "p_wilcoxon_holm" not in result:
                raise KeyError(
                    f"{group}_vs_baseline is in the report table but absent from the Holm family; "
                    "add it to augmentation_groups or to the step-matched/unadjusted branch."
                )
            p_value = f"{result['p_wilcoxon_holm']:.4g}"
        lines.append(f"| {DISPLAY[group]} | " + " | ".join(cells + [delta, p_value]) + " |")
    lines.extend(["", "## Direct prespecified ablations", "",
                  "| Comparison | Δ mAP@0.5 | 95% paired-bootstrap CI | Wilcoxon p | Rank-biserial r | Cohen dz |",
                  "|---|---:|---:|---:|---:|---:|"])
    for a, b in direct_pairs:
        result = comparisons[f"{a}_vs_{b}"]["mAP50"]
        ci = result["delta_ci95_bootstrap"]
        lines.append(
            f"| {DISPLAY[a]} vs {DISPLAY[b]} | {100*result['delta_mean']:+.2f} pp | "
            f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {result['p_wilcoxon_two_sided']:.4g} | "
            f"{result['rank_biserial']:.3f} | {result['cohen_dz']:.3f} |"
        )
    fr = friedman["mAP50"]
    lines.extend([
        "", "## Global paired comparison", "",
        f"Friedman χ²({fr['df']}) = {fr['chi_square']:.3f}, p = {fr['p']:.4g}, "
        f"Kendall's W = {fr['kendalls_W']:.3f}.",
        "", "Shapiro–Wilk results for paired differences are diagnostic only; the prespecified "
        "primary pairwise test remains the two-sided Wilcoxon signed-rank test.",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
