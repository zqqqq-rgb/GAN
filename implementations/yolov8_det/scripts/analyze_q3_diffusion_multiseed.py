# -*- coding: utf-8 -*-
"""Aggregate the 3 x 10 nested diffusion augmentation experiment.

Generator seed is not treated as 30 independent replicates. For inference
against real-only, the three generator-seed results are averaged within each
paired detector-training seed, leaving ten paired experimental units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import analyze_q3_fixedq_statistics as base


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "data/yolo_dataset/q3_fixedq_results.json"
BASELINE = ROOT / "data/yolo_dataset/results_leakfix.json"
GENERATIVE = ROOT / "data/yolo_dataset/q3_generative_metrics.json"
OUTPUT = ROOT / "data/yolo_dataset/q3_diffusion_multiseed_statistics.json"
OUTPUT_MD = ROOT / "deliverables/q3_diffusion_multiseed_statistics.md"
GENERATOR_GROUPS = {42: "Diffusion_LDM", 1: "Diffusion_LDM_gs1", 2: "Diffusion_LDM_gs2"}
GENERATOR_METRIC_NAMES = {
    42: "ClassConditional_Latent_DDPM",
    1: "ClassConditional_Latent_DDPM_seed1",
    2: "ClassConditional_Latent_DDPM_seed2",
}


def describe(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def main() -> None:
    raw = json.loads(RESULTS.read_text(encoding="utf-8"))
    baseline_raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    generative_raw = json.loads(GENERATIVE.read_text(encoding="utf-8"))
    matrices: dict[str, np.ndarray] = {}
    per_generator: dict[str, dict] = {}
    for metric in base.METRICS:
        rows = []
        for generator_seed, group in GENERATOR_GROUPS.items():
            values = []
            for detector_seed in base.SEEDS:
                key = f"q300_{group}_s{detector_seed}"
                if key not in raw or metric not in raw[key]:
                    raise KeyError(f"Missing complete result: {key}")
                values.append(raw[key][metric])
            rows.append(values)
            per_generator.setdefault(str(generator_seed), {})[metric] = describe(np.asarray(values))
        matrices[metric] = np.asarray(rows, dtype=float)

    detector_paired = {}
    for metric, matrix in matrices.items():
        averaged = matrix.mean(axis=0)
        baseline_values = np.asarray([
            baseline_raw[base.baseline_key(seed)][metric] for seed in base.SEEDS
        ], dtype=float)
        detector_paired[metric] = {
            "diffusion_three_generator_seed_average": base.summary(averaged),
            "real_only": base.summary(baseline_values),
            "paired_comparison": base.compare(averaged, baseline_values),
            "generator_seed_mean_sd": describe(matrix.mean(axis=1)),
        }

    generation_metrics = {}
    for generator_seed, name in GENERATOR_METRIC_NAMES.items():
        record = generative_raw.get(name)
        if not record or "error" in record:
            raise KeyError(f"Missing complete generative metrics: {name}")
        generation_metrics[str(generator_seed)] = record

    output = {
        "design": {
            "generator_training_seeds": list(GENERATOR_GROUPS),
            "detector_training_seeds": base.SEEDS,
            "real_training_images": 193,
            "held_out_images": 48,
            "synthetic_budget_per_run": 300,
            "inference_unit": "detector seed after averaging the three generator seeds; n=10 paired units",
            "primary_test": "two-sided paired Wilcoxon signed-rank",
            "status": "post-hoc robustness extension; all generator-seed outcomes reported",
        },
        "per_generator_seed": per_generator,
        "detector_paired_aggregate": detector_paired,
        "generative_metrics": generation_metrics,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    m = detector_paired["mAP50"]
    cmp = m["paired_comparison"]
    ci = cmp["delta_ci95_bootstrap"]
    lines = [
        "# Diffusion three-generator-seed robustness analysis",
        "",
        "The three generator seeds are averaged within each detector seed before inference, so the 30 runs are not treated as independent replicates.",
        "",
        "| Generator seed | mAP@0.5 mean ± SD | mAP@0.5:0.95 mean ± SD | Precision mean ± SD | Recall mean ± SD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in GENERATOR_GROUPS:
        s = per_generator[str(seed)]
        cells = [f"{100*s[x]['mean']:.2f} ± {100*s[x]['sd']:.2f}" for x in base.METRICS]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f"Across generator seeds, detector-seed-paired mean mAP@0.5 was {100*m['diffusion_three_generator_seed_average']['mean']:.2f}% "
        f"versus {100*m['real_only']['mean']:.2f}% for real-only (delta {100*cmp['delta_mean']:+.2f} pp, "
        f"95% paired-bootstrap CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}], "
        f"Wilcoxon p={cmp['p_wilcoxon_two_sided']:.6f}, wins/ties/losses={cmp['wins_ties_losses']}).",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
