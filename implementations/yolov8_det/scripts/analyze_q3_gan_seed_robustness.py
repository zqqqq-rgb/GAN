# -*- coding: utf-8 -*-
"""Summarize GAN-seed robustness without treating detector runs as GAN replicates."""

from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]


import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(str(_release_root))
DET_PATH = ROOT / "data/yolo_dataset/q3_fixedq_results.json"
GEN_PATH = ROOT / "data/yolo_dataset/q3_generative_metrics.json"
OUT_JSON = ROOT / "data/yolo_dataset/q3_gan_seed_robustness.json"
OUT_CSV = ROOT / "data/yolo_dataset/q3_gan_seed_robustness.csv"
OUT_MD = ROOT / "deliverables/q3_gan_seed_robustness.md"

DETECTOR_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
CONFIGS = {
    "TextProjection_noMSES": {
        42: "E7proj_mses_nosp",
        1: "E7proj_mses_nosp_s1",
        2: "E7proj_mses_nosp_s2",
    },
    "ClassProjection_noMSES": {
        42: "E6proj_mses_nosp",
        1: "E6proj_mses_nosp_s1",
        2: "E6proj_mses_nosp_s2",
    },
    "TextProjection_MSES": {
        42: "E7proj_mses_sp",
        1: "E7proj_mses_sp_s1",
        2: "E7proj_mses_sp_s2",
    },
}


def mean_sd(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "n": int(len(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> None:
    det = json.loads(DET_PATH.read_text(encoding="utf-8"))
    gen = json.loads(GEN_PATH.read_text(encoding="utf-8"))
    output = {
        "protocol": {
            "gan_seeds": [42, 1, 2],
            "detector_seeds": DETECTOR_SEEDS,
            "split": "fixed 193 train / 48 held-out validation",
            "q": 300,
            "inference_note": (
                "The ten detector runs are nested within each GAN seed. GAN-level summaries use "
                "the three GAN-seed means (n=3); detector runs are not counted as 30 independent GAN replicates."
            ),
        },
        "configurations": {},
    }
    rows = []
    for config, groups in CONFIGS.items():
        item = {"detection_by_gan_seed": {}, "detection_across_gan_seeds": {}, "generative_by_gan_seed": {}}
        for gan_seed, group in groups.items():
            seed_metrics = {metric: [] for metric in METRICS}
            for detector_seed in DETECTOR_SEEDS:
                key = f"q300_{group}_s{detector_seed}"
                result = det.get(key)
                if not result or "error" in result:
                    raise KeyError(f"Missing reliable result: {key}")
                for metric in METRICS:
                    seed_metrics[metric].append(float(result[metric]))
            item["detection_by_gan_seed"][str(gan_seed)] = {
                metric: mean_sd(values) for metric, values in seed_metrics.items()
            }
            gen_key = config if gan_seed == 42 else f"{config}_seed{gan_seed}"
            g = gen.get(gen_key)
            if not g or "error" in g:
                raise KeyError(f"Missing reliable result: {gen_key}")
            item["generative_by_gan_seed"][str(gan_seed)] = {
                "fid": float(g["fid_torchvision_inception"]),
                "kid": float(g["kid"]["mean"]),
                "lpips_pooled": float(g["lpips_pooled"]["mean"]),
                "mses_discrepancy": float(g["mses_discrepancy"]["mean"]),
            }

        for metric in METRICS:
            gan_means = [item["detection_by_gan_seed"][str(s)][metric]["mean"] for s in groups]
            item["detection_across_gan_seeds"][metric] = mean_sd(gan_means)
            summary = item["detection_across_gan_seeds"][metric]
            rows.append({"configuration": config, "domain": "detection", "metric": metric, **summary})
        item["generative_across_gan_seeds"] = {}
        for metric in ["fid", "kid", "lpips_pooled", "mses_discrepancy"]:
            vals = [item["generative_by_gan_seed"][str(s)][metric] for s in groups]
            item["generative_across_gan_seeds"][metric] = mean_sd(vals)
            rows.append({
                "configuration": config,
                "domain": "generative",
                "metric": metric,
                **item["generative_across_gan_seeds"][metric],
            })
        output["configurations"][config] = item

    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["configuration", "domain", "metric", "mean", "sd", "n", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GAN-seed robustness under fixed Q=300",
        "",
        "> Each GAN seed is evaluated with the same ten detector-training seeds. The inferential unit for GAN robustness is the GAN seed (n=3), not the 30 nested detector runs.",
        "",
        "| Configuration | mAP@0.5 across GAN seeds | mAP@0.5:0.95 across GAN seeds | FID across GAN seeds |",
        "|---|---:|---:|---:|",
    ]
    for config, item in output["configurations"].items():
        a = item["detection_across_gan_seeds"]["mAP50"]
        b = item["detection_across_gan_seeds"]["mAP50_95"]
        c = item["generative_across_gan_seeds"]["fid"]
        lines.append(f"| {config} | {a['mean']:.4f} ± {a['sd']:.4f} | {b['mean']:.4f} ± {b['sd']:.4f} | {c['mean']:.2f} ± {c['sd']:.2f} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_JSON)
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
