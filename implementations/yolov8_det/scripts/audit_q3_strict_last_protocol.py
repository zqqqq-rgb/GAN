# -*- coding: utf-8 -*-
"""Fail closed unless every manuscript-facing detector result uses the strict protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/yolo_dataset"
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FIXED_GROUPS = [
    "baseline_stepmatched", "E5_tradit", "FastGAN", "copypaste",
    "E7proj_mses_nosp", "E6proj_mses_sp", "E6proj_mses_nosp", "E7proj_mses_sp",
    "E7proj_mses_nosp_s1", "E7proj_mses_nosp_s2",
    "E6proj_mses_nosp_s1", "E6proj_mses_nosp_s2",
    "E7proj_mses_sp_s1", "E7proj_mses_sp_s2",
    "Diffusion_LDM", "Diffusion_LDM_gs1", "Diffusion_LDM_gs2",
]


def strict(record: dict, key: str) -> None:
    required = ["mAP50", "mAP50_95", "precision", "recall"]
    if any(name not in record for name in required):
        raise ValueError(f"Incomplete metrics: {key}")
    if record.get("checkpoint") != "last.pt":
        raise ValueError(f"Non-last checkpoint: {key}")
    if record.get("train_val") is not False or record.get("fixed_epochs") != 30:
        raise ValueError(f"Non-strict training protocol: {key}")
    if record.get("detector_batch") != 4:
        raise ValueError(f"Unexpected detector batch size: {key}")


def main() -> None:
    baseline = json.loads((DATA / "results_leakfix.json").read_text(encoding="utf-8"))
    fixed = json.loads((DATA / "q3_fixedq_results.json").read_text(encoding="utf-8"))
    qsens = json.loads((DATA / "q3_q_sensitivity_current.json").read_text(encoding="utf-8"))
    checked = 0
    for seed in SEEDS:
        key = "lf_baseline" if seed == 42 else f"lf_baseline_s{seed}"
        strict(baseline[key], key)
        checked += 1
    for group in FIXED_GROUPS:
        for seed in SEEDS:
            key = f"q300_{group}_s{seed}"
            strict(fixed[key], key)
            checked += 1
    for q in [100, 200, 300, 500]:
        for seed in SEEDS:
            key = f"q{q}_s{seed}"
            strict(qsens[key], key)
            checked += 1
    output = {
        "status": "PASS",
        "checked_results": checked,
        "checkpoint": "last.pt",
        "training_validation": False,
        "fixed_epochs": 30,
        "detector_batch": 4,
        "evaluation_images": 48,
        "limitation": "The same fixed evaluation split has been inspected historically; no new independent images were available.",
    }
    path = DATA / "q3_strict_last_protocol_audit.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
