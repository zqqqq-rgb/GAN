# -*- coding: utf-8 -*-
"""Image-level bootstrap uncertainty for the fixed-Q detector comparison.

Detector-training seeds are optimization repeats on one fixed 48-image test
set.  They are therefore averaged within each method.  Uncertainty for the
held-out image sample is then estimated by paired bootstrap resampling of the
48 test images.  The same bootstrap draw is used for every method and seed.

All validation predictions are cached so the bootstrap can be reproduced
without loading the detector checkpoints again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import compute_ap


ROOT = Path(__file__).resolve().parents[3]
YOLO_ROOT = ROOT / "data" / "yolo_dataset"
VAL_LIST = YOLO_ROOT / "splits_leakfix" / "val.txt"
RUNS_LF = ROOT / "implementations" / "yolov8_det" / "runs_leakfix"
RUNS_Q = ROOT / "implementations" / "yolov8_det" / "runs_q3_fixedq"
PREDICTIONS_JSON = YOLO_ROOT / "q3_image_bootstrap_predictions.json"
SUMMARY_JSON = YOLO_ROOT / "q3_image_bootstrap_statistics.json"
OUTPUT_MD = ROOT / "deliverables" / "q3_image_bootstrap_statistics.md"
VAL_YAML = YOLO_ROOT / "data_lf_baseline.yaml"

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BOOTSTRAP_SEED = 20260921
DEFAULT_REPEATS = 10_000
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

METHODS = {
    "baseline": "Real-only",
    "baseline_stepmatched": "Real-only (step matched)",
    "E5_tradit": "Traditional augmentation",
    "FastGAN": "FastGAN",
    "copypaste": "CopyPaste (exact boxes)",
    "E6proj_mses_nosp": "Class projection, no MSES",
    "E6proj_mses_sp": "Class projection+MSES",
    "E7proj_mses_nosp": "Text projection, no MSES",
    "E7proj_mses_sp": "Text projection+MSES",
    "Diffusion_LDM": "Class-conditional latent DDPM",
}


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_images_and_gt() -> tuple[list[Path], dict[str, list[float]]]:
    images = [Path(line.strip()) for line in VAL_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(images) != 48:
        raise ValueError(f"Expected 48 held-out images, found {len(images)}")
    gt = {}
    for image_path in images:
        rows = [row for row in label_path(image_path).read_text(encoding="utf-8").splitlines() if row.strip()]
        if len(rows) != 1:
            raise ValueError(f"Expected one box for {image_path}, found {len(rows)}")
        _, xc, yc, bw, bh = map(float, rows[0].split())
        with Image.open(image_path) as image:
            width, height = image.size
        gt[image_path.name] = [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ]
    return images, gt


def checkpoint(group: str, detector_seed: int, generator_variant: str | None = None) -> Path:
    if group == "baseline":
        suffix = "" if detector_seed == 42 else f"_s{detector_seed}"
        return RUNS_LF / f"yolov8n_baseline_lf{suffix}" / "weights" / "last.pt"
    actual_group = generator_variant or group
    return RUNS_Q / f"q300_{actual_group}_s{detector_seed}" / "weights" / "last.pt"


def generator_variants(group: str) -> list[str | None]:
    if group == "Diffusion_LDM":
        return ["Diffusion_LDM", "Diffusion_LDM_gs1", "Diffusion_LDM_gs2"]
    return [None]


def iou_one_to_many(gt_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=float)
    left_top = np.maximum(gt_box[:2], boxes[:, :2])
    right_bottom = np.minimum(gt_box[2:], boxes[:, 2:])
    wh = np.clip(right_bottom - left_top, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    gt_area = max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    pred_area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = gt_area + pred_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class ImageStatsValidator(DetectionValidator):
    """Capture the validator's exact per-image matching statistics."""

    captured: dict[str, dict] = {}

    def update_metrics(self, preds, batch) -> None:
        before = len(self.metrics.stats["tp"])
        super().update_metrics(preds, batch)
        for offset, image_path in enumerate(batch["im_file"]):
            index = before + offset
            self.captured[Path(image_path).name] = {
                "confidence": self.metrics.stats["conf"][index].astype(float).tolist(),
                "tp50": self.metrics.stats["tp"][index][:, 0].astype(int).tolist(),
                "n_targets": int(len(self.metrics.stats["target_cls"][index])),
            }


def collect_predictions(force: bool = False) -> dict:
    images, gt = load_images_and_gt()
    if PREDICTIONS_JSON.exists() and not force:
        return json.loads(PREDICTIONS_JSON.read_text(encoding="utf-8"))

    os.environ.setdefault("WANDB_DISABLED", "true")
    output = {
        "protocol": {
            "held_out_images": len(images),
            "confidence_floor": 0.001,
            "nms_iou": 0.70,
            "imgsz": 640,
            "checkpoint": "last.pt",
        },
        "image_order": [path.name for path in images],
        "models": {},
    }
    total = sum(len(generator_variants(group)) * len(SEEDS) for group in METHODS)
    completed = 0
    for group in METHODS:
        for variant in generator_variants(group):
            variant_key = variant or group
            for detector_seed in SEEDS:
                model_key = f"{variant_key}_s{detector_seed}"
                pt = checkpoint(group, detector_seed, variant)
                if not pt.exists():
                    raise FileNotFoundError(pt)
                model = YOLO(str(pt))
                ImageStatsValidator.captured = {}
                metrics = model.val(
                    validator=ImageStatsValidator,
                    data=str(VAL_YAML),
                    imgsz=640,
                    batch=8,
                    device=0,
                    workers=0,
                    plots=False,
                    verbose=False,
                )
                if set(ImageStatsValidator.captured) != {path.name for path in images}:
                    raise RuntimeError(f"Incomplete per-image statistics for {model_key}")
                output["models"][model_key] = ImageStatsValidator.captured
                output.setdefault("validator_map50", {})[model_key] = float(metrics.box.map50)
                PREDICTIONS_JSON.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
                completed += 1
                print(f"[PREDICT {completed}/{total}] {model_key}", flush=True)
                del model
    PREDICTIONS_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def ap50_for_sample(model_rows: dict, image_names: list[str], sampled_indices: np.ndarray) -> float:
    confidences = []
    true_positive = []
    for sampled_index in sampled_indices:
        row = model_rows[image_names[int(sampled_index)]]
        conf = np.asarray(row["confidence"], dtype=float)
        tp = np.asarray(row["tp50"], dtype=float)
        confidences.extend(conf.tolist())
        true_positive.extend(tp.tolist())
    if not confidences:
        return 0.0
    order = np.argsort(-np.asarray(confidences, dtype=float), kind="stable")
    tp = np.asarray(true_positive, dtype=float)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_targets = sum(model_rows[image_names[int(index)]]["n_targets"] for index in sampled_indices)
    recall = tp_cum / n_targets
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return float(compute_ap(recall, precision)[0])


def model_keys_for_method(group: str, detector_seed: int) -> list[str]:
    return [f"{variant or group}_s{detector_seed}" for variant in generator_variants(group)]


def aggregate_method_ap(predictions: dict, group: str, sampled_indices: np.ndarray) -> float:
    image_names = predictions["image_order"]
    detector_seed_means = []
    for detector_seed in SEEDS:
        generator_aps = [
            ap50_for_sample(predictions["models"][key], image_names, sampled_indices)
            for key in model_keys_for_method(group, detector_seed)
        ]
        detector_seed_means.append(float(np.mean(generator_aps)))
    return float(np.mean(detector_seed_means))


_WORKER_PREDICTIONS: dict | None = None


def init_bootstrap_worker(predictions: dict) -> None:
    global _WORKER_PREDICTIONS
    _WORKER_PREDICTIONS = predictions


def bootstrap_chunk(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    if _WORKER_PREDICTIONS is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    start, samples = task
    groups = [group for group in METHODS if group != "baseline"]
    deltas = np.empty((len(samples), len(groups)), dtype=float)
    for row_index, sampled in enumerate(samples):
        base = aggregate_method_ap(_WORKER_PREDICTIONS, "baseline", sampled)
        for column_index, group in enumerate(groups):
            deltas[row_index, column_index] = (
                aggregate_method_ap(_WORKER_PREDICTIONS, group, sampled) - base
            )
    return start, deltas


def summarize(predictions: dict, repeats: int, workers: int) -> dict:
    n_images = len(predictions["image_order"])
    full_sample = np.arange(n_images, dtype=int)
    point = {group: aggregate_method_ap(predictions, group, full_sample) for group in METHODS}

    validation_errors = []
    for group in METHODS:
        for detector_seed in SEEDS:
            for key in model_keys_for_method(group, detector_seed):
                recalculated = ap50_for_sample(
                    predictions["models"][key], predictions["image_order"], full_sample
                )
                validation_errors.append(abs(recalculated - predictions["validator_map50"][key]))
    max_validation_error = float(max(validation_errors))
    if max_validation_error > 5e-4:
        raise RuntimeError(
            f"Per-image AP reconstruction differs from validator by {max_validation_error:.6f}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n_images, size=(repeats, n_images), dtype=np.int16)
    groups = [group for group in METHODS if group != "baseline"]
    delta_matrix = np.empty((repeats, len(groups)), dtype=float)
    chunk_size = max(10, min(100, (repeats + max(workers, 1) * 20 - 1) // (max(workers, 1) * 20)))
    tasks = [
        (start, samples[start : start + chunk_size])
        for start in range(0, repeats, chunk_size)
    ]
    completed = 0
    if workers == 1:
        init_bootstrap_worker(predictions)
        iterator = map(bootstrap_chunk, tasks)
        for start, chunk_deltas in iterator:
            delta_matrix[start : start + len(chunk_deltas)] = chunk_deltas
            completed += len(chunk_deltas)
            print(f"[BOOTSTRAP {completed}/{repeats}]", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_bootstrap_worker,
            initargs=(predictions,),
        ) as executor:
            futures = [executor.submit(bootstrap_chunk, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                start, chunk_deltas = future.result()
                delta_matrix[start : start + len(chunk_deltas)] = chunk_deltas
                completed += len(chunk_deltas)
                print(f"[BOOTSTRAP {completed}/{repeats}]", flush=True)

    bootstrap_delta = {
        group: delta_matrix[:, column_index]
        for column_index, group in enumerate(groups)
    }

    methods = {}
    for group, label in METHODS.items():
        methods[group] = {"label": label, "mean_seed_ensemble_ap50": point[group]}
        if group != "baseline":
            delta = bootstrap_delta[group]
            methods[group].update(
                {
                    "delta_vs_baseline": point[group] - point["baseline"],
                    "image_bootstrap_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
                    "bootstrap_probability_delta_gt_0": float(np.mean(delta > 0)),
                }
            )
    output = {
        "design": {
            "inference_unit": "held-out image",
            "held_out_images": n_images,
            "bootstrap_repeats": repeats,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "workers": workers,
            "pairing": "same resampled images for every method and detector seed",
            "seed_handling": "average detector-training seeds within method before computing method contrast",
            "max_abs_ap50_reconstruction_error": max_validation_error,
            "interpretation": (
                "conditional image-sampling uncertainty for the fixed trained-model ensemble; "
                "detector seeds remain optimization-robustness repeats, not independent data replicates"
            ),
        },
        "methods": methods,
    }
    SUMMARY_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Image-level bootstrap for fixed-Q=300 detection",
        "",
        "Detector-training seeds are averaged within each method. The paired 95% confidence interval is obtained by resampling the 48 held-out images, not by treating detector seeds as independent data replicates.",
        "",
        "| Method | Ensemble AP50 | Delta vs Real-only | Image-bootstrap 95% CI | P(delta > 0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, label in METHODS.items():
        row = methods[group]
        if group == "baseline":
            lines.append(f"| {label} | {100 * row['mean_seed_ensemble_ap50']:.2f} | — | — | — |")
        else:
            lo, hi = row["image_bootstrap_ci95"]
            lines.append(
                f"| {label} | {100 * row['mean_seed_ensemble_ap50']:.2f} | "
                f"{100 * row['delta_vs_baseline']:+.2f} pp | "
                f"[{100 * lo:+.2f}, {100 * hi:+.2f}] | "
                f"{row['bootstrap_probability_delta_gt_0']:.3f} |"
            )
    lines += [
        "",
        "The bootstrap probability is descriptive and is not used as a frequentist p-value. The interval is conditional on the fixed split and the already trained model ensemble.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {SUMMARY_JSON}")
    print(f"[DONE] {OUTPUT_MD}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    predictions = collect_predictions(force=args.force_predict)
    if not args.predict_only:
        summarize(predictions, repeats=args.bootstrap_repeats, workers=max(1, args.workers))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
