"""Audit the 241-image cable annotation set without changing any labels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = ROOT / "data" / "yolo_dataset"
SPLIT_PATH = YOLO_ROOT / "splits_leakfix" / "split_info.json"
IMAGE_DIR = YOLO_ROOT / "images" / "all_real"
LABEL_DIR = YOLO_ROOT / "labels" / "all_real"
SOURCE_PATH = ROOT / "implementations" / "yolov8_det" / "scripts" / "build_dataset.py"
OUT_JSON = YOLO_ROOT / "q3_real_annotation_audit.json"
OUT_MD = ROOT / "deliverables" / "q3_real_annotation_audit.md"


def load_annotation_module():
    spec = importlib.util.spec_from_file_location("cable_build_dataset", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def parse_label(text: str) -> tuple[list[list[float]], list[str]]:
    rows: list[list[float]] = []
    errors: list[str] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_no}: expected 5 fields")
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            errors.append(f"line {line_no}: nonnumeric field")
            continue
        cls, xc, yc, bw, bh = row
        if cls != 0:
            errors.append(f"line {line_no}: class is not 0")
        if not all(math.isfinite(x) for x in row):
            errors.append(f"line {line_no}: nonfinite value")
        if bw <= 0 or bh <= 0:
            errors.append(f"line {line_no}: nonpositive size")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            errors.append(f"line {line_no}: normalized value outside range")
        if xc - bw / 2 < -1e-6 or xc + bw / 2 > 1 + 1e-6:
            errors.append(f"line {line_no}: horizontal boundary overflow")
        if yc - bh / 2 < -1e-6 or yc + bh / 2 > 1 + 1e-6:
            errors.append(f"line {line_no}: vertical boundary overflow")
        rows.append(row)
    return rows, errors


def main() -> None:
    split_info = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    samples: list[dict[str, str]] = []
    for split in ("train", "val"):
        for defect_type, names in split_info[split].items():
            samples.extend(
                {"split": split, "defect_type": defect_type, "name": name}
                for name in names
            )

    annotation_module = load_annotation_module()
    original_fallback = annotation_module._fallback_bbox
    fallback_called = False

    def tracked_fallback(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return original_fallback(*args, **kwargs)

    annotation_module._fallback_bbox = tracked_fallback

    records = []
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_files = []
    missing_images = []
    missing_labels = []
    recompute_mismatches = []
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    for sample in samples:
        image_path = IMAGE_DIR / sample["name"]
        label_path = LABEL_DIR / f"{Path(sample['name']).stem}.txt"
        if not image_path.exists():
            missing_images.append(sample)
            continue
        digest = sha256(image_path)
        hashes[digest].append(sample)
        label_exists = label_path.exists()
        if not label_exists:
            missing_labels.append(sample)
            existing = ""
        else:
            existing = label_path.read_text(encoding="utf-8")
        rows, errors = parse_label(existing)
        if not rows:
            errors.append("no valid boxes")
        if errors:
            invalid_files.append({**sample, "errors": errors})
        for _, _, _, bw, bh in rows:
            widths.append(bw)
            heights.append(bh)
            areas.append(bw * bh)

        fallback_called = False
        recomputed = annotation_module.annotate_image(image_path, debug=False)
        used_fallback = fallback_called
        exact_match = existing.strip() == recomputed.strip()
        if not exact_match:
            recompute_mismatches.append(sample)
        records.append(
            {
                **sample,
                "sha256": digest,
                "label_exists": label_exists,
                "box_count": len(rows),
                "valid": not errors,
                "used_explicit_fallback": used_fallback,
                "recomputed_nonempty": bool(recomputed.strip()),
                "exact_recompute_match": exact_match,
            }
        )

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    train_hashes = {r["sha256"] for r in records if r["split"] == "train"}
    val_hashes = {r["sha256"] for r in records if r["split"] == "val"}
    fallback_by_split = Counter(r["split"] for r in records if r["used_explicit_fallback"])
    fallback_by_type = Counter(r["defect_type"] for r in records if r["used_explicit_fallback"])
    totals_by_split = Counter(r["split"] for r in records)
    totals_by_type = Counter(r["defect_type"] for r in records)

    result = {
        "protocol": {
            "split_manifest": str(SPLIT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "annotation_source": str(SOURCE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "scope": "241 unique images in fixed 193/48 split",
            "non_mutating": True,
            "manual_review": "48 held-out images reviewed by one reviewer; no label edits reported",
        },
        "counts": {
            "expected": len(samples),
            "audited": len(records),
            "train": totals_by_split.get("train", 0),
            "val": totals_by_split.get("val", 0),
            "missing_images": len(missing_images),
            "missing_labels": len(missing_labels),
            "invalid_label_files": len(invalid_files),
            "empty_recomputed_labels": sum(not r["recomputed_nonempty"] for r in records),
            "exact_recompute_matches": sum(r["exact_recompute_match"] for r in records),
            "recompute_mismatches": len(recompute_mismatches),
            "explicit_fallback": sum(r["used_explicit_fallback"] for r in records),
            "duplicate_hash_groups_within_241": len(duplicate_groups),
            "train_val_hash_overlap": len(train_hashes & val_hashes),
        },
        "fallback": {
            "overall_rate": sum(r["used_explicit_fallback"] for r in records) / len(records),
            "by_split": {
                split: {
                    "count": fallback_by_split.get(split, 0),
                    "total": totals_by_split[split],
                    "rate": fallback_by_split.get(split, 0) / totals_by_split[split],
                }
                for split in sorted(totals_by_split)
            },
            "by_defect_type": {
                defect_type: {
                    "count": fallback_by_type.get(defect_type, 0),
                    "total": totals_by_type[defect_type],
                    "rate": fallback_by_type.get(defect_type, 0) / totals_by_type[defect_type],
                }
                for defect_type in sorted(totals_by_type)
            },
        },
        "bbox_normalized": {
            "width": describe(widths),
            "height": describe(heights),
            "area": describe(areas),
            "total_boxes": len(areas),
        },
        "issues": {
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "invalid_label_files": invalid_files,
            "recompute_mismatches": recompute_mismatches,
            "duplicate_hash_groups": duplicate_groups,
        },
        "records": records,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    c = result["counts"]
    pct = 100 * result["fallback"]["overall_rate"]
    md = [
        "# 真实图像标注可计算审计",
        "",
        "本审计不修改标签，也不替代人工或第二复核者；它只验证固定规则、文件完整性和划分隔离。",
        "",
        f"- 固定划分：{c['train']} 张训练图、{c['val']} 张留出图，共 {c['audited']} 张。",
        f"- 标签完整性：缺失 {c['missing_labels']}，格式/越界异常 {c['invalid_label_files']}，空重算结果 {c['empty_recomputed_labels']}。",
        f"- 规则重算一致性：{c['exact_recompute_matches']}/{c['audited']} 完全一致；不一致 {c['recompute_mismatches']}。",
        f"- 显式兜底框：{c['explicit_fallback']}/{c['audited']}（{pct:.2f}%）。",
        f"- 哈希隔离：241 张内重复哈希组 {c['duplicate_hash_groups_within_241']}；训练/留出哈希交集 {c['train_val_hash_overlap']}。",
        f"- 框数：{result['bbox_normalized']['total_boxes']}；归一化面积中位数 {result['bbox_normalized']['area']['median']:.4f}。",
        "",
        "## 显式兜底比例（按划分）",
        "",
    ]
    for split, row in result["fallback"]["by_split"].items():
        md.append(f"- {split}: {row['count']}/{row['total']}（{100*row['rate']:.2f}%）")
    md.extend(["", "## 显式兜底比例（按缺陷类型）", ""])
    for defect_type, row in result["fallback"]["by_defect_type"].items():
        md.append(f"- {defect_type}: {row['count']}/{row['total']}（{100*row['rate']:.2f}%）")
    md.extend(
        [
            "",
            "## 解释边界",
            "",
            "`explicit_fallback` 仅统计标注函数实际调用 `_fallback_bbox` 的样本；它不评价非兜底框的语义正确性。留出集的一人复核结论与本计算审计应分开报告。",
        ]
    )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), **c}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
