"""Deterministic qualitative panels for GAN-seed robustness and failure analysis."""
from __future__ import annotations

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[0]


import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(str(_release_root))
GEN = ROOT / "data/yolo_dataset/images/leakfix"
OUT = ROOT / "deliverables/figures"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["dent", "scratch", "deformation", "bulge"]
SEEDS = [42, 1, 2]
SOURCES = {
    42: GEN / "gen_E7proj_mses_sp_leak_5k",
    1: GEN / "gen_e7proj_mses_sp_leak_s1_5k",
    2: GEN / "gen_e7proj_mses_sp_leak_s2_5k",
}
PRED = ROOT / "data/yolo_dataset/pseudo_predictions_leakfix.json"
PRED_GROUPS = {1: "E7proj_mses_sp_s1", 2: "E7proj_mses_sp_s2"}


def stable_pick(paths: list[Path], key: str) -> Path:
    """Pick the smallest SHA-256 rank; independent of Python hash randomization."""
    return min(paths, key=lambda p: hashlib.sha256(f"{key}|{p.name}".encode()).hexdigest())


def open_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def seed_panel() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.3))
    for row, seed in enumerate(SEEDS):
        for col, cls in enumerate(CLASSES):
            candidates = sorted((SOURCES[seed] / cls).glob("*.png"))
            pick = stable_pick(candidates, f"seed-panel|{seed}|{cls}")
            axes[row, col].imshow(open_gray(pick), cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(cls.replace("deformation", "deform."), fontsize=8)
            if col == 0: axes[row, col].set_ylabel(f"GAN seed {seed}", fontsize=8)
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(0.5); spine.set_color("#666666")
    fig.text(0.01, 0.98, "a", weight="bold", fontsize=10, va="top")
    fig.tight_layout(pad=0.5)
    save(fig, "q3_fixed_gan_seed_examples")


def failure_panel() -> None:
    predictions = json.loads(PRED.read_text(encoding="utf-8"))
    cells = []
    for seed in [1, 2]:
        group = PRED_GROUPS[seed]
        group_pred = predictions[group]
        flat = GEN / f"gen_e7proj_mses_sp_leak_s{seed}_5k_flat"
        for cls in CLASSES:
            names = sorted(name for name in group_pred if name.startswith(cls + "_"))
            no_box = [name for name in names if not any(row[0] >= 0.10 for row in group_pred[name])]
            if no_box:
                name = stable_pick([Path(x) for x in no_box], f"failure|{seed}|{cls}").name
                status = "No box ≥0.10"
            else:
                name = min(names, key=lambda x: max((r[0] for r in group_pred[x]), default=-1))
                status = "Lowest confidence"
            cells.append((seed, cls, flat / name, status))
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7))
    for ax, (seed, cls, path, status) in zip(axes.flat, cells):
        ax.imshow(open_gray(path), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{cls.replace('deformation', 'deform.')}\n{status}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if cls == CLASSES[0]: ax.set_ylabel(f"GAN seed {seed}", fontsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color("#666666")
    fig.text(0.01, 0.98, "b", weight="bold", fontsize=10, va="top")
    fig.tight_layout(pad=0.5)
    save(fig, "q3_fixed_failure_cases")


if __name__ == "__main__":
    seed_panel()
    failure_panel()
    protocol = {
        "example_selection": "minimum SHA-256 rank of 'seed-panel|GAN seed|class|filename'",
        "failure_definition": "teacher detector has no box at confidence >=0.10; deterministic SHA-256 selection within each GAN-seed/class stratum",
        "display": "grayscale, no contrast manipulation, 600 dpi PNG and TIFF",
    }
    (OUT / "q3_fixed_qualitative_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(OUT)
