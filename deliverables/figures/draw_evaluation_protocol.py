"""Draw the leakage-controlled generation-to-detection evaluation protocol."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
INK = "#17202A"
GRAY = "#6B7280"
BLUE = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"


def box(ax, x, y, w, h, text, edge, face, fontsize=9.2, lw=1.6):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                           linewidth=lw, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.25, zorder=3)


def arrow(ax, start, end, color=GRAY, connection="arc3", lw=1.7):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=lw, color=color, connectionstyle=connection, zorder=1))


def elbow(ax, points, color=GRAY, lw=1.7):
    ax.add_line(Line2D([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                       color=color, lw=lw, zorder=1, solid_capstyle="round"))
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.955, "Leakage-controlled generation-to-detection evaluation",
            ha="center", va="center", fontsize=15, fontweight="bold", color=INK)

    box(ax, 0.035, 0.78, 0.14, 0.10, "245 acquired images", GRAY, "#F3F4F6")
    box(ax, 0.215, 0.78, 0.16, 0.10, "Hash deduplication\nremove 4 repeated frames", ORANGE, "#FFF5DF")
    box(ax, 0.415, 0.78, 0.14, 0.10, "241 unique images", GRAY, "#F3F4F6")
    box(ax, 0.595, 0.78, 0.17, 0.10, "Fixed stratified split\nseed = 42", BLUE, "#EAF6FC")
    arrow(ax, (0.175, 0.83), (0.215, 0.83))
    arrow(ax, (0.375, 0.83), (0.415, 0.83))
    arrow(ax, (0.555, 0.83), (0.595, 0.83))

    box(ax, 0.16, 0.57, 0.18, 0.11, "Training partition\n193 images", BLUE, "#EAF6FC", fontsize=10)
    box(ax, 0.67, 0.57, 0.18, 0.11, "Held-out test partition\n48 images", RED, "#FDEEE8", fontsize=10)
    arrow(ax, (0.68, 0.78), (0.30, 0.68), BLUE, "arc3,rad=0.12")
    arrow(ax, (0.68, 0.78), (0.76, 0.68), RED, "arc3,rad=-0.08")

    ax.plot([0.57, 0.57], [0.14, 0.72], color=RED, linewidth=2.2, linestyle=(0, (7, 4)))
    ax.text(0.58, 0.48, "NO ACCESS during model training or sample selection",
            rotation=90, ha="left", va="center", fontsize=9, color=RED, fontweight="bold")

    box(ax, 0.035, 0.36, 0.15, 0.10, "Train GAN / baseline\nmodels", PURPLE, "#F9ECF4")
    box(ax, 0.215, 0.36, 0.15, 0.10, "Generate balanced\ncandidate pool", PURPLE, "#F9ECF4")
    box(ax, 0.395, 0.36, 0.13, 0.10, "Teacher YOLOv8n\npseudo boxes", GREEN, "#E9F7F2")
    arrow(ax, (0.25, 0.57), (0.11, 0.46), BLUE, "arc3,rad=0.1")
    arrow(ax, (0.185, 0.41), (0.215, 0.41), PURPLE)
    arrow(ax, (0.365, 0.41), (0.395, 0.41), GREEN)
    arrow(ax, (0.25, 0.57), (0.46, 0.46), GREEN, "arc3,rad=-0.08")

    box(ax, 0.10, 0.18, 0.18, 0.10, "Accept boxes at conf ≥ 0.10\ndiscard images without boxes", GREEN, "#E9F7F2", fontsize=8.8)
    box(ax, 0.32, 0.18, 0.19, 0.10, "Fixed-Q training set\n193 real + 300 synthetic", BLUE, "#EAF6FC", fontsize=9.2)
    arrow(ax, (0.46, 0.36), (0.24, 0.28), GREEN, "arc3,rad=0.12")
    arrow(ax, (0.28, 0.23), (0.32, 0.23), BLUE)

    box(ax, 0.64, 0.34, 0.20, 0.12,
        "YOLOv8n training\n10 seeds: 42, 1, …, 9\nsame split for every method",
        BLUE, "#EAF6FC", fontsize=9.5)
    box(ax, 0.64, 0.16, 0.20, 0.10,
        "Final evaluation only\nmAP@0.5, mAP@0.5:0.95, P, R",
        RED, "#FDEEE8", fontsize=9.2)
    arrow(ax, (0.51, 0.23), (0.64, 0.38), BLUE, "arc3,rad=-0.12")
    arrow(ax, (0.74, 0.34), (0.74, 0.26), RED)
    # Test partition bypasses detector training entirely and feeds final evaluation only.
    elbow(ax, [(0.85, 0.625), (0.87, 0.625), (0.87, 0.24), (0.84, 0.24)], RED, 1.7)

    box(ax, 0.90, 0.22, 0.075, 0.18,
        "Paired by seed\nWilcoxon\nHolm correction\nBootstrap CI",
        ORANGE, "#FFF5DF", fontsize=8.2)
    arrow(ax, (0.84, 0.175), (0.90, 0.225), ORANGE)

    ax.text(0.28, 0.085,
            "CopyPaste: exact transformed boxes.\nPseudo-label groups: report acceptance rate and class composition.",
            ha="center", va="center", fontsize=7.8, color=GRAY)
    ax.text(0.76, 0.075,
            "Ten runs quantify detector-optimization variability,\nnot independent data-split variability.",
            ha="center", va="center", fontsize=7.8, color=GRAY)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    fig.savefig(OUT_DIR / "figure2_evaluation_protocol.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure2_evaluation_protocol.png", dpi=300, bbox_inches="tight")
    print(OUT_DIR / "figure2_evaluation_protocol.svg")
    print(OUT_DIR / "figure2_evaluation_protocol.png")


if __name__ == "__main__":
    main()
