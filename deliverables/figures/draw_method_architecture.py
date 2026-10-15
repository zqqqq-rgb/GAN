"""Draw a publication-ready, code-faithful architecture schematic."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
COLORS = {
    "blue": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light": "#F8FAFC",
    "ink": "#17202A",
}


def box(ax, xy, wh, text, edge, face="#FFFFFF", fontsize=9, lw=1.5, style="round,pad=0.012"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                           edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["ink"], linespacing=1.25, zorder=3)
    return patch


def arrow(ax, start, end, color=None, lw=1.6, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                            linewidth=lw, color=color or COLORS["gray"],
                            connectionstyle=connection, zorder=1)
    ax.add_patch(patch)
    return patch


def dashed_container(ax, xy, wh, title, edge):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                           linewidth=1.3, linestyle=(0, (4, 3)),
                           edgecolor=edge, facecolor="none", zorder=0)
    ax.add_patch(patch)
    ax.text(x + 0.015, y + h - 0.028, title, ha="left", va="top",
            fontsize=10, fontweight="bold", color=edge)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.965, "Text-projection WGAN-GP with unpaired multi-scale edge-statistics matching",
            ha="center", va="center", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.927, "Training on a condition-matched but spatially unpaired real/fake batch",
            ha="center", va="center", fontsize=9.5, color=COLORS["gray"])

    # Condition and generator path.
    dashed_container(ax, (0.025, 0.57), (0.47, 0.31), "Conditioned generation", COLORS["blue"])
    box(ax, (0.045, 0.735), (0.14, 0.09), "Four fixed defect\ndescriptions  c", COLORS["purple"], "#F9ECF4")
    box(ax, (0.215, 0.735), (0.13, 0.09), "Frozen CLIP\nViT-B/32", COLORS["purple"], "#F9ECF4")
    box(ax, (0.375, 0.735), (0.095, 0.09), "$e(c)\\in\\mathbb{R}^{512}$", COLORS["purple"], "#F9ECF4")
    arrow(ax, (0.185, 0.78), (0.215, 0.78), COLORS["purple"])
    arrow(ax, (0.345, 0.78), (0.375, 0.78), COLORS["purple"])

    box(ax, (0.05, 0.615), (0.115, 0.075), "$z\\sim\\mathcal{N}(0,I)$\n128-D", COLORS["orange"], "#FFF5DF")
    box(ax, (0.215, 0.595), (0.145, 0.115), "Generator  $G$\nconcat $[z;e(c)]$\n$8\\times8\\rightarrow512\\times512$", COLORS["blue"], "#EAF6FC")
    box(ax, (0.395, 0.615), (0.075, 0.075), "$\\tilde{x}$\nRGB", COLORS["blue"], "#EAF6FC")
    arrow(ax, (0.165, 0.652), (0.215, 0.652), COLORS["orange"])
    arrow(ax, (0.422, 0.735), (0.33, 0.71), COLORS["purple"], connection="arc3,rad=-0.16")
    arrow(ax, (0.36, 0.652), (0.395, 0.652), COLORS["blue"])

    # Discriminator path.
    dashed_container(ax, (0.52, 0.57), (0.455, 0.31), "True image-condition projection critic", COLORS["green"])
    box(ax, (0.54, 0.745), (0.095, 0.075), "Real $x$ or\nfake $\\tilde{x}$", COLORS["gray"], "#F3F4F6")
    box(ax, (0.665, 0.71), (0.13, 0.145), "Shared CNN $f$\n6 spectral-norm\nconv blocks\n$512\\times8\\times8$", COLORS["green"], "#E9F7F2")
    box(ax, (0.82, 0.785), (0.14, 0.06), "$u(P_{4\\times4}(f))$  realism score", COLORS["green"], "#E9F7F2", fontsize=8.2)
    box(ax, (0.82, 0.70), (0.14, 0.06), "$h=GAP(f),\\ h\\in\\mathbb{R}^{512}$", COLORS["green"], "#E9F7F2", fontsize=8.2)
    arrow(ax, (0.635, 0.782), (0.665, 0.782), COLORS["gray"])
    arrow(ax, (0.795, 0.795), (0.82, 0.815), COLORS["green"])
    arrow(ax, (0.795, 0.755), (0.82, 0.73), COLORS["green"])
    arrow(ax, (0.47, 0.652), (0.54, 0.765), COLORS["blue"], connection="arc3,rad=-0.18")

    box(ax, (0.54, 0.61), (0.12, 0.075), "$V\\bar e(c)$\nprojected text", COLORS["purple"], "#F9ECF4")
    box(ax, (0.695, 0.61), (0.115, 0.075), "$\\langle h,V\\bar e(c)\\rangle$\n$/\\sqrt{512}$", COLORS["purple"], "#F9ECF4", fontsize=8.5)
    box(ax, (0.855, 0.59), (0.105, 0.075), "$D(x,c)=$\nrealism + projection", COLORS["green"], "#E9F7F2", fontsize=8.2)
    arrow(ax, (0.422, 0.735), (0.60, 0.685), COLORS["purple"], connection="arc3,rad=0.10")
    arrow(ax, (0.66, 0.648), (0.695, 0.648), COLORS["purple"])
    arrow(ax, (0.89, 0.70), (0.80, 0.67), COLORS["green"], connection="arc3,rad=-0.12")
    arrow(ax, (0.81, 0.648), (0.855, 0.63), COLORS["purple"])
    arrow(ax, (0.93, 0.785), (0.93, 0.665), COLORS["green"], connection="arc3,rad=0.22")

    # Loss panel.
    dashed_container(ax, (0.025, 0.13), (0.95, 0.37), "Training-only objectives", COLORS["orange"])
    box(ax, (0.05, 0.31), (0.25, 0.12),
        "WGAN-GP\n$\\mathcal{L}_{adv}+\\lambda_{gp}\\mathcal{L}_{gp}$\n$\\lambda_{gp}=50$",
        COLORS["orange"], "#FFF5DF")
    box(ax, (0.375, 0.31), (0.25, 0.12),
        "Batch feature matching\n$\\|\\mathbb{E}[\\phi(\\tilde{x})]-\\mathbb{E}[\\phi(x)]\\|_2^2$\n$\\lambda_{fm}=50$",
        COLORS["blue"], "#EAF6FC", fontsize=8.5)
    box(ax, (0.70, 0.31), (0.25, 0.12),
        "Unpaired MSES\nSobel $(m,|g_x|,|g_y|)$ at 32/16/8\nmatch batch mean + 0.5 SD",
        COLORS["purple"], "#F9ECF4", fontsize=8.5)

    box(ax, (0.235, 0.175), (0.53, 0.075),
        "$\\mathcal{L}_G=\\mathcal{L}_{adv}+50\\mathcal{L}_{fm}+5\\mathcal{L}_{MSES}$"
        "    (MSES disabled only in the no-MSES ablation)",
        COLORS["ink"], "#F8FAFC", fontsize=10, lw=1.7)
    arrow(ax, (0.175, 0.31), (0.34, 0.25), COLORS["orange"], connection="arc3,rad=0.12")
    arrow(ax, (0.50, 0.31), (0.50, 0.25), COLORS["blue"])
    arrow(ax, (0.825, 0.31), (0.66, 0.25), COLORS["purple"], connection="arc3,rad=-0.12")

    ax.text(0.5, 0.075,
            "MSES compares distributions, not arbitrary sample pairs; MSES and the critic are training-only, while fixed CLIP embeddings are cached.",
            ha="center", va="center", fontsize=9.5, color=COLORS["gray"])
    ax.text(0.5, 0.04,
            "Solid arrows: forward/data flow     Dashed containers: functional modules",
            ha="center", va="center", fontsize=8, color=COLORS["gray"])

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    fig.savefig(OUT_DIR / "figure1_method_architecture.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure1_method_architecture.png", dpi=300, bbox_inches="tight")
    print(OUT_DIR / "figure1_method_architecture.svg")
    print(OUT_DIR / "figure1_method_architecture.png")


if __name__ == "__main__":
    main()
