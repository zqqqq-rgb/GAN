from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data/yolo_dataset/q3_fixedq_statistics.json", encoding="utf-8") as f:
    stats = json.load(f)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

colors = {
    "baseline": "#666666",
    "class_no": "#56B4E9",
    "class_yes": "#0072B2",
    "text_no": "#E69F00",
    "text_yes": "#D55E00",
    "copy": "#009E73",
}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

# A: paired detector-seed changes against real-only.
ax = axes[0]
base = np.array(stats["per_group"]["baseline"]["mAP50"]["values"]) * 100
groups = [
    ("Class\nno MSES", "E6proj_mses_nosp", colors["class_no"]),
    ("Class\n+MSES", "E6proj_mses_sp", colors["class_yes"]),
    ("Text\nno MSES", "E7proj_mses_nosp", colors["text_no"]),
    ("Text\n+MSES", "E7proj_mses_sp", colors["text_yes"]),
]
for x, (label, key, color) in enumerate(groups):
    vals = np.array(stats["per_group"][key]["mAP50"]["values"]) * 100
    for b, v in zip(base, vals):
        ax.plot([x - 0.16, x + 0.16], [b, v], color="#BDBDBD", lw=0.6, alpha=0.7, zorder=1)
    ax.scatter(np.full(10, x - 0.16), base, s=11, facecolor="white", edgecolor=colors["baseline"], lw=0.7, zorder=2)
    ax.scatter(np.full(10, x + 0.16), vals, s=12, color=color, edgecolor="white", lw=0.35, zorder=3)
ax.axhline(base.mean(), color=colors["baseline"], lw=0.8, ls="--", alpha=0.8)
ax.set_xticks(range(4), [g[0] for g in groups])
ax.set_ylabel("mAP@0.5 (%)")
ax.set_title("Paired detector seeds (n=10)")
ax.grid(axis="y", color="#E5E5E5", lw=0.5)

# B: complete 2 x 2 factorial means and SD.
ax = axes[1]
x = np.array([0, 1])
for cond, keys, color, marker in [
    ("Class projection", ["E6proj_mses_nosp", "E6proj_mses_sp"], colors["class_yes"], "s"),
    ("Text projection", ["E7proj_mses_nosp", "E7proj_mses_sp"], colors["text_yes"], "o"),
]:
    means = [stats["per_group"][k]["mAP50"]["mean"] * 100 for k in keys]
    sds = [stats["per_group"][k]["mAP50"]["sd"] * 100 for k in keys]
    ax.errorbar(x, means, yerr=sds, label=cond, color=color, marker=marker,
                lw=1.2, ms=4, capsize=3)
ax.set_xticks(x, ["Without MSES", "With MSES"])
ax.set_ylabel("mAP@0.5 (%)")
ax.set_title("Condition × MSES ablation")
ax.legend(frameon=False, loc="lower left")
ax.grid(axis="y", color="#E5E5E5", lw=0.5)

# C: pseudo-label acceptance under the common protocol.
ax = axes[2]
labels = ["Class\nno MSES", "Class\n+MSES", "Text\nno MSES", "Text\n+MSES"]
# Strict-teacher acceptance from the full 5000-image candidate pools
# (q3_fixedq_manifest.json accepted_pool; DCGAN / WGAN-GP+FM / DiffAugment
# fell to 3 / 142 / 98 accepted and are excluded from the fixed-Q family).
accepted = np.array([1408, 2037, 2339, 1272])
candidates = 5000
bar_colors = [colors["class_no"], colors["class_yes"], colors["text_no"], colors["text_yes"]]
bars = ax.bar(np.arange(4), accepted / candidates * 100, color=bar_colors, width=0.68)
ax.axhline(300 / candidates * 100, color="#333333", ls="--", lw=0.8, label="Q=300")
for bar, n in zip(bars, accepted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{n}/{candidates}",
            ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(np.arange(4), labels)
ax.set_ylim(0, 60)
ax.set_ylabel("Accepted candidates (%)")
ax.set_title("Pseudo-label acceptance")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color="#E5E5E5", lw=0.5)

for label, ax in zip("ABC", axes):
    ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

fig.tight_layout(w_pad=1.2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure3_quantitative_evidence.{ext}", dpi=600 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
