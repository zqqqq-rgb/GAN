# -*- coding: utf-8 -*-

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, wilcoxon

ROOT = Path(str(_release_root))
DATA = ROOT / "data/yolo_dataset/q3_q_sensitivity_current.json"
BASE = ROOT / "data/yolo_dataset/results_leakfix.json"
OUT = ROOT / "data/yolo_dataset/q3_q_sensitivity_statistics.json"
FIG = ROOT / "deliverables/figures/q3_q_sensitivity_current"
REPORT = ROOT / "deliverables/q3_q_sensitivity_current.md"
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
QS = [100, 200, 300, 500]

d = json.loads(DATA.read_text())
b = json.loads(BASE.read_text())
baseline = np.array([b["lf_baseline" if s == 42 else f"lf_baseline_s{s}"]["mAP50"] for s in SEEDS])
result = {"protocol": {"q": QS, "seeds": SEEDS, "nested_prefix_design": True,
                       "selection_seed": 20260912, "baseline_pairing": "same detector seed"}, "q": {}}
for q in QS:
    values = np.array([d[f"q{q}_s{s}"]["mAP50"] for s in SEEDS])
    delta = values - baseline
    ci = t.interval(.95, len(delta)-1, loc=delta.mean(), scale=delta.std(ddof=1)/np.sqrt(len(delta)))
    result["q"][str(q)] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "delta_mean": float(delta.mean()), "delta_sd": float(delta.std(ddof=1)),
        "delta_95ci": [float(ci[0]), float(ci[1])], "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "values": values.tolist()}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(figsize=(3.5, 2.7))
rng = np.random.default_rng(20260912)
means, sds = [], []
for i, q in enumerate(QS):
    vals = np.array(result["q"][str(q)]["values"])
    means.append(vals.mean()); sds.append(vals.std(ddof=1))
    ax.scatter(np.full(len(vals), q) + rng.normal(0, 5, len(vals)), vals, s=12,
               color="#0072B2", alpha=.45, linewidths=0)
ax.errorbar(QS, means, yerr=sds, color="#D55E00", marker="o", linestyle="-",
            capsize=3, linewidth=1.2, label="Mean ± SD (n=10)")
ax.axhline(baseline.mean(), color="#333333", linestyle="--", linewidth=1,
           label=f"Baseline mean ({baseline.mean():.3f})")
ax.set_xlabel("Accepted synthetic images (Q)"); ax.set_ylabel("mAP@0.5")
ax.set_xticks(QS); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7); fig.tight_layout()
fig.savefig(str(FIG)+".png", dpi=600, bbox_inches="tight")
fig.savefig(str(FIG)+".pdf", bbox_inches="tight")
plt.close(fig)

lines=["# Nested Q sensitivity (current protocol)","", "| Q | mAP@0.5 | Paired delta vs baseline (95% CI) | Wilcoxon p |","|---:|---:|---:|---:|"]
for q in QS:
    r=result["q"][str(q)]; lines.append(f"| {q} | {r['mean']:.4f} ± {r['sd']:.4f} | {r['delta_mean']:+.4f} [{r['delta_95ci'][0]:+.4f}, {r['delta_95ci'][1]:+.4f}] | {r['wilcoxon_p']:.4f} |")
REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
print(REPORT)
