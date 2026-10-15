"""Recreate the three reference panels from the original experiment data."""
from pathlib import Path
import csv
import json
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
STATS_PATH = ROOT / "data/yolo_dataset/q3_fixedq_statistics.json"
MANIFEST_PATH = ROOT / "data/yolo_dataset/q3_fixedq_manifest.json"
stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
FONT_PATH = Path("C:/Windows/Fonts/times.ttf")
for filename in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(str(FONT_PATH.with_name(filename)))
assert font_manager.FontProperties(fname=str(FONT_PATH)).get_name() == "Times New Roman"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10.5,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.2,
    "axes.edgecolor": "#262626", "axes.labelcolor": "#262626", "text.color": "#202020",
    "xtick.color": "#262626", "ytick.color": "#262626", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.unicode_minus": False,
})
GROUPS = [
    ("Class\nno MSES", "E6proj_mses_nosp", "#56B4E9"),
    ("Class\n+MSES", "E6proj_mses_sp", "#0072B2"),
    ("Text\nno MSES", "E7proj_mses_nosp", "#E69F00"),
    ("Text\n+MSES", "E7proj_mses_sp", "#D55E00"),
]
FILENAMES = ["A_paired_detector_seeds", "B_condition_MSES_ablation", "C_pseudo_label_acceptance"]
baseline = np.asarray(stats["per_group"]["baseline"]["mAP50"]["values"]) * 100
n = len(baseline)
candidates = 5000
q = manifest["q"]
accepted = np.asarray([manifest["groups"][key]["accepted_pool"] for _, key, _ in GROUPS])
assert accepted.tolist() == [1408, 2037, 2339, 1272]
assert n == 10

def figure_frame(title):
    fig = plt.figure(figsize=(4.8, 4.2), dpi=160)
    ax = fig.add_axes([0.175, 0.245, 0.78, 0.595])
    fig.text(0.555, 0.942, title, fontsize=13, ha="center", va="center")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    ax.tick_params(axis="x", pad=7)
    ax.tick_params(axis="y", pad=5)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    return fig, ax

def swarm_x(ax, center, values, marker_area=21):
    """Separate coincident markers horizontally without changing their values."""
    distance = np.sqrt(marker_area) * ax.figure.dpi / 72 * 1.08
    positions = np.zeros(len(values))
    occupied = []
    center_px = ax.transData.transform((center, 0))[0]
    for idx in np.argsort(values, kind="stable"):
        y_px = ax.transData.transform((center, values[idx]))[1]
        found = False
        for step in range(8):
            for sign in ([1] if step == 0 else [1, -1]):
                x_px = center_px + sign * step * distance
                if all(np.hypot(x_px - ox, y_px - oy) >= distance for ox, oy in occupied):
                    positions[idx] = ax.transData.inverted().transform((x_px, y_px))[0]
                    occupied.append((x_px, y_px))
                    found = True
                    break
            if found:
                break
        assert found
    return positions

def validate_layout(fig):
    """Check all figure text for clipping, collisions, and the requested font."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        texts += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
    for legend in fig.legends:
        texts += list(legend.get_texts())
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    canvas = fig.bbox
    boxes = []
    for text in texts:
        assert text.get_fontproperties().get_name() == "Times New Roman", text.get_text()
        bbox = text.get_window_extent(renderer)
        assert bbox.x0 >= canvas.x0 and bbox.x1 <= canvas.x1, f"Horizontal clipping: {text.get_text()}"
        assert bbox.y0 >= canvas.y0 and bbox.y1 <= canvas.y1, f"Vertical clipping: {text.get_text()}"
        boxes.append((text.get_text(), bbox))
    for i, (label, bbox) in enumerate(boxes):
        for other_label, other_bbox in boxes[i + 1:]:
            assert not bbox.overlaps(other_bbox), f"Text overlap: {label!r} / {other_label!r}"
    return len(texts)

layout_reports = []
def export(fig, name):
    text_count = validate_layout(fig)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=600, facecolor="white")
    preview_dir = OUT / "preview"
    preview_dir.mkdir(exist_ok=True)
    fig.savefig(preview_dir / f"{name}.png", dpi=160)
    layout_reports.append({"figure": name, "verified_text_items": text_count,
        "text_overlap": False, "text_clipping": False, "font": "Times New Roman",
        "png_dpi": 600, "png_pixels": [2880, 2520]})
    plt.close(fig)

# A: All 10 paired observations and the original real-only reference mean.
fig, ax = figure_frame("Paired detector seeds (n = 10)")
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(43, 66)
ax.set_yticks(np.arange(45, 65.1, 5))
ax.set_ylabel("mAP@0.5 (%)")
ax.set_xticks(range(4), [label for label, _, _ in GROUPS])
for x, (_, key, color) in enumerate(GROUPS):
    values = np.asarray(stats["per_group"][key]["mAP50"]["values"]) * 100
    assert len(values) == n
    bx = swarm_x(ax, x - 0.185, baseline)
    vx = swarm_x(ax, x + 0.185, values)
    for bxi, vxi, b, v in zip(bx, vx, baseline, values):
        ax.plot([bxi, vxi], [b, v], color="#B7BCC4", linewidth=0.65, alpha=0.8, zorder=1)
    ax.scatter(bx, baseline, s=21, facecolor="white", edgecolor="#6B6B6B", linewidth=0.85, zorder=3)
    ax.scatter(vx, values, s=23, facecolor=color, edgecolor="white", linewidth=0.5, zorder=4)
ax.axhline(baseline.mean(), color="#727272", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
fig.legend(handles=[
    Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="#6B6B6B",
           markersize=4.8, label="Real only"),
    Line2D([], [], color="#727272", linestyle=(0, (4, 3)), linewidth=1.0, label="Real-only mean"),
], loc="lower center", bbox_to_anchor=(0.55, 0.055), ncol=2, frameon=False, columnspacing=1.5, handlelength=2.1)
export(fig, FILENAMES[0])

# B: Original means and sample SD; slightly separate overlapping error bars.
fig, ax = figure_frame("Condition × MSES ablation")
ax.set_xlim(-0.23, 1.23)
ax.set_ylim(51.5, 64.1)
ax.set_yticks(np.arange(52, 64.1, 2))
ax.set_xticks([0, 1], ["Without MSES", "With MSES"])
ax.set_ylabel("mAP@0.5 (%)")
handles = []
for label, keys, color, marker, offset in [
    ("Class projection", [GROUPS[0][1], GROUPS[1][1]], "#0072B2", "s", -0.035),
    ("Text projection", [GROUPS[2][1], GROUPS[3][1]], "#D55E00", "o", 0.035),
]:
    means = np.asarray([stats["per_group"][key]["mAP50"]["mean"] for key in keys]) * 100
    sds = np.asarray([stats["per_group"][key]["mAP50"]["sd"] for key in keys]) * 100
    for key, mean, sd in zip(keys, means, sds):
        raw = np.asarray(stats["per_group"][key]["mAP50"]["values"]) * 100
        np.testing.assert_allclose(raw.mean(), mean, atol=1e-10)
        np.testing.assert_allclose(raw.std(ddof=1), sd, atol=1e-10)
    handle = ax.errorbar(np.array([0, 1]) + offset, means, yerr=sds, color=color, marker=marker,
        linewidth=1.4, elinewidth=1.1, markersize=5.5, capsize=4.0, capthick=1.1,
        markeredgecolor="white", markeredgewidth=0.45, label=label, zorder=3)
    handles.append(handle)
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.555, 0.080), ncol=2,
           frameon=False, columnspacing=1.3, handlelength=2.1)
fig.text(0.555, 0.045, "Mean ± SD; n = 10", fontsize=9, color="#555555", ha="center")
export(fig, FILENAMES[1])

# C: Candidate pool counts and Q from the original manifest.
fig, ax = figure_frame("Pseudo-label acceptance")
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(0, 60)
ax.set_yticks(np.arange(0, 61, 10))
ax.set_ylabel("Accepted candidates (%)")
ax.set_xticks(range(4), [label for label, _, _ in GROUPS])
percentages = accepted / candidates * 100
bars = ax.bar(range(4), percentages, color=[color for _, _, color in GROUPS], width=0.65,
              edgecolor="white", linewidth=0.5, zorder=2)
for bar, count in zip(bars, accepted):
    ax.annotate(f"{count}/{candidates}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10.4)
threshold = ax.axhline(q / candidates * 100, color="#444444", linestyle=(0, (4, 3)), linewidth=1.0, zorder=3)
fig.legend([threshold], ["Q = 300 (6%)"], loc="lower center", bbox_to_anchor=(0.555, 0.055),
           frameon=False, handlelength=2.5)
export(fig, FILENAMES[2])

source_data = {"source_statistics": str(STATS_PATH), "source_manifest": str(MANIFEST_PATH),
    "detector_seeds": stats["design"]["detector_training_seeds"],
    "baseline_mAP50_percent": baseline.tolist(), "baseline_mean_percent": float(baseline.mean()),
    "candidate_pool_per_group": candidates, "Q": q, "groups": []}
for (label, key, color), count in zip(GROUPS, accepted):
    metric = stats["per_group"][key]["mAP50"]
    source_data["groups"].append({"label": label.replace("\n", " "), "key": key, "color": color,
        "mAP50_percent": (np.asarray(metric["values"]) * 100).tolist(),
        "mean_percent": metric["mean"] * 100, "sd_percent": metric["sd"] * 100,
        "accepted_count": int(count), "accepted_percent": float(count / candidates * 100)})
(OUT / "plotted_data.json").write_text(json.dumps(source_data, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "layout_verification.json").write_text(json.dumps(layout_reports, indent=2), encoding="utf-8")
with (OUT / "summary_data.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["Group", "Mean mAP@0.5 (%)", "SD (%)", "Accepted candidates", "Candidate pool", "Acceptance (%)"])
    for group in source_data["groups"]:
        writer.writerow([group["label"], group["mean_percent"], group["sd_percent"], group["accepted_count"],
                         candidates, group["accepted_percent"]])

readme = """# 图 3：三张独立科研图片

三张图片直接使用原图绘图脚本对应的原始数据重新绘制，未从截图估读数值。

- A_paired_detector_seeds：10 个配对检测器训练种子的结果；空心点表示 Real only，彩色点表示各增强方案，灰线连接同一个种子，虚线为 Real-only 平均值。为分开非常接近的点，仅调整少量水平方向位置，纵向数值及配对关系保持原样。
- B_condition_MSES_ablation：Class/Text projection 在有无 MSES 条件下的均值 ± 样本标准差（SD），n = 10。误差线略作水平错开以避免互相覆盖。
- C_pseudo_label_acceptance：每组候选池均为 5000 张，接受数依次为 1408、2037、2339、1272；Q = 300 对应 6% 的参考线。

已去除三张图左上角的 A、B、C 面板标号。所有字体（标题、轴标签、刻度、图例和注释）均为 Times New Roman（新罗马）。三张图尺寸一致：4.8 × 4.2 英寸；PNG 为 600 dpi、2880 × 2520 像素，白色背景。PDF 嵌入 TrueType 字体，SVG 的文字转为矢量路径以保持外观。preview 子目录仅用于快速预览。

已检查文字边界，无文字之间重叠，无画布裁切。

## 原始数据来源

原图脚本：deliverables/make_q3_paper_figures.py
统计数据：data/yolo_dataset/q3_fixedq_statistics.json
接受数量：data/yolo_dataset/q3_fixedq_manifest.json

plotted_data.json 保存此次绘图的全部数值；summary_data.csv 保存汇总值；redraw_figures.py 为可复现的绘图源代码。

## 重新生成

在该项目目录运行：python deliverables/figure3_split_times_new_roman/redraw_figures.py

需要 matplotlib 和 numpy，以及 Windows 的 Times New Roman 字体。
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
archive_path = OUT / "three_figures_Times_New_Roman.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(OUT.iterdir()):
        if file.is_file() and file != archive_path:
            archive.write(file, arcname=file.name)
print(json.dumps({"output_directory": str(OUT), "figures": layout_reports, "archive": str(archive_path)}, indent=2))
