"""Draw the three quantitative panels at manuscript print size."""
from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "figure3_split_times_new_roman/plotted_data.json"
data = json.loads(SOURCE.read_text(encoding="utf-8"))
groups = data["groups"]
baseline = np.asarray(data["baseline_mAP50_percent"])
labels = ["Class\nno MSES", "Class\n+MSES", "Text\nno MSES", "Text\n+MSES"]
WIDTH_MM, HEIGHT_MM = 180, 78
PANEL_WIDTH_MM = WIDTH_MM / 3
PLOT_WIDTH_MM = PLOT_HEIGHT_MM = 45
PLOT_LEFT_MM, PLOT_BOTTOM_MM = 13.5, 22
HEADER_Y_MM, LEGEND_TOP_MM = 73.5, 11.5
DPI = 600
NAME = "figure3_combined_equal_panels_180mm"

for font in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / font))
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "axes.edgecolor": "#262626", "axes.labelcolor": "#262626", "text.color": "#202020",
    "xtick.color": "#262626", "ytick.color": "#262626",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=240)
axes = [fig.add_axes([
    (i * PANEL_WIDTH_MM + PLOT_LEFT_MM) / WIDTH_MM,
    PLOT_BOTTOM_MM / HEIGHT_MM,
    PLOT_WIDTH_MM / WIDTH_MM,
    PLOT_HEIGHT_MM / HEIGHT_MM,
]) for i in range(3)]
panel_centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axes]
titles = ["Paired detector seeds (n = 10)", "Condition × MSES ablation", "Pseudo-label acceptance"]
for letter, ax, title in zip("ABC", axes, titles):
    box = ax.get_position()
    fig.text(box.x0 - 8.64 / WIDTH_MM, HEADER_Y_MM / HEIGHT_MM, letter,
             fontsize=11, fontweight="bold", ha="left", va="center")
    fig.text(box.x0 + box.width / 2, HEADER_Y_MM / HEIGHT_MM, title,
             fontsize=9, ha="center", va="center")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.4)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=3)
    ax.set_ylabel("mAP@0.5 (%)", labelpad=4)


def swarm_positions(ax, center, values, marker_area=8):
    """Preserve y values while separating dots that coincide."""
    spacing = np.sqrt(marker_area) * fig.dpi / 72 * 1.12
    xcenter = ax.transData.transform((center, 0))[0]
    result = np.zeros(len(values))
    placed = []
    for idx in np.argsort(values, kind="stable"):
        yp = ax.transData.transform((center, values[idx]))[1]
        found = False
        for step in range(8):
            for direction in ([1] if step == 0 else [1, -1]):
                xp = xcenter + direction * step * spacing
                if all(np.hypot(xp - ox, yp - oy) >= spacing for ox, oy in placed):
                    result[idx] = ax.transData.inverted().transform((xp, yp))[0]
                    placed.append((xp, yp))
                    found = True
                    break
            if found:
                break
        assert found
    return result


# A: keep every original observation and seed pairing.
ax = axes[0]
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(43, 66)
ax.set_yticks([45, 50, 55, 60, 65])
ax.set_xticks(range(4), labels)
for x, group in enumerate(groups):
    values = np.asarray(group["mAP50_percent"])
    bx = swarm_positions(ax, x - 0.185, baseline)
    vx = swarm_positions(ax, x + 0.185, values)
    for bxi, vxi, b, v in zip(bx, vx, baseline, values):
        ax.plot([bxi, vxi], [b, v], color="#B7BCC4", linewidth=0.5, alpha=0.75, zorder=1)
    ax.scatter(bx, baseline, s=8, facecolor="white", edgecolor="#6B6B6B", linewidth=0.55, zorder=3)
    ax.scatter(vx, values, s=9, facecolor=group["color"], edgecolor="white", linewidth=0.28, zorder=4)
ax.axhline(data["baseline_mean_percent"], color="#727272", linestyle=(0, (4, 3)), linewidth=0.75, zorder=2)
fig.legend(handles=[
    Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor="#6B6B6B",
           markeredgewidth=0.55, markersize=3.1, label="Real only"),
    Line2D([], [], color="#727272", linestyle=(0, (4, 3)), linewidth=0.75, label="Real-only mean"),
], loc="upper center", bbox_to_anchor=(panel_centers[0], LEGEND_TOP_MM / HEIGHT_MM), frameon=False, ncol=1,
   borderaxespad=0, labelspacing=0.35, handlelength=2.0, handletextpad=0.6)

# B: original means ± sample standard deviation, ten detector seeds.
ax = axes[1]
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(51.5, 64.1)
ax.set_yticks([52, 54, 56, 58, 60, 62, 64])
ax.set_xticks([0, 1], ["Without\nMSES", "With\nMSES"])
handles = []
for title, indexes, color, marker, offset in [
    ("Class projection", [0, 1], "#0072B2", "s", -0.035),
    ("Text projection", [2, 3], "#D55E00", "o", 0.035),
]:
    means = [groups[i]["mean_percent"] for i in indexes]
    sds = [groups[i]["sd_percent"] for i in indexes]
    for i, mean, sd in zip(indexes, means, sds):
        raw = np.asarray(groups[i]["mAP50_percent"])
        np.testing.assert_allclose(raw.mean(), mean, atol=1e-10)
        np.testing.assert_allclose(raw.std(ddof=1), sd, atol=1e-10)
    handle = ax.errorbar(np.asarray([0, 1]) + offset, means, yerr=sds, label=title, color=color,
        marker=marker, markersize=3.7, markeredgecolor="white", markeredgewidth=0.3,
        linewidth=1.05, elinewidth=0.8, capsize=2.8, capthick=0.8, zorder=3)
    handles.append(handle)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(panel_centers[1], LEGEND_TOP_MM / HEIGHT_MM), frameon=False,
           ncol=1, borderaxespad=0, labelspacing=0.35, handlelength=2.0, handletextpad=0.6)

# C: original candidate pool acceptance counts.
ax = axes[2]
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(0, 60)
ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax.set_ylabel("Accepted candidates (%)", labelpad=4)
ax.set_xticks(range(4), labels)
counts = np.asarray([group["accepted_count"] for group in groups])
pool = data["candidate_pool_per_group"]
bars = ax.bar(range(4), counts / pool * 100, width=0.65,
              color=[group["color"] for group in groups], edgecolor="white", linewidth=0.4, zorder=2)
for bar, count in zip(bars, counts):
    ax.annotate(f"{count}/{pool}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
threshold = ax.axhline(data["Q"] / pool * 100, color="#444444", linestyle=(0, (4, 3)), linewidth=0.75, zorder=3)
fig.legend([threshold], ["Q = 300 (6%)"], loc="upper center",
           bbox_to_anchor=(panel_centers[2], LEGEND_TOP_MM / HEIGHT_MM),
           frameon=False, borderaxespad=0, handlelength=2.0, handletextpad=0.6)


def check_layout():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        texts += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
    for legend in fig.legends:
        texts += list(legend.get_texts())
    texts = [text for text in texts if text.get_visible() and text.get_text().strip()]
    boxes = []
    for text in texts:
        bbox = text.get_window_extent(renderer)
        assert text.get_fontproperties().get_name() == "Times New Roman"
        assert text.get_fontsize() >= 7
        assert bbox.x0 >= 0 and bbox.x1 <= fig.bbox.x1, f"Horizontal clipping: {text.get_text()}"
        assert bbox.y0 >= 0 and bbox.y1 <= fig.bbox.y1, f"Vertical clipping: {text.get_text()}"
        boxes.append((text.get_text(), bbox))
    for i, (text, bbox) in enumerate(boxes):
        for other_text, other_bbox in boxes[i + 1:]:
            assert not bbox.overlaps(other_bbox), f"Text overlap: {text!r} / {other_text!r}"
    panel_geometry = []
    for label, ax in zip("ABC", axes):
        box = ax.get_position()
        width = box.width * WIDTH_MM
        height = box.height * HEIGHT_MM
        np.testing.assert_allclose([width, height], [45, 45], atol=1e-10)
        panel_geometry.append({"label": label, "outer_panel_mm": [PANEL_WIDTH_MM, HEIGHT_MM],
                               "plot_mm": [round(width, 6), round(height, 6)],
                               "plot_bottom_mm": round(box.y0 * HEIGHT_MM, 6),
                               "plot_top_mm": round(box.y1 * HEIGHT_MM, 6)})
    return {"width_mm": WIDTH_MM, "height_mm": HEIGHT_MM, "dpi": DPI, "font": "Times New Roman",
            "panel_labels": ["A", "B", "C"], "text_count": len(texts),
            "equal_panel_geometry": panel_geometry,
            "minimum_font_pt": min(t.get_fontsize() for t in texts),
            "text_overlap": False, "text_clipping": False}


report = check_layout()
# Each export retains the exact canvas dimensions; do not apply bbox_inches='tight'.
for extension in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"{NAME}.{extension}", dpi=DPI, facecolor="white")
fig.savefig(OUT / f"{NAME}_preview.png", dpi=240, facecolor="white")
with Image.open(OUT / f"{NAME}.png") as png:
    report["png_pixels"] = list(png.size)
    report["png_dpi"] = list(png.info["dpi"])
plt.close(fig)
(OUT / "layout_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(OUT / "plotted_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

readme = """# 合并后的论文图片：子图大小一致

三个子图横向排列，保留 A、B、C 标号。所有文字采用 Times New Roman；按实际印刷尺寸设置刻度 8 pt、轴标签 9 pt、标题 9 pt、图例 7.5 pt、面板标号 11 pt。图形直接用原始数值重绘，配对关系、均值、标准差和接受数量与此前三张独立图片一致。

三个子图采用相同的 60 × 78 mm 区域，每个绘图区为 45 × 45 mm；绘图区上、下边界对齐，ABC 标号与标题对齐，横轴均为双行标签，图例首行对齐。B 图误差线为 10 个种子的均值 ± 样本标准差（SD）。

画布尺寸：180 × 78 mm（18 × 7.8 cm），适用于常见论文双栏通栏布局。
PNG：600 dpi，白色背景，适合插入 Word。
PDF：矢量图，嵌入 Times New Roman 字体，适合 LaTeX 和投稿。
SVG：矢量图，文字转为路径以保持外观。

## 插入方式

Word：插入 figure3_combined_equal_panels_180mm.png，保持纵横比，宽度设置为 18 cm，高度为 7.8 cm；如正文版心较窄，可以等比例设置为 16 cm，高度约 6.93 cm。关闭 Word 图片压缩或选择高保真，可避免清晰度降低。

LaTeX：使用 figure3_combined_equal_panels_180mm.pdf，例如双栏稿件可在 figure* 环境中使用 \\includegraphics[width=\\textwidth]{figure3_combined_equal_panels_180mm.pdf}。

不同期刊的版心宽度可能不同；此文件按 18 cm 通用宽度制作。

绘图源代码：draw_combined_figure.py；完整数值：plotted_data.json。
所有文字已经检查，未发现文字重叠或裁切。
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
archive_path = OUT / "combined_equal_panels.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    package_files = [f"{NAME}.{ext}" for ext in ("png", "pdf", "svg")]
    package_files += [f"{NAME}_preview.png", "draw_combined_figure.py", "plotted_data.json",
                      "layout_verification.json", "README.md"]
    for filename in package_files:
        archive.write(OUT / filename, arcname=filename)
print(json.dumps(report, indent=2))
print(str(archive_path))
