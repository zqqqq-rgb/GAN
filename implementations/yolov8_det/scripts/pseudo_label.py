'\npseudo_label.py\n================\n两阶段伪标签：用 baseline YOLOv8n（196张真实图训练）对各组生成/增强图推理，\n以预测框替代原"中心弱监督"标注，重跑4.3检测实验。\n\n模式:\n  stats : 仅推理并统计各阈值下的标注覆盖率（不写文件）\n  write : 以 --conf 阈值写入伪标签(备份原中心框标签)并重建split\n\nUsage:\n  python -S pseudo_label.py stats\n  python -S pseudo_label.py write --conf 0.25\n'

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import os
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, _release_sysconfig.get_paths()['purelib'])

PROJECT_ROOT = Path(str(_release_root))
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
BASELINE_PT = PROJECT_ROOT / "implementations" / "yolov8_det" / "runs" / "yolov8n_baseline" / "weights" / "best.pt"

GROUPS = ["E5_tradit", "E4_dcgan", "E2_wgan_gp", "E1_sp", "E7_clip"]
IMGSZ = 640
PRED_JSON = YOLO_ROOT / "pseudo_predictions.json"
LABEL_BACKUP = YOLO_ROOT / "labels_centerbox_backup"


def run_predictions():
    """用baseline模型对5组共5000张图推理，原始预测框存JSON"""
    from ultralytics import YOLO
    model = YOLO(str(BASELINE_PT))
    all_preds = {}
    for grp in GROUPS:
        img_dir = YOLO_ROOT / "images" / f"gen_{grp}"
        preds = {}
        results = model.predict(source=str(img_dir), conf=0.05, imgsz=IMGSZ,
                                stream=True, verbose=False)
        for r in results:
            fname = Path(r.path).name
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for bb, cf in zip(xyxy, confs):
                    # xyxy -> normalized xywh (YOLO label format)
                    x1, y1, x2, y2 = bb
                    xc = ((x1 + x2) / 2) / r.orig_shape[1]
                    yc = ((y1 + y2) / 2) / r.orig_shape[0]
                    w = (x2 - x1) / r.orig_shape[1]
                    h = (y2 - y1) / r.orig_shape[0]
                    boxes.append([round(float(cf), 4),
                                  round(float(xc), 6), round(float(yc), 6),
                                  round(float(w), 6), round(float(h), 6)])
            preds[fname] = boxes
        all_preds[grp] = preds
        print(f"[PRED] {grp}: {len(preds)} images done")
    PRED_JSON.write_text(json.dumps(all_preds), encoding="utf-8")
    print(f"[SAVE] {PRED_JSON}")
    return all_preds


def print_stats(all_preds):
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print("\n===== 标注覆盖率统计 (每图至少1个 conf>=t 的框的图像比例) =====")
    header = f"{'group':<12}{'total':>7}" + "".join(f"{t:>9}" for t in thresholds)
    print(header)
    for grp in GROUPS:
        preds = all_preds[grp]
        n = len(preds)
        row = f"{grp:<12}{n:>7}"
        for t in thresholds:
            cov = sum(1 for boxes in preds.values()
                      if any(b[0] >= t for b in boxes))
            row += f"{cov / n * 100:>8.1f}%"
        print(row)
    # 平均框数 & 置信度分布 (t=0.25)
    print("\n===== t=0.25 时的细节 =====")
    for grp in GROUPS:
        preds = all_preds[grp]
        kept = {f: [b for b in boxes if b[0] >= 0.25] for f, boxes in preds.items()}
        labeled = {f: bs for f, bs in kept.items() if bs}
        nbox = sum(len(bs) for bs in labeled.values())
        confs = [b[0] for bs in labeled.values() for b in bs]
        confs.sort()
        med = confs[len(confs) // 2] if confs else 0
        print(f"{grp:<12} labeled={len(labeled)}/{len(preds)}  "
              f"avg_box={nbox / max(len(labeled), 1):.2f}  median_conf={med:.3f}")


def train_baseline_lines(grp):
    """读 baseline split (196真实图绝对路径) 复用"""
    p = YOLO_ROOT / "splits" / "train_baseline.txt"
    return p.read_text(encoding="utf-8").splitlines()


def write_labels(conf_th):
    """写入伪标签: 备份原标签 -> 覆盖写 -> 重建split"""
    all_preds = json.loads(PRED_JSON.read_text())
    # 1. 备份现有中心框标签
    if not LABEL_BACKUP.exists():
        LABEL_BACKUP.mkdir(parents=True)
        for grp in GROUPS:
            src = YOLO_ROOT / "labels" / f"gen_{grp}"
            dst = LABEL_BACKUP / f"gen_{grp}"
            shutil.copytree(src, dst)
        print(f"[BACKUP] 中心框标签 -> {LABEL_BACKUP}")
    else:
        print(f"[BACKUP] 已存在备份, 跳过")
    # 2. 写伪标签 (只保留 conf>=th 的框; 无框图不写标签文件)
    summary = {}
    for grp in GROUPS:
        preds = all_preds[grp]
        label_dir = YOLO_ROOT / "labels" / f"gen_{grp}"
        # 清空旧txt
        for f in label_dir.glob("*.txt"):
            f.unlink()
        n_labeled = 0
        for fname, boxes in preds.items():
            kept = [b for b in boxes if b[0] >= conf_th]
            if not kept:
                continue  # 无框 -> 不生成标签 -> split中排除
            stem = Path(fname).stem
            lines = ["0 " + " ".join(f"{v:.6f}" for v in b[1:]) for b in kept]
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_labeled += 1
        summary[grp] = n_labeled
        print(f"[WRITE] gen_{grp}: {n_labeled}/{len(preds)} 张获得伪标签")
    # 3. 重建 split (real 196 + 有标签的生成图)
    for grp in GROUPS:
        train_txt = YOLO_ROOT / "splits" / f"train_{grp}.txt"
        img_dir = YOLO_ROOT / "images" / f"gen_{grp}"
        labeled_stems = {f.stem for f in (YOLO_ROOT / "labels" / f"gen_{grp}").glob("*.txt")}
        # 先重置split为196真实 + 1000生成（统一结构）
        real_lines = train_baseline_lines(grp)
        gen_lines = []
        for f in sorted(img_dir.glob("*.png")):
            p = str(f.resolve())
            if f.stem in labeled_stems:
                gen_lines.append(p)
        all_lines = real_lines + gen_lines
        train_txt.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
        print(f"[SPLIT] train_{grp}.txt: {len(all_lines)} 行 (真实{len(real_lines)}+生成{len(gen_lines)})")
    # 4. 清理旧label cache (真实图cache不受影响, 但保险起见删gen相关)
    (YOLO_ROOT / "labels" / "all_real.cache").unlink(missing_ok=True)
    print("[DONE] 伪标签写入完成")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if mode == "stats":
        if PRED_JSON.exists():
            preds = json.loads(PRED_JSON.read_text())
        else:
            preds = run_predictions()
        print_stats(preds)
    elif mode == "write":
        conf = 0.25
        if "--conf" in sys.argv:
            conf = float(sys.argv[sys.argv.index("--conf") + 1])
        print(f"[MODE] write, conf_th={conf}")
        write_labels(conf)
