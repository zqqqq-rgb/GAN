# -*- coding: utf-8 -*-
'\nleakfix_det_pipeline.py — 泄漏修复版检测闭环 (阶段B)\n\n协议:\n  教师 baseline YOLOv8n 在 193 张泄漏修复训练子集上训练 (30ep/640/batch4/seed42/workers=0)\n  -> 对 5 组 leakfix 生成图推理 (conf>=0.10 伪标签, 无框图剔除)\n  -> 各增强组 = 193 real + 有效生成图, YOLO 训练 + 48 张 held-out val 评估\n  -> results 写入 data/yolo_dataset/results_leakfix.json\n\n子命令:\n  baseline           教师 baseline 训练\n  pseudo             伪标签推理+写入+重建split (需 baseline 完成)\n  det [--seeds 42]   各组检测训练+评估 (默认 seed42; 5种子显著性用 --seeds 42 1 2 3 4)\n  all                baseline -> pseudo -> det\n\n用法:\n  python leakfix_det_pipeline.py all\n  python leakfix_det_pipeline.py det --seeds 42 1 2 3 4 --groups baseline E7_clip\n'

# Public-release portability only; experiment parameters are unchanged.
from pathlib import Path as _ReleasePath
import sys as _release_sys
import sysconfig as _release_sysconfig
_release_root = _ReleasePath(__file__).resolve().parents[2]

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(str(_release_root))
YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
SPLIT_DIR = YOLO_ROOT / "splits_leakfix"
GEN_ROOT = YOLO_ROOT / "images" / "leakfix"
LBL_ROOT = YOLO_ROOT / "labels" / "leakfix"
RUNS_DIR = PROJECT_ROOT / "implementations" / "yolov8_det" / "runs_leakfix"
RESULTS_JSON = YOLO_ROOT / "results_leakfix.json"
PRED_JSON = YOLO_ROOT / "pseudo_predictions_leakfix.json"
PY = _release_sys.executable
SITE = _release_sysconfig.get_paths()['purelib']

# 组名 -> 生成图目录名 (leakfix)
GEN_GROUPS = {
    "E5_tradit": "gen_E5_tradit_leak",
    "E4_dcgan": "gen_E4_dcgan_leak",
    "E2_wgan_gp": "gen_E2_wgan_gp_leak",
    "E1_sp": "gen_E1_sp_leak",
    "E7_clip": "gen_E7_clip_leak_flat",   # 4类子目录 flatten (YOLO loader 不递归)
    "E6_cc": "gen_E6_leak_flat",          # 4类子目录 flatten
    "copypaste": "gen_copypaste_leak",  # oracle 精确标注, 不走伪标签
    "E2DA": "gen_E2_wgan_gp_da_leak",    # StageA2: WGAN-GP+DiffAugment (Zhao 2020), 1000 png 512px
    "FastGAN": "gen_FastGAN_leak",        # StageA2: lightweight-gan (Cui 2021), 1000 jpg 256px
    "E7proj_mses_sp": "gen_E7proj_mses_sp_leak_5k_flat",
    "E7proj_mses_nosp": "gen_E7proj_mses_nosp_leak_5k_flat",
    "E6proj_mses_sp": "gen_E6proj_mses_sp_leak_5k_flat",
    "E6proj_mses_nosp": "gen_E6proj_mses_nosp_leak_5k_flat",
    "E7proj_mses_nosp_s1": "gen_e7proj_mses_nosp_leak_s1_5k_flat",
    "E7proj_mses_nosp_s2": "gen_e7proj_mses_nosp_leak_s2_5k_flat",
    "E6proj_mses_nosp_s1": "gen_e6proj_mses_nosp_leak_s1_5k_flat",
    "E6proj_mses_nosp_s2": "gen_e6proj_mses_nosp_leak_s2_5k_flat",
    "E7proj_mses_sp_s1": "gen_e7proj_mses_sp_leak_s1_5k_flat",
    "E7proj_mses_sp_s2": "gen_e7proj_mses_sp_leak_s2_5k_flat",
    "E7proj_mses_sp_full": "gen_E7proj_mses_sp_leak_5k_flat",
    "Diffusion_LDM": "gen_Diffusion_LDM_s42_5k_flat",
    "Diffusion_LDM_gs1": "gen_Diffusion_LDM_s1_5k_flat",
    "Diffusion_LDM_gs2": "gen_Diffusion_LDM_s2_5k_flat",
}
PSEUDO_GROUPS = ["E5_tradit", "E4_dcgan", "E2_wgan_gp", "E1_sp", "E7_clip", "E6_cc", "E2DA", "FastGAN",
                 "E7proj_mses_sp", "E7proj_mses_nosp", "E6proj_mses_sp", "E6proj_mses_nosp"]
DET_GROUPS = ["baseline", "E5_tradit", "E4_dcgan", "E2_wgan_gp", "E1_sp", "E7_clip", "E6_cc", "copypaste", "E2DA", "FastGAN",
              "E7proj_mses_sp", "E7proj_mses_nosp", "E6proj_mses_sp", "E6proj_mses_nosp"]


def _find_image(img_dir, stem):
    """根据 stem 在 img_dir 找实际图片文件 (支持 .png/.jpg/.jpeg/.bmp)"""
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return str(p).replace("\\", "/")
    return None

EPOCHS, IMGSZ, BATCH, PATIENCE = 30, 640, 4, 0
GPU_COOLDOWN_SECONDS = 20


def _console_safe(text):
    """Keep captured child-process diagnostics printable on Windows GBK consoles."""
    encoding = sys.stdout.encoding or "utf-8"
    return (text or "").encode(encoding, errors="backslashreplace").decode(encoding)


def train_yolo(data_yaml, run_dir, seed=42, weights="yolov8n.pt"):
    """子进程训练 (python -S + sys.path 注入, 防 DataLoader/缓存问题)"""
    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "from ultralytics import YOLO; "
        "m = YOLO(r'%s'); "
        "m.train(data=r'%s', epochs=%d, imgsz=%d, batch=%d, seed=%d, "
        "patience=%d, val=False, project=r'%s', name='%s', exist_ok=True, "
        "plots=False, save=True, verbose=False, cache=False, workers=0)"
        % (SITE, weights, data_yaml, EPOCHS, IMGSZ, BATCH, seed,
           PATIENCE, str(RUNS_DIR), run_dir.name)
    )
    r = subprocess.run([PY, "-S", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT))
    time.sleep(GPU_COOLDOWN_SECONDS)
    if r.returncode != 0:
        print(_console_safe(r.stdout[-1500:]))
        print(_console_safe(r.stderr[-1500:]))
    return r.returncode == 0


def eval_yolo(best_pt, data_yaml):
    code = (
        "import sys, json; sys.path.insert(0, r'%s'); "
        "from ultralytics import YOLO; "
        "m = YOLO(r'%s'); "
        "metrics = m.val(data=r'%s', imgsz=%d, batch=%d, plots=False, "
        "save_json=False, verbose=False, workers=0); "
        "out = {'mAP50': float(metrics.box.map50), "
        "'mAP50_95': float(metrics.box.map), "
        "'precision': float(metrics.box.mp), "
        "'recall': float(metrics.box.mr)}; "
        "print('JSON:' + json.dumps(out))"
        % (SITE, best_pt, data_yaml, IMGSZ, BATCH)
    )
    r = subprocess.run([PY, "-S", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT))
    for line in (r.stdout or "").splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    print((r.stdout or "")[-1000:], (r.stderr or "")[-1000:])
    return None


def build_yaml(group, tag=""):
    """group 专属 data.yaml; tag 用于多种子区分 (同一 split 可共用)"""
    name = f"data_lf_{group}{tag}.yaml"
    y = YOLO_ROOT / name
    train_rel = "splits_leakfix/train_baseline.txt" if group == "baseline" \
        else f"splits_leakfix/train_{group}.txt"
    content = (f"# leakfix {group}\n"
               f"path: {str(YOLO_ROOT).replace(chr(92), '/')}\n"
               f"train: {train_rel}\n"
               f"val: splits_leakfix/val.txt\n"
               f"nc: 1\nnames:\n  0: defect\n")
    y.write_text(content, encoding="utf-8")
    return str(y)


def cmd_baseline(seed=42):
    yaml_path = build_yaml("baseline")
    run_dir = RUNS_DIR / f"yolov8n_baseline_lf"
    print(f"[BASELINE] train on 193 real, val on 48 held-out -> {run_dir.name}")
    ok = train_yolo(yaml_path, run_dir, seed=seed)
    print("[BASELINE]", "OK" if ok else "FAILED")
    return ok


def cmd_pseudo(conf=0.10, groups=None):
    """教师推理 N 组生成图 -> 写伪标签 -> 重建 split (groups=None 则跑全部 PSEUDO_GROUPS)"""
    from ultralytics import YOLO  # noqa: 主进程加载一次即可, 推理用子进程太慢
    last_pt = RUNS_DIR / "yolov8n_baseline_lf" / "weights" / "last.pt"
    assert last_pt.exists(), "教师 baseline 未训练, 先跑 baseline 子命令"
    model = YOLO(str(last_pt))

    groups = groups or PSEUDO_GROUPS

    LBL_ROOT.mkdir(parents=True, exist_ok=True)
    all_preds = {}
    for grp in groups:
        img_dir = GEN_ROOT / GEN_GROUPS[grp]
        preds = {}
        for r in model.predict(source=str(img_dir), conf=0.05, imgsz=IMGSZ,
                               stream=True, verbose=False, workers=0):
            fname = Path(r.path).name
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for bb, cf in zip(xyxy, confs):
                    x1, y1, x2, y2 = bb
                    xc = ((x1 + x2) / 2) / r.orig_shape[1]
                    yc = ((y1 + y2) / 2) / r.orig_shape[0]
                    w = (x2 - x1) / r.orig_shape[1]
                    h = (y2 - y1) / r.orig_shape[0]
                    boxes.append([round(float(cf), 4), round(float(xc), 6),
                                  round(float(yc), 6), round(float(w), 6), round(float(h), 6)])
            preds[fname] = boxes
        all_preds[grp] = preds
        cov = sum(1 for bs in preds.values() if any(b[0] >= conf for b in bs))
        print(f"[PRED] {grp}: {len(preds)} imgs, coverage@{conf}: {cov}/{len(preds)}")

    # 合并已有 predictions, 避免覆盖旧组 (兼容--groups子集跑)
    if PRED_JSON.exists():
        prev = json.loads(PRED_JSON.read_text(encoding="utf-8"))
    else:
        prev = {}
    prev.update(all_preds)
    PRED_JSON.write_text(json.dumps(prev), encoding="utf-8")

    # 写伪标签
    real_lines = (SPLIT_DIR / "train.txt").read_text(encoding="utf-8").splitlines()
    summary = {}
    for grp in groups:
        lbl_dir = LBL_ROOT / GEN_GROUPS[grp]
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        lbl_dir.mkdir(parents=True)
        n = 0
        for fname, boxes in all_preds[grp].items():
            kept = [b for b in boxes if b[0] >= conf]
            if not kept:
                continue
            stem = Path(fname).stem
            lines = ["0 " + " ".join(f"{v:.6f}" for v in b[1:]) for b in kept]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += 1
        summary[grp] = n
        # 重建 split: 193 real + 有标签生成图 (跨扩展名查找 .png/.jpg/.jpeg/.bmp)
        img_dir = GEN_ROOT / GEN_GROUPS[grp]
        gen_lines = []
        for f in lbl_dir.glob("*.txt"):
            p = _find_image(img_dir, f.stem)
            if p:
                gen_lines.append(p)
        out = SPLIT_DIR / f"train_{grp}.txt"
        out.write_text("\n".join(real_lines + sorted(gen_lines)) + "\n", encoding="utf-8")
        print(f"[SPLIT] {out.name}: {len(real_lines)} real + {len(gen_lines)} pseudo-labeled")
    print("[PSEUDO] summary:", summary)


def cmd_det(seeds, groups, force_train=False, strict_resume=False):
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    for seed in seeds:
        for grp in groups:
            key = f"lf_{grp}_s{seed}" if seed != 42 else f"lf_{grp}"
            prior = results.get(key, {})
            strict_done = (prior.get("checkpoint") == "last.pt" and
                           prior.get("train_val") is False and
                           prior.get("fixed_epochs") == EPOCHS and
                           prior.get("detector_batch") == BATCH)
            needs_train = force_train or (strict_resume and not strict_done)
            if grp == "baseline":
                run_dir = RUNS_DIR / "yolov8n_baseline_lf"
                if seed != 42:
                    run_dir = RUNS_DIR / f"yolov8n_baseline_lf_s{seed}"
                    if needs_train or not (run_dir / "weights" / "last.pt").exists():
                        if not train_yolo(build_yaml("baseline"), run_dir, seed=seed):
                            raise RuntimeError(f"Training failed for {key}; stopping to protect the GPU and preserve protocol consistency")
                elif needs_train:
                    if not train_yolo(build_yaml("baseline"), run_dir, seed=seed):
                        raise RuntimeError(f"Training failed for {key}; stopping to protect the GPU and preserve protocol consistency")
                checkpoint_pt = run_dir / "weights" / "last.pt"
            else:
                run_dir = RUNS_DIR / f"yolov8n_{grp}_lf" if seed == 42 else \
                    RUNS_DIR / f"yolov8n_{grp}_lf_s{seed}"
                if needs_train or not (run_dir / "weights" / "last.pt").exists():
                    ok = train_yolo(build_yaml(grp), run_dir, seed=seed)
                    if not ok:
                        results[key] = {"error": "train failed"}
                        RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
                        raise RuntimeError(f"Training failed for {key}; stopping to protect the GPU and preserve protocol consistency")
                checkpoint_pt = run_dir / "weights" / "last.pt"
            m = eval_yolo(str(checkpoint_pt), build_yaml(grp if grp != "baseline" else "baseline"))
            if m:
                m.update({"seed": seed, "checkpoint": "last.pt", "train_val": False,
                          "fixed_epochs": EPOCHS, "detector_batch": BATCH})
                results[key] = m
                RESULTS_JSON.write_text(json.dumps(results, indent=2))
                print(f"[DET] {key}: mAP50={m['mAP50']:.4f} mAP50_95={m['mAP50_95']:.4f} "
                      f"P={m['precision']:.4f} R={m['recall']:.4f}")
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print("[DONE] ->", RESULTS_JSON)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["baseline", "pseudo", "det", "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--groups", nargs="+", default=DET_GROUPS)
    p.add_argument("--pseudo-groups", nargs="+", default=None,
                   help="仅伪标这几组 (默认全部 PSEUDO_GROUPS); 用于增量加入新 baseline")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--force-train", action="store_true",
                   help="固定轮次从头重训并覆盖同名运行")
    p.add_argument("--strict-resume", action="store_true",
                   help="仅重训尚无当前严格协议元数据的运行")
    args = p.parse_args()

    if args.cmd in ("baseline", "all"):
        cmd_baseline(seed=args.seeds[0])
    if args.cmd in ("pseudo", "all"):
        cmd_pseudo(conf=args.conf, groups=args.pseudo_groups)
    if args.cmd in ("det", "all"):
        cmd_det(args.seeds, args.groups, force_train=args.force_train,
                strict_resume=args.strict_resume)


if __name__ == "__main__":
    main()
