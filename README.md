# GAN paper: reproducible experiment code

这是工业缺陷生成增强研究的代码整理版。主方法为文本条件投影 WGAN-GP 与多尺度边缘统计匹配（MSES），包含类别条件对照、固定预算检测、统计分析与 NEU-DET / GC10-DET 外部验证。

本仓库整理的是**论文最终严格协议及其依赖代码**，不是工作区的整体备份。历史模型作为依赖或生成指标对照保留，不代表全部历史实验均进入最终检测结论。

## Public reproducibility records

The code is directly browsable in this repository. [MVA Online Resource 1](reproducibility/MVA_Online_Resource_1.zip) contains 361 files, including archived numerical outcomes, de-identified manifests and 320 strict detector configuration snapshots. The [checksum index](reproducibility/INDEX.json) and [archive description](reproducibility/ARCHIVE.json) make the package auditable. See [public release scope](docs/PUBLIC_RELEASE.md) for exclusions and verification. The author has requested that Chinese and English manuscripts remain private.

## Included / excluded

- Included: GAN training/sampling, latent diffusion extension, pseudo-labeling, fixed-Q detection, external validation, statistical analyses, annotation/protocol audits, figure source, configurations and dependencies.
- Excluded: raw/annotated images, original private image names, generated pools, model weights, training logs, Word manuscripts, manuscript-editing scripts, third-party full-text papers, credentials and caches. De-identified fixed memberships are available in the supplement; they are not original private filename lists.
- Original experimental source files were not modified. Only release copies received path/interpreter portability changes and a train-only default for the two core GAN scripts. `release_manifest.json` records source/release hashes and changes.

This release includes source code and a de-identified reproducibility archive. Manuscripts remain private. It is not a claim of complete data availability or one-click numerical reproduction. Exact reproduction needs the licensed datasets, archived fixed split/selection manifests and original checkpoints. Do not create a new random split and call it the published split.

## Code layout

| Directory | Purpose |
|---|---|
| `implementations/wgan_gp/` | projection GANs, MSES, generation, metrics, baseline wrappers |
| `implementations/diffusion/` | post-hoc class-conditional latent DDPM/DDIM extension |
| `implementations/yolov8_det/scripts/` | preparation, teacher labels, detection, paired statistics |
| `deliverables/` | rule-annotation audit, quantitative/qualitative figure source |
| `docs/` | paper-to-code map, protocol, data preparation, release limitations |
| `tools/` | non-training release checks |

## Repository organization

Use the browsable source directories as the canonical code copy. Download the current repository through **Code → Download ZIP**, or clone it:

```bash
git clone https://github.com/zqqqq-rgb/GAN.git
cd GAN
```

The separately maintained root-level code ZIP was removed because all 72 of its members already existed identically in the source tree. Earlier versions remain recoverable from Git history. The archive in `reproducibility/` is retained: it contains experiment records, not a duplicate source release.

Text/class projection, with/without MSES, dataset-specific detection pipelines, and individual/combined plot variants serve different experiments or outputs. They are not interchangeable duplicate files and remain separate. Keep the existing protocol, manifest paths and dependency relationships when adding code.

`.gitattributes` keeps text files in LF form on checkout so the published source hashes remain verifiable on Windows as well as other platforms. ZIP archives are treated as binary.

## Environment

The experiments used Windows 11, Python 3.13.5, PyTorch 2.6.0 + CUDA 12.4, torchvision 0.21.0, Ultralytics 8.4.129 and an RTX 4070 Laptop GPU (8 GB). `requirements.txt` records observed library versions, not a guarantee that every platform has compatible wheels. Install platform-appropriate PyTorch first, then install the remaining dependencies. OpenAI CLIP and model dependencies may download separately under their own terms.

Activate the intended Python environment. Detector subprocesses now use that interpreter and its site-packages rather than a workstation-specific Anaconda path.

```bash
python -m pip install -r requirements.txt
python tools/check_release.py
python tools/check_supplement.py
```

`check_release.py` checks syntax, release hashes, basic protocol constants and excluded-file patterns without importing GPU training scripts or launching training. This release has not been end-to-end retrained on a clean machine.

## Read before running

1. Follow [data preparation](docs/DATA.md) and restore the fixed experiment manifests through an authorized channel.
2. Read [protocol and seed definitions](docs/PROTOCOL.md).
3. Find each experiment in [paper-to-code map](docs/PAPER_CODE_MAP.md).
4. Launch only one GPU task at a time on an 8 GB GPU. Use separate output directories for reruns.
5. Inspect preparation scripts before running: some replace existing generated folders or rewrite splits. Do not run historical `build_dataset.py` as the final fixed-split pipeline; it is retained as the source of the rule-box function used by the annotation audit.

Example core GAN training: run from `implementations/wgan_gp/` so its relative output paths stay consistent.

```bash
python cwgan_gp_clip_proj.py --data_path ../../data/cable_leakfix_train --n_epochs 500 --batch_size 4 --lambda_sp 5 --lambda_fm 50 --seed 42 --tag e7proj_mses_sp_leak
```

For text/no-MSES change `--lambda_sp` to `0`. For class projection use `cwgan_gp_proj.py`. The archived protocol, not an inferred default, determines the experiment tag, source dataset, sampling seed and candidate count. Generation on an 8 GB GPU should pass `--batch_size 4` explicitly.

Example main fixed-Q detector pipeline, **after** the real-only teacher, fixed training/test lists, accepted synthetic pools and labels have been prepared:

```bash
python implementations/yolov8_det/scripts/q3_fixedq_pipeline.py all --q 300 --seeds 42 1 2 3 4 5 6 7 8 9 --groups baseline_stepmatched E5_tradit FastGAN copypaste E6proj_mses_nosp E6proj_mses_sp E7proj_mses_nosp E7proj_mses_sp Diffusion_LDM Diffusion_LDM_gs1 Diffusion_LDM_gs2 --strict-resume
```

Statistical scripts consume archived JSON results under the dataset directories. Some execute their analysis immediately when launched; do not import them just to inspect their functions. Image-level bootstrap additionally requires checkpoints and per-image predictions and may run GPU inference; it is not a code-only smoke test.

## Interpretation boundaries

- Real-image resampling matches extra training records/steps **and changes empirical image weights**; it is not a pure optimization-step intervention.
- Improved FID or teacher acceptance is not proof of improved detection or semantic controllability.
- Report the Holm-adjusted comparisons. Non-significance is not proof of zero effect; do not restore superseded historical gains.
- Absolute mAP gains are percentage points; relative gains are percentages. Original analysis JSON typically stores absolute differences. Do not change units without defining the conversion and bootstrap statistic.
- GC10's selected pseudo-labeled subset lacks teacher boxes for three classes. This is a coverage limitation of generation plus teacher filtering, not proof that the generator cannot produce those classes.

## Attribution

The repository retains the upstream PyTorch-GAN MIT notice in `LICENSE`. External dependencies and pretrained weights remain subject to their own licenses; see [third-party notes](docs/THIRD_PARTY.md). Dataset redistribution is not authorized by this code release.
