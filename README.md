# GAN paper experiment code

工业缺陷生成增强论文的代码整理版：文本条件投影 WGAN-GP、多尺度边缘统计匹配（MSES）、对照模型、固定预算检测和 NEU-DET / GC10-DET 外部验证。

## Download and use

完整源码位于本仓库的 [gan_paper_github_release.zip](gan_paper_github_release.zip)。网页上传采用压缩包发布，以保留源码原有目录结构。请先下载并解压；不要在仓库根目录直接执行训练命令。

解压后有 67 个文件，其中 52 个 Python 文件，包含以下内容：

- `implementations/wgan_gp/`：GAN 训练、生成、MSES 和生成指标。
- `implementations/diffusion/`：条件潜空间扩散扩展。
- `implementations/yolov8_det/scripts/`：数据准备、教师伪标签、固定预算检测、配对统计及外部验证。
- `deliverables/`：标注审计与论文图表的源代码。
- `docs/`：实验协议、论文与代码对应表、数据准备及第三方依赖说明。
- `tools/check_release.py`：不启动 GPU 训练的代码发布检查。

使用前请阅读压缩包内的 `README.md`、`docs/PROTOCOL.md` 和 `docs/DATA.md`。压缩包包含依赖版本记录及 `release_manifest.json`，后者记录源文件与发布文件的 SHA-256。

## Availability and interpretation

本次为代码发布，不包含原始或生成图像、私有图像名称与划分清单、模型权重、训练日志、论文文件或凭据。精确复现实验数值仍需合法获取的数据、固定实验清单及原始检查点；本代码包没有在全新环境中重新完成端到端训练。

真实图重采样对照同时匹配额外训练记录/步数并改变经验样本权重，不能解释为纯优化步数干预。生成指标与教师接收率不能替代检测收益或语义可控性证据，应以配对分析及 Holm 校正后的结论为准。

## Release checks

本地检查通过：52 个 Python 文件语法检查、58 个来源文件哈希核对、固定检测协议常量检查，以及不应发布的文件类型检查。这些是代码整理检查，不代表训练结果重新复现。

压缩包 SHA-256：

```text
113bd18d4445333cde262d0a43abcaf67e50a75b7e607634ec8dd653d58a23a0
```

## Attribution

保留上游 PyTorch-GAN 的 MIT 声明，见 `LICENSE`。外部依赖、预训练权重及数据集遵循各自的许可；详见压缩包内 `docs/THIRD_PARTY.md`。本代码发布不授予数据集再分发权限。
