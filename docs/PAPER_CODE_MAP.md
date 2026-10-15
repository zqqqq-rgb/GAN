# Paper-to-code map

| Paper component | Primary code |
|---|---|
| Text projection + MSES, feature matching, gradient penalty | `implementations/wgan_gp/cwgan_gp_clip_proj.py` |
| Class projection and MSES ablation | `implementations/wgan_gp/cwgan_gp_proj.py` |
| Frozen-text and class-conditioned candidate generation | `cgenerate_images_clip.py`, `cgenerate_images.py` in the same directory |
| Three core configurations with generating seeds 1/2 | `implementations/wgan_gp/run_q3_gan_multiseed.ps1` |
| Generator seed42 2x2 comparison | `implementations/wgan_gp/run_q3_projection_ablation.ps1` |
| DCGAN / WGAN-GP / DiffAugment | `dcgan_train.py`, `wgan_gp.py`, `diffaugment.py`, `generate_images.py` |
| FastGAN-style baseline | `train_fastgan.py`, `gen_fastgan_1000.py` (external lightweight-gan implementation) |
| Traditional / exact-box CopyPaste | `traditional_augment.py`; `implementations/yolov8_det/scripts/build_copypaste.py` |
| Post-hoc latent diffusion extension | `implementations/diffusion/conditional_latent_ddpm.py` |
| FID, KID, class-internal LPIPS, MSES distance | `implementations/wgan_gp/q3_generative_metrics.py` and three JSON configs |
| Independent condition-evaluator check | `implementations/wgan_gp/q3_condition_classifier.py` |
| Cable deduplication/frozen split | `implementations/yolov8_det/scripts/leakfix_prepare.py` |
| Teacher prediction and accepted pools | `implementations/yolov8_det/scripts/leakfix_det_pipeline.py` |
| Main fixed-Q and real-resampling detection | `implementations/yolov8_det/scripts/q3_fixedq_pipeline.py` |
| Main paired/Holm statistics | `analyze_q3_fixedq_statistics.py` |
| Generating-seed robustness | `analyze_q3_gan_seed_robustness.py` |
| Diffusion three-seed detector aggregation | `analyze_q3_diffusion_multiseed.py` |
| Per-model paired image bootstrap | `analyze_q3_image_bootstrap.py` |
| Per-defect AP and FN/FP | `analyze_q3_per_defect_type.py` |
| Nested budget sensitivity | `q3_q_sensitivity_current.py`, `analyze_q3_q_sensitivity_current.py` |
| Final checkpoint/protocol audit | `audit_q3_strict_last_protocol.py` |
| Rule-box reproducibility audit | `deliverables/audit_real_annotations.py`, helper `build_dataset.py` |
| NEU preparation and comparison | `neu_det_prepare.py`, `q3_neu_projection_pipeline.py`, `q3_neu_comparators_pipeline.py` |
| NEU statistical comparisons | `analyze_q3_neu_projection.py`, `analyze_q3_neu_comparators.py` |
| NEU generator comparator runner | `implementations/wgan_gp/run_q3_neu_comparators.ps1` |
| GC10 train/test audit and preparation | `gc10_prepare.py` |
| GC10 real-only / real resampling | `gc10_external_pipeline.py` |
| GC10 exact-box CopyPaste | `gc10_copypaste_pipeline.py` |
| GC10 single-condition GAN preparation | `implementations/wgan_gp/prepare_gc10_gan.py` |
| GC10 fixed-teacher GAN augmentation and Holm comparisons | `gc10_gan_external_pipeline.py` |
| Architecture/protocol figures | `deliverables/figures/draw_method_architecture.py`, `draw_evaluation_protocol.py` |
| Quantitative and qualitative figures | `deliverables/make_q3_paper_figures.py`, `make_q3_fixed_qualitative.py` |
| Individual/combined condition-ablation panels | `deliverables/figure3_split_times_new_roman/redraw_figures.py`, `figure3_combined_paper/draw_combined_figure.py` |

Unqualified detector/statistics filenames above reside in `implementations/yolov8_det/scripts/`. The old E1-E7 names inside archived results are retained for traceability; they are not an alternative experimental protocol.

Historical `cwgan_gp.py`, `cwgan_gp_clip.py`, `train_E6_neu.py`, `generate_E6_neu.py`, `fid_lpips_eval.py` are included only where earlier model/baseline dependencies require them. `build_dataset.py` contains the automatic rule-box function, but its executable historical split workflow is not the final leakage-free preparation entry point. Figure source includes original figure-generation variants; final manuscript fonts, legends and manual editing may differ. Word-editing scripts and misleading legacy detector pipelines are deliberately excluded.
