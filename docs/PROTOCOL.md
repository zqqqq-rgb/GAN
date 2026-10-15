# Final paper protocol

## Dataset boundaries and detector training

| Dataset | Real train | Held out | Detector image size | Batch | Epochs |
|---|---:|---:|---:|---:|---:|
| Cable | 193 | 48 | 640 | 4 | 30 |
| NEU-DET | 1439 | 360 | 320 | 4 | 30 |
| GC10-DET | 1604 | 688 | 320 | 4 | 30 |

All detector comparisons use YOLOv8n, detection seeds `[42,1,2,3,4,5,6,7,8,9]`, `patience=0`, `val=False`, `workers=0`, and final `last.pt` instead of selected `best.pt`. Ultralytics can still perform a final-epoch validation internally even with `val=False`. No early stopping or checkpoint choice may be based on a held-out score. GC10 training YAML points both train and val at training records; only the post-training evaluation YAML points at the 688 held-out images.

The main dataset final held-out evaluation is reused across prespecified paired detector seeds, not treated as additional independent test data. Public-data GANs and teacher/selection pipelines must not read the held-out partition.

## Random settings

| Operation | Setting |
|---|---|
| Cable stratified real-data split | seed 42; frozen 193/48 membership |
| Main fixed-Q accepted synthetic selection | seed 20260910 |
| Nested Q-sensitivity selection | seed 20260912; Q = 100, 200, 300, 500 |
| NEU Q=300 selection/resampling | seed 20260912 |
| GC10 split and Q=300 selection | seed 20260922 |
| GC10 candidate latent sampling | seed 20260922 |
| Core GAN robustness | generating seeds 42, 1, 2 for text/no-MSES, class/no-MSES, text/+MSES |
| Class/+MSES core comparison | generating seed 42 only |
| GC10 text/+MSES external generator | generating seed 42 only |
| Latent diffusion extension | generating seeds 42, 1, 2 |
| Detector paired repetitions | seeds 42 and 1 through 9 |

The four condition/MSES configurations form a 2x2 comparison at generator seed 42; only three configurations have three-GAN-seed robustness evaluation. Do not claim four complete three-seed configurations. The default latent-sampling seeds of historical scripts are not a substitute for the archived command/manifest.

## Model and annotation settings

- Core cable GAN: 512x512, 500 epochs, batch 4; lambda_fm=50, lambda_sp=5 (MSES) or 0 (ablation). Check the archived arguments for learning rate, gradient penalty and critic steps.
- MSES: pooled Sobel statistics at grids 32,16,8, population standard deviation, std-term weight 0.5.
- GC10 GAN: 1604 training images, one condition per image, 500 epochs, batch 4. For 327 multi-label images choose the least prevalent present training class, tie broken by class index. Original detector labels retain all classes.
- GC10 candidates: ten conditions x 500 images = 5000 at 512x512. Teacher is fixed Real-only seed42 `last.pt`; retained confidence >=0.10; Q=300. The GAN condition is not a pseudo-label.
- Cable teacher starts prediction at 0.05; retains boxes at >=0.10. FN/FP summaries use 0.25, not the maximum-F1 operating point used for reported Recall.
- NEU public-data generators are trained at 256x256. Do not reuse leaked historical NEU experiments as final evidence.
- Latent DDPM is a post-hoc extension with a frozen pretrained VAE, 20000 updates, three generating seeds, 50-step DDIM and guidance 2.0. Its latent-space batch setting is separate from the detector batch-4 protocol.

## Statistical units

Keep detector-seed, image-bootstrap and generating-seed analyses separate. Paired Wilcoxon comparisons against Real-only and against resampling are independent Holm families as specified by each analysis script. Control-vs-baseline supplementary p values are unadjusted. The GC10 method's three comparisons constitute its separate three-test Holm family.

Image-level bootstrap resamples the 48 held-out images together across methods, recomputes AP separately for each fixed model, then averages scores without fusing prediction boxes. It does not replace an independent production-line test set. The JSON statistics use absolute mAP differences unless explicitly documented otherwise.

Training scripts retain cuDNN benchmark settings used in the experiments; exact bitwise determinism is not claimed. Preserve the original split/selection artifacts to avoid path sorting changes across operating systems.

## Safety

Run GPU stages serially. On CUDA/driver errors or a black screen, stop rather than automatically restarting. A final filename alone does not prove a detector finished 30 epochs; verify `results.csv`, `args.yaml`, checkpoint and result metadata. Some original runners have weaker completion checks than the final monitoring workflow: perform that check before reusing an output. PowerShell scripts prepare/overwrite generated subsets and should be reviewed before running.
