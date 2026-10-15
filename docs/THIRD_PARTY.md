# Third-party and pretrained-model notices

- Upstream PyTorch-GAN by Erik Linder-Norén: the original MIT license text is retained in `LICENSE`.
- OpenAI CLIP is an external dependency. Install from its official `openai/CLIP` source rather than an unrelated package of the same name; pretrained models are downloaded separately.
- The FastGAN-style experiments use `lightweight-gan` 1.2.1. This repository publishes the experiment wrapper, not a newly authored FastGAN implementation.
- YOLOv8n uses the installed Ultralytics distribution and COCO-pretrained `yolov8n.pt`. Weights are not bundled. Users must review Ultralytics' applicable license for their use case.
- LPIPS, torchvision pretrained networks, diffusers and the diffusion VAE (`stabilityai/sd-vae-ft-mse`) are external dependencies. Their software/model terms are separate from the upstream PyTorch-GAN notice.
- NEU-DET and the designated GC10-DET export remain subject to their respective data-source conditions. Dataset images and labels are not redistributed here.

This notice preserves attribution; it does not relicense third-party dependencies, model weights, datasets or proprietary annotations. An author-created code release is not an authorization to redistribute restricted data.
