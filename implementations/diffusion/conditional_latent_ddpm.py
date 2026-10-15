# -*- coding: utf-8 -*-
"""Class-conditional latent DDPM/DDIM baseline for the cable-defect study.

The held-out 48-image test set is never read by this script.  A frozen
``stabilityai/sd-vae-ft-mse`` VAE encodes only the 193 training images.  A
small class-conditional U-Net is then trained in the 64x64 latent space with
classifier-free guidance.  Sampling uses deterministic DDIM and decodes to
512x512 RGB images.

This is intentionally a conservative modern baseline, not a claim that the
diffusion architecture is optimal.  Downstream evaluation must use the same
teacher, fixed Q=300 and detector seeds as the GAN groups.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "cable_leakfix_train"
DEFAULT_CACHE = ROOT / "data" / "diffusion" / "cable_ldm_latents_s42.pt"
DEFAULT_CHECKPOINT_DIR = ROOT / "implementations" / "diffusion" / "checkpoints" / "cable_ldm_s42"
DEFAULT_OUTPUT = ROOT / "data" / "yolo_dataset" / "images" / "leakfix" / "gen_Diffusion_LDM_s42_5k"
DEFAULT_FLAT_OUTPUT = ROOT / "data" / "yolo_dataset" / "images" / "leakfix" / "gen_Diffusion_LDM_s42_5k_flat"
DEFAULT_FLAT_1K = ROOT / "data" / "yolo_dataset" / "images" / "leakfix" / "gen_Diffusion_LDM_s42_flat"
DEFAULT_PROTOCOL = ROOT / "data" / "diffusion" / "q3_diffusion_protocol.json"
VAE_ID = "stabilityai/sd-vae-ft-mse"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_TO_ENGLISH = {
    "凹点": "dent",
    "划痕": "scratch",
    "形变": "deformation",
    "鼓包": "bulge",
    "dent": "dent",
    "scratch": "scratch",
    "deformation": "deformation",
    "bulge": "bulge",
}
ENGLISH_ORDER = ["dent", "scratch", "deformation", "bulge"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_records(data_root: Path) -> list[dict[str, object]]:
    records = []
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        class_name = CLASS_TO_ENGLISH.get(class_dir.name)
        if class_name is None:
            raise ValueError(f"Unknown class directory: {class_dir.name}")
        for image_path in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            records.append(
                {
                    "path": image_path,
                    "relative_path": image_path.relative_to(data_root).as_posix(),
                    "class_name": class_name,
                    "class_id": ENGLISH_ORDER.index(class_name),
                    "sha256": sha256(image_path),
                }
            )
    if len(records) != 193:
        raise ValueError(f"Expected exactly 193 training images, found {len(records)}")
    return records


def augment_image(path: Path, seed: int, size: int = 512) -> torch.Tensor:
    """Apply the same augmentation family used by the GAN training scripts."""
    rng = random.Random(seed)
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(image, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        if rng.random() < 0.5:
            image = TF.hflip(image)
        if rng.random() < 0.5:
            image = TF.vflip(image)
        image = TF.rotate(
            image,
            rng.uniform(-10.0, 10.0),
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0],
        )
        image = TF.adjust_brightness(image, rng.uniform(0.85, 1.15))
        image = TF.adjust_contrast(image, rng.uniform(0.85, 1.15))
        image = TF.adjust_saturation(image, rng.uniform(0.90, 1.10))
        tensor = TF.to_tensor(image)
    return tensor.mul(2.0).sub(1.0)


def load_vae(device: torch.device, dtype: torch.dtype) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(VAE_ID)
    vae.requires_grad_(False).eval().to(device=device, dtype=dtype)
    vae.enable_slicing()
    return vae


@torch.inference_mode()
def build_cache(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache)
    manifest_path = cache_path.with_suffix(".json")
    records = source_records(Path(args.data))
    expected = len(records) * args.augment_repeats
    if cache_path.exists() and manifest_path.exists() and not args.overwrite:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(payload["labels"]) == expected:
            print(f"[CACHE] complete: {cache_path} ({expected} latents)", flush=True)
            return
        raise ValueError("Existing latent cache does not match the requested protocol; use --overwrite explicitly")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = load_vae(device, dtype)
    scaling_factor = float(vae.config.scaling_factor)
    latents: list[torch.Tensor] = []
    labels: list[int] = []
    sources: list[int] = []
    start = time.time()
    pending_images: list[torch.Tensor] = []
    pending_labels: list[int] = []
    pending_sources: list[int] = []

    def flush() -> None:
        if not pending_images:
            return
        images = torch.stack(pending_images).to(device=device, dtype=dtype)
        encoded = vae.encode(images).latent_dist.mean * scaling_factor
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("Non-finite VAE latents; retry cache creation on CPU/float32")
        latents.extend(encoded.cpu().half().unbind(0))
        labels.extend(pending_labels)
        sources.extend(pending_sources)
        pending_images.clear()
        pending_labels.clear()
        pending_sources.clear()

    for repeat in range(args.augment_repeats):
        for source_index, record in enumerate(records):
            augmentation_seed = args.seed * 1_000_003 + repeat * len(records) + source_index
            pending_images.append(augment_image(record["path"], augmentation_seed))
            pending_labels.append(int(record["class_id"]))
            pending_sources.append(source_index)
            if len(pending_images) >= args.vae_batch_size:
                flush()
        print(f"[CACHE] repeat {repeat + 1}/{args.augment_repeats}", flush=True)
    flush()

    payload = {
        "latents": torch.stack(latents),
        "labels": torch.tensor(labels, dtype=torch.long),
        "source_indices": torch.tensor(sources, dtype=torch.int32),
        "class_names": ENGLISH_ORDER,
        "vae_id": VAE_ID,
        "vae_scaling_factor": scaling_factor,
        "image_size": 512,
        "seed": args.seed,
        "augment_repeats": args.augment_repeats,
    }
    torch.save(payload, cache_path)
    manifest = {
        "status": "complete",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "training_images": len(records),
        "latent_records": len(labels),
        "class_counts_real": dict(sorted(Counter(str(r["class_name"]) for r in records).items())),
        "class_counts_latent": dict(sorted(Counter(ENGLISH_ORDER[label] for label in labels).items())),
        "image_size": 512,
        "latent_shape": list(payload["latents"].shape[1:]),
        "augment_repeats": args.augment_repeats,
        "augmentation_seed": args.seed,
        "vae_id": VAE_ID,
        "vae_scaling_factor": scaling_factor,
        "source_files": [
            {key: value for key, value in record.items() if key != "path"} for record in records
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CACHE] wrote {cache_path}: {len(labels)} x {tuple(payload['latents'].shape[1:])}", flush=True)


def make_model() -> UNet2DModel:
    return UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        block_out_channels=(64, 128, 192, 256),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
        norm_num_groups=32,
        num_class_embeds=5,  # four defect conditions plus classifier-free null condition
    )


def checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path]:
    return checkpoint_dir / "latest.pt", checkpoint_dir / "final.pt"


def save_checkpoint(
    path: Path,
    model: UNet2DModel,
    ema_model: UNet2DModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "class_names": ENGLISH_ORDER,
            "config": {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "condition_dropout": args.condition_dropout,
                "ema_decay": args.ema_decay,
                "noise_timesteps": args.noise_timesteps,
                "cache": str(Path(args.cache).resolve()),
            },
        },
        temporary,
    )
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    cache = torch.load(Path(args.cache), map_location="cpu", weights_only=False)
    latents = cache["latents"]
    labels = cache["labels"]
    if latents.shape[1:] != (4, 64, 64) or len(labels) != len(latents):
        raise ValueError(f"Unexpected cache shape: {tuple(latents.shape)} / {tuple(labels.shape)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal latent diffusion training requires CUDA on this workstation")
    set_seed(args.seed)

    dataset = TensorDataset(latents, labels)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        generator=generator,
    )
    model = make_model().to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    warmup_steps = min(args.warmup_steps, max(1, args.max_steps // 10))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.noise_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path, final_path = checkpoint_paths(checkpoint_dir)
    resume_path = final_path if final_path.exists() else latest_path
    global_step = 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema_model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        scaler.load_state_dict(state["scaler"])
        global_step = int(state["global_step"])
        print(f"[RESUME] {resume_path}, step={global_step}", flush=True)
    if global_step >= args.max_steps:
        print(f"[TRAIN] already complete at step {global_step}", flush=True)
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_micro_batches = 0
    interval_start = time.time()
    epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        for batch_latents, batch_labels in loader:
            batch_latents = batch_latents.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_labels = batch_labels.to(device=device, non_blocking=True)
            null_mask = torch.rand(batch_labels.shape, device=device) < args.condition_dropout
            conditioned_labels = batch_labels.clone()
            conditioned_labels[null_mask] = 4
            noise = torch.randn_like(batch_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(batch_latents, noise, timesteps)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(noisy_latents, timesteps, class_labels=conditioned_labels).sample
                loss = F.mse_loss(prediction.float(), noise.float()) / args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            running_micro_batches += 1

            micro_step = getattr(train, "_micro_step", 0) + 1
            train._micro_step = micro_step
            if micro_step % args.gradient_accumulation != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.mul_(args.ema_decay).add_(parameter, alpha=1.0 - args.ema_decay)

            if global_step % args.log_interval == 0:
                elapsed = time.time() - interval_start
                avg_loss = running_loss / max(1, running_micro_batches)
                print(
                    f"[TRAIN] step={global_step}/{args.max_steps} epoch={epoch} "
                    f"loss={avg_loss:.6f} lr={lr_scheduler.get_last_lr()[0]:.3e} "
                    f"sec/{args.log_interval}={elapsed:.1f}",
                    flush=True,
                )
                running_loss = 0.0
                running_micro_batches = 0
                interval_start = time.time()
            if global_step % args.save_interval == 0:
                save_checkpoint(latest_path, model, ema_model, optimizer, lr_scheduler, scaler, global_step, args)
                print(f"[CHECKPOINT] {latest_path}", flush=True)
            if global_step >= args.max_steps:
                break

    save_checkpoint(final_path, model, ema_model, optimizer, lr_scheduler, scaler, global_step, args)
    save_checkpoint(latest_path, model, ema_model, optimizer, lr_scheduler, scaler, global_step, args)
    print(f"[TRAIN] complete: {final_path}", flush=True)


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(args.checkpoint_dir)
    _, final_path = checkpoint_paths(checkpoint_dir)
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {final_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Sampling 5000 images requires CUDA on this workstation")
    model = make_model().to(device)
    state = torch.load(final_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema_model"])
    model.eval()
    vae = load_vae(device, torch.float16)
    scaling_factor = float(vae.config.scaling_factor)
    scheduler = DDIMScheduler(
        num_train_timesteps=args.noise_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.sampling_steps, device=device)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in enumerate(ENGLISH_ORDER):
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(class_dir.glob("*.png")))
        if existing > args.n_per_class:
            raise ValueError(f"{class_dir} has {existing} images, expected at most {args.n_per_class}")
        start_index = existing
        while start_index < args.n_per_class:
            current_batch = min(args.sample_batch_size, args.n_per_class - start_index)
            batch_generator = torch.Generator(device=device).manual_seed(
                args.sample_seed + class_id * 100_000 + start_index
            )
            latent = torch.randn((current_batch, 4, 64, 64), generator=batch_generator, device=device)
            labels = torch.full((current_batch,), class_id, dtype=torch.long, device=device)
            null_labels = torch.full_like(labels, 4)
            for timestep in scheduler.timesteps:
                model_input = torch.cat([latent, latent], dim=0)
                label_input = torch.cat([null_labels, labels], dim=0)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predicted = model(model_input, timestep, class_labels=label_input).sample
                unconditioned, conditioned = predicted.chunk(2)
                guided = unconditioned + args.guidance_scale * (conditioned - unconditioned)
                latent = scheduler.step(guided, timestep, latent, eta=0.0).prev_sample
            decoded = vae.decode((latent / scaling_factor).to(dtype=torch.float16)).sample.float()
            decoded = decoded.add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
            for offset, image_tensor in enumerate(decoded):
                index = start_index + offset
                image = TF.to_pil_image(image_tensor)
                image.save(class_dir / f"{class_name}_{index:05d}.png", compress_level=3)
            start_index += current_batch
            if start_index % 100 == 0 or start_index == args.n_per_class:
                print(f"[SAMPLE] {class_name}: {start_index}/{args.n_per_class}", flush=True)

    build_flat_outputs(
        output_root,
        Path(args.flat_output),
        Path(args.flat_1k),
        n_per_class=args.n_per_class,
    )
    protocol_path = Path(args.protocol)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    payload.update(
        {
            "status": "generation_complete",
            "model_family": "class-conditional latent DDPM trained with classifier-free guidance; DDIM sampling",
            "training_data": "193-image cable_leakfix_train only",
            "held_out_data_used": False,
            "vae_id": VAE_ID,
            "train_seed": args.seed,
            "sample_seed": args.sample_seed,
            "generated_images": 4 * args.n_per_class,
            "per_class": args.n_per_class,
            "sampling_steps": args.sampling_steps,
            "guidance_scale": args.guidance_scale,
            "output": str(output_root.resolve()),
            "flat_output": str(Path(args.flat_output).resolve()),
            "flat_1k": str(Path(args.flat_1k).resolve()),
            "downstream_protocol": {
                "teacher": "same real-only YOLOv8n teacher as all GAN groups",
                "pseudo_label_threshold": 0.10,
                "fixed_q": 300,
                "detector_seeds": [42, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "primary_outcome": "mAP@0.5",
                "analysis_label": "post-hoc protocol extension; include in updated Holm family",
            },
        }
    )
    protocol_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAMPLE] complete: {output_root}", flush=True)


def build_flat_outputs(
    source_root: Path,
    flat_5k: Path,
    flat_1k: Path,
    n_per_class: int,
) -> None:
    for destination in (flat_5k, flat_1k):
        destination.mkdir(parents=True, exist_ok=True)
    all_images: list[Path] = []
    subset_images: list[Path] = []
    for class_name in ENGLISH_ORDER:
        class_images = sorted((source_root / class_name).glob("*.png"))
        if len(class_images) != n_per_class:
            raise ValueError(f"Expected {n_per_class} {class_name} images, found {len(class_images)}")
        all_images.extend(class_images)
        subset_images.extend(class_images[: min(250, n_per_class)])
    expected_all = 4 * n_per_class
    expected_subset = 4 * min(250, n_per_class)
    for destination, expected, images in (
        (flat_5k, expected_all, all_images),
        (flat_1k, expected_subset, subset_images),
    ):
        existing = sorted(destination.glob("*.png"))
        if len(existing) == expected and {p.name for p in existing} == {p.name for p in images}:
            continue
        if existing:
            raise ValueError(f"Refusing to overwrite non-matching files in {destination}")
        for image in images:
            shutil.copy2(image, destination / image.name)
        print(f"[FLAT] {destination}: {expected}", flush=True)


def write_initial_protocol(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        return
    payload = {
        "status": "protocol_frozen_before diffusion result inspection",
        "created_unix": time.time(),
        "model_family": "class-conditional latent DDPM with classifier-free guidance and DDIM sampling",
        "training_data": "193-image cable_leakfix_train only",
        "held_out_data_used": False,
        "vae_id": VAE_ID,
        "train_seed": args.seed,
        "cache_augment_repeats": args.augment_repeats,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "condition_dropout": args.condition_dropout,
        "ema_decay": args.ema_decay,
        "noise_timesteps": args.noise_timesteps,
        "sample_seed": args.sample_seed,
        "sampling_steps": args.sampling_steps,
        "guidance_scale": args.guidance_scale,
        "generated_target": 5000,
        "downstream_fixed_q": 300,
        "detector_seeds": [42, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "primary_outcome": "mAP@0.5",
        "analysis_status": "post-hoc extension to the original study; all outcomes will be reported",
    }
    protocol_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cache", "train", "sample", "all"])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--flat-output", default=str(DEFAULT_FLAT_OUTPUT))
    parser.add_argument("--flat-1k", default=str(DEFAULT_FLAT_1K))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=20260914)
    parser.add_argument("--augment-repeats", type=int, default=8)
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--noise-timesteps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--n-per-class", type=int, default=1250)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_initial_protocol(args)
    if args.command in {"cache", "all"}:
        build_cache(args)
    if args.command in {"train", "all"}:
        train(args)
    if args.command in {"sample", "all"}:
        sample(args)


if __name__ == "__main__":
    main()
