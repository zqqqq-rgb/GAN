# -*- coding: utf-8 -*-
"""Independent four-class evaluator for conditional generation.

The evaluator is trained only on the 193 real training images for a fixed
number of epochs.  The 48 held-out images are used once for reporting, never
for checkpoint selection.  Five fixed-seed ResNet-18 models are ensembled for
descriptive condition-consistency evaluation of generated folders.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "data" / "cable_leakfix_train"
FULL_ROOT = PROJECT_ROOT / "data" / "cable_defect_classified"
SPLIT_INFO = PROJECT_ROOT / "data" / "yolo_dataset" / "splits_leakfix" / "split_info.json"
MODEL_DIR = PROJECT_ROOT / "implementations" / "wgan_gp" / "q3_condition_models"
RESULT_PATH = PROJECT_ROOT / "data" / "yolo_dataset" / "q3_condition_consistency.json"

CLASS_NAMES = ["凹点", "划痕", "形变", "鼓包"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
GENERATED_TO_REAL = {
    "dent": "凹点",
    "scratch": "划痕",
    "deformation": "形变",
    "bulge": "鼓包",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SEEDS = [42, 1, 2, 3, 4]


class ListedImages(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_rows() -> list[tuple[Path, int]]:
    rows = []
    for name in CLASS_NAMES:
        for path in sorted((TRAIN_ROOT / name).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((path, CLASS_TO_INDEX[name]))
    if len(rows) != 193:
        raise ValueError(f"Expected 193 training images, found {len(rows)}")
    return rows


def held_out_rows() -> list[tuple[Path, int]]:
    split = json.loads(SPLIT_INFO.read_text(encoding="utf-8"))
    rows = []
    for name in CLASS_NAMES:
        for filename in split["val"][name]:
            path = FULL_ROOT / name / filename
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append((path, CLASS_TO_INDEX[name]))
    if len(rows) != 48:
        raise ValueError(f"Expected 48 held-out images, found {len(rows)}")
    return rows


def transforms_for_training():
    return transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
                             ResNet18_Weights.IMAGENET1K_V1.transforms().std),
    ])


def transforms_for_evaluation():
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
                             ResNet18_Weights.IMAGENET1K_V1.transforms().std),
    ])


def new_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def train_one(seed: int, epochs: int, batch_size: int, device: torch.device):
    set_seed(seed)
    dataset = ListedImages(train_rows(), transforms_for_training())
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    model = new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[TRAIN] seed={seed} epoch={epoch + 1}/{epochs} loss={np.mean(losses):.4f}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"resnet18_seed{seed}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "seed": seed,
        "epochs": epochs,
        "class_names": CLASS_NAMES,
        "training_images": len(dataset),
    }, path)
    print("[SAVE]", path)


@torch.inference_mode()
def predict(model: nn.Module, rows: list[tuple[Path, int]], batch_size: int, device: torch.device):
    loader = DataLoader(ListedImages(rows, transforms_for_evaluation()), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    probabilities, labels, paths = [], [], []
    model.eval()
    for images, batch_labels, batch_paths in loader:
        probabilities.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
        labels.extend(batch_labels.numpy().tolist())
        paths.extend(batch_paths)
    return np.concatenate(probabilities, axis=0), np.asarray(labels), paths


def load_models(device: torch.device) -> list[nn.Module]:
    models = []
    for seed in SEEDS:
        path = MODEL_DIR / f"resnet18_seed{seed}.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run train first")
        checkpoint = torch.load(path, map_location=device)
        model = new_model(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    return models


def metric_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "per_class_f1": {
            name: float(value)
            for name, value in zip(CLASS_NAMES, f1_score(labels, predicted, average=None, labels=range(4), zero_division=0))
        },
        "confusion_matrix": confusion_matrix(labels, predicted, labels=range(4)).tolist(),
    }


def generated_rows(root: Path, max_per_condition: int) -> list[tuple[Path, int]]:
    rows = []
    for generated_name, real_name in GENERATED_TO_REAL.items():
        directory = root / generated_name
        paths = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths[:max_per_condition]:
            rows.append((path, CLASS_TO_INDEX[real_name]))
    if not rows:
        raise FileNotFoundError(f"No condition subdirectories under {root}")
    return rows


def evaluate(batch_size: int, max_per_condition: int, config_path: Path, device: torch.device):
    models = load_models(device)
    held_rows = held_out_rows()
    per_seed = {}
    held_probabilities = []
    for seed, model in zip(SEEDS, models):
        probabilities, labels, _ = predict(model, held_rows, batch_size, device)
        held_probabilities.append(probabilities)
        per_seed[str(seed)] = metric_summary(labels, probabilities)
    held_ensemble = np.mean(held_probabilities, axis=0)
    output = {
        "protocol": {
            "classifier": "ImageNet-pretrained ResNet-18, all layers fine-tuned",
            "training_images": 193,
            "held_out_images": 48,
            "training_seeds": SEEDS,
            "checkpoint_selection": "fixed final epoch; held-out data not used for model selection",
            "class_names": CLASS_NAMES,
        },
        "held_out_real": {
            "per_seed": per_seed,
            "ensemble": metric_summary(labels, held_ensemble),
        },
        "generated": {},
    }

    configs = json.loads(config_path.read_text(encoding="utf-8"))
    for config in configs:
        root = Path(config["root"])
        rows = generated_rows(root, max_per_condition)
        model_probabilities = []
        per_model = {}
        for seed, model in zip(SEEDS, models):
            probabilities, generated_labels, _ = predict(model, rows, batch_size, device)
            model_probabilities.append(probabilities)
            per_model[str(seed)] = metric_summary(generated_labels, probabilities)
        ensemble = np.mean(model_probabilities, axis=0)
        output["generated"][config["name"]] = {
            "root": str(root),
            "n": len(rows),
            "per_seed": per_model,
            "ensemble": metric_summary(generated_labels, ensemble),
        }
        print(f"[GENERATED] {config['name']}: accuracy={output['generated'][config['name']]['ensemble']['accuracy']:.4f}")

    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]", RESULT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "evaluate", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_per_condition", type=int, default=250)
    parser.add_argument("--config", default=str(Path(__file__).with_name("q3_condition_config.json")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.cmd in {"train", "all"}:
        for seed in SEEDS:
            path = MODEL_DIR / f"resnet18_seed{seed}.pth"
            if path.exists():
                print("[SKIP]", path)
            else:
                train_one(seed, args.epochs, args.batch_size, device)
    if args.cmd in {"evaluate", "all"}:
        evaluate(args.batch_size, args.max_per_condition, Path(args.config), device)


if __name__ == "__main__":
    main()
