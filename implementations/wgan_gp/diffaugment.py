# -*- coding: utf-8 -*-
"""
DiffAugment (Zhao et al., "DiffAugment: Data-Free Augmentation for GAN Training", CVPR 2020)
可微分增强 —— 梯度可流过生成器，仅在判别器输入端施加。
策略: color, translation, cutout (官方推荐小数据策略)
参考官方实现: https://github.com/csdwren/DiffAugment (PyTorch版, MIT)
"""
import torch
import torch.nn.functional as F

POLICY = "color,translation,cutout"


def DiffAugment(x, policy=POLICY):
    if policy:
        for p in policy.split(","):
            for f in AUG_FNS[p]:
                x = f(x)
    return x


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) / 2.0


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def rand_translation(x, ratio=0.125):
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    batch, channel, height, width = x.size()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch, dtype=torch.long, device=x.device),
        torch.arange(height, dtype=torch.long, device=x.device),
        torch.arange(width, dtype=torch.long, device=x.device),
        indexing="ij")
    new_x = torch.clamp(grid_x + tx.reshape(-1, 1, 1).expand(-1, height, width), 0, width - 1)
    new_y = torch.clamp(grid_y + ty.reshape(-1, 1, 1).expand(-1, height, width), 0, height - 1)
    return x.permute(0, 2, 3, 1)[grid_batch, new_x, new_y].permute(0, 3, 1, 2).contiguous()


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij")
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUG_FNS = {
    "brightness": [rand_brightness],
    "saturation": [rand_saturation],
    "contrast": [rand_contrast],
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
