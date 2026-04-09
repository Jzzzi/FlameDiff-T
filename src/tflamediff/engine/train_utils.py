from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_optimizer(parameters, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = config.get("name", "adamw").lower()
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=tuple(config.get("betas", [0.9, 0.999])))
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay, betas=tuple(config.get("betas", [0.9, 0.999])))
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], max_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = config.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(config.get("min_lr", 1e-6)))
    raise ValueError(f"Unsupported scheduler: {name}")


def maybe_build_summary_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def resolve_precision(config: dict[str, Any]) -> tuple[bool, torch.dtype | None]:
    precision = str(config.get("precision", "fp16")).lower()
    amp_enabled = bool(config.get("amp", True))
    if not amp_enabled or precision in {"fp32", "32", "none"}:
        return False, None
    if precision in {"fp16", "16", "16-mixed"}:
        return True, torch.float16
    if precision in {"bf16", "bfloat16"}:
        return True, torch.bfloat16
    raise ValueError(f"Unsupported precision setting: {precision}")
