"""
engine/training/schedulers.py

Hand-written LR schedulers.  All return a LambdaLR so they plug
directly into the Trainer without any extra wrapping.
"""

from __future__ import annotations
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine decay with linear warm-up.  min_lr_ratio=0.1 matches Chinchilla."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 1.0 - (1 - min_lr_ratio) * progress)
    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        return min(1.0, step / max(1, num_warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer: Optimizer, cfg) -> LambdaLR:
    """
    Build the cosine scheduler from a TrainConfig.

    Convenience wrapper so experiments don't need to import schedulers directly.
    """
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )
