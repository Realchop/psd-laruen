"""Shared optimiser setup, so architectures differ only in architecture.

A learning rate schedule is a confound: comparing a transformer on warmup plus
cosine against a convolution on a flat rate measures the schedule as much as
the model. Both get the same treatment here.
"""

import math
from collections.abc import Iterable

import torch
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LambdaLR

# Fraction of training spent warming up, when no explicit step count is given.
WARMUP_FRACTION = 0.05


def warmup_cosine(
    optimizer: optim.Optimizer, total_steps: int, warmup_steps: int
) -> LambdaLR:
    """Linear warmup, then cosine decay to zero.

    Transformers are unstable in the first few hundred steps at a rate they
    otherwise train happily at, which is what warmup is for.
    """
    warmup = max(1, min(warmup_steps, total_steps - 1)) if total_steps > 1 else 1

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_steps: int | None = None,
) -> OptimizerLRScheduler:
    """An AdamW plus warmup-cosine schedule, in Lightning's expected shape."""
    optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    if warmup_steps is None:
        warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": warmup_cosine(optimizer, total_steps, warmup_steps),
            "interval": "step",
        },
    }
