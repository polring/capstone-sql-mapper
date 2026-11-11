"""Optimizer settings shared by normal training and the memorization check."""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    accumulation_steps: int = 4
    epochs: int = 12
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    patience: int = 3
    seed: int = field(kw_only=True)
    precision: str = "auto"

    def __post_init__(self):
        if min(self.batch_size, self.accumulation_steps, self.epochs) < 1:
            raise ValueError("batch_size, accumulation_steps and epochs must be positive")
        if not (0 < self.min_lr <= self.lr and 0 <= self.warmup_ratio < 1):
            raise ValueError("invalid learning-rate range or warmup ratio")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.eps <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0 or self.patience < 0:
            raise ValueError("invalid optimizer or patience setting")
        if self.precision not in ("auto", "bf16", "fp32"):
            raise ValueError("precision must be auto, bf16 or fp32")

    def learning_rate(self, step: int, total: int) -> float:
        """One-based optimizer update; the final cosine step reaches min_lr."""
        warmup = min(math.ceil(total * self.warmup_ratio), total - 1)
        if step <= warmup:
            return self.lr * step / warmup
        progress = (step - warmup - 1) / max(1, total - warmup - 1)
        return self.min_lr + (self.lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
