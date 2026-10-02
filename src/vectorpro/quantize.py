"""Quantizers applied to values passed between cell steps."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Quantizer(ABC):
    """Maps cell outputs before they are carried to the next step."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Identity(Quantizer):
    """Keeps values continuous (soft execution, used for training)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __repr__(self) -> str:
        return "Identity()"


class HardThreshold(Quantizer):
    """Snaps values to 0/1. Gradients pass straight through."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hard = (x >= self.threshold).to(x.dtype)
        return x + (hard - x).detach()

    def __repr__(self) -> str:
        return f"HardThreshold({self.threshold})"
