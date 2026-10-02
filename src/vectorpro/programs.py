"""Compositions of executables. Intermediate values never leave tensor space."""

from __future__ import annotations

import torch

from vectorpro.execution import BitExecutable
from vectorpro.quantize import Quantizer


class Fold(BitExecutable):
    """Left fold of a binary executable over ``arity`` operands.

    Each intermediate result is truncated to the operand width (keeps the low
    ``W`` bits), so e.g. folding addition gives a sum modulo ``2**W``.
    """

    def __init__(self, step: BitExecutable, arity: int) -> None:
        if step.arity != 2:
            raise ValueError(f"Fold needs a binary step, got arity {step.arity}")
        if arity < 2:
            raise ValueError("Fold needs at least two operands")
        self.step = step
        self.arity = arity

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        width = operands.shape[-1]
        if self.step.output_width(width) < width:
            raise ValueError("step output is narrower than its operands")
        acc = operands[:, 0]
        for k in range(1, operands.shape[1]):
            pair = torch.stack([acc, operands[:, k]], dim=1)
            acc = self.step.execute(pair, quantizer)[:, :width]
        return acc
