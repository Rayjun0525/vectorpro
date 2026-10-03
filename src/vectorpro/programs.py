"""Compositions of executables. Intermediate values never leave tensor space."""

from __future__ import annotations

from typing import Sequence

import torch

from vectorpro.execution import BitExecutable
from vectorpro.quantize import Quantizer


def _require_binary(name: str, executable: BitExecutable) -> None:
    if executable.arity != 2:
        raise ValueError(f"{name} must be binary, got arity {executable.arity}")


class Fold(BitExecutable):
    """Left fold of a binary executable over ``arity`` operands.

    Each intermediate result is truncated to the operand width (keeps the low
    ``W`` bits), so e.g. folding addition gives a sum modulo ``2**W``.
    """

    def __init__(self, step: BitExecutable, arity: int) -> None:
        _require_binary("Fold step", step)
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

    def children(self) -> Sequence[BitExecutable]:
        return (self.step,)

    def compiled(self) -> Fold:
        return Fold(self.step.compiled(), self.arity)


class ShiftAddMultiply(BitExecutable):
    """Unsigned multiply: ``sum_i gate(a, b_i) << i`` accumulated with ``adder``.

    ``gate`` is a binary executable applied to ``a`` and the broadcast bit
    ``b_i`` (an AND, if learned correctly); ``adder`` must emit at least the
    operand width. Accumulation runs at width ``2W``, so the adder is used at
    twice the operand width.
    """

    arity = 2

    def __init__(self, gate: BitExecutable, adder: BitExecutable) -> None:
        _require_binary("gate", gate)
        _require_binary("adder", adder)
        self.gate = gate
        self.adder = adder

    def output_width(self, width: int) -> int:
        return 2 * width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        batch, _, width = operands.shape
        a, b = operands[:, 0], operands[:, 1]
        acc = a.new_zeros(batch, 2 * width)
        for i in range(width):
            partial = self.gate.execute(
                torch.stack([a, b[:, i : i + 1].expand(-1, width)], dim=1), quantizer
            )[:, :width]
            shifted = torch.cat(
                [a.new_zeros(batch, i), partial, a.new_zeros(batch, width - i)], dim=1
            )
            acc = self.adder.execute(torch.stack([acc, shifted], dim=1), quantizer)[:, : 2 * width]
        return acc

    def children(self) -> Sequence[BitExecutable]:
        return (self.gate, self.adder)

    def compiled(self) -> ShiftAddMultiply:
        return ShiftAddMultiply(self.gate.compiled(), self.adder.compiled())
