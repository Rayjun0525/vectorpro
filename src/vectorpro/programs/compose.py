"""Generic compositions: folding and output windows."""

from __future__ import annotations

from typing import Sequence

import torch

from vectorpro.execution import BitExecutable
from vectorpro.quantize import Quantizer


def require_arity(name: str, executable: BitExecutable, arity: int) -> None:
    if executable.arity != arity:
        raise ValueError(f"{name} must have arity {arity}, got {executable.arity}")


class Fold(BitExecutable):
    """Left fold of a binary executable over ``arity`` operands.

    Each intermediate result is truncated to the operand width (keeps the low
    ``W`` bits), so e.g. folding addition gives a sum modulo ``2**W``.
    """

    def __init__(self, step: BitExecutable, arity: int) -> None:
        require_arity("Fold step", step, 2)
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


class Window(BitExecutable):
    """The ``word``-th ``W``-bit word of an executable's output.

    Turns wide results into fixed-width ones: word 0 of an adder is the sum
    modulo ``2**W``; word 1 of a divider is the remainder.
    """

    def __init__(self, inner: BitExecutable, word: int = 0) -> None:
        self.inner = inner
        self.word = word
        self.arity = inner.arity

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        width = operands.shape[-1]
        start = self.word * width
        if self.inner.output_width(width) < start + width:
            raise ValueError(f"inner output has no word {self.word} at width {width}")
        return self.inner.execute(operands, quantizer)[:, start : start + width]

    def children(self) -> Sequence[BitExecutable]:
        return (self.inner,)

    def compiled(self) -> Window:
        return Window(self.inner.compiled(), self.word)
