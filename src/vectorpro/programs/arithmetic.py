"""Multi-step arithmetic programs.

Programs here only wire: shift, slice, concatenate, broadcast and constants.
Every bit of computation goes through the units they are built from, so a
program is exactly as correct as its (verifiable) units.
"""

from __future__ import annotations

from typing import Sequence

import torch

from vectorpro.execution import BitExecutable
from vectorpro.programs.compose import require_arity
from vectorpro.quantize import Quantizer


class ShiftAddMultiply(BitExecutable):
    """Unsigned multiply: ``sum_i gate(a, b_i) << i`` accumulated with ``adder``.

    ``gate`` is a binary executable applied to ``a`` and the broadcast bit
    ``b_i`` (an AND, if learned correctly); ``adder`` must emit at least the
    operand width. Accumulation runs at width ``2W``, so the adder is used at
    twice the operand width.
    """

    arity = 2

    def __init__(self, gate: BitExecutable, adder: BitExecutable) -> None:
        require_arity("gate", gate, 2)
        require_arity("adder", adder, 2)
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


class RestoringDivide(BitExecutable):
    """Unsigned restoring long division. Output: quotient, then remainder (``2W`` bits).

    For each bit of ``a`` from the top: ``r = (r << 1) | a_i``; ``subtractor``
    gives ``r - b`` and a borrow-out (``r < b``). ``mux`` then keeps ``r`` or the
    difference and emits the quotient bit (``0`` on borrow, else ``1``).

    ``subtractor`` must output the ``W+1``-bit difference followed by the
    borrow; ``mux(x, y, s)`` picks ``y`` where ``s`` is 1, bitwise. Division by
    zero yields an all-ones quotient and the dividend as remainder (RISC-V).
    """

    arity = 2

    def __init__(self, subtractor: BitExecutable, mux: BitExecutable) -> None:
        require_arity("subtractor", subtractor, 2)
        require_arity("mux", mux, 3)
        self.subtractor = subtractor
        self.mux = mux

    def output_width(self, width: int) -> int:
        return 2 * width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        batch, _, width = operands.shape
        if self.subtractor.output_width(width + 1) != width + 2:
            raise ValueError("subtractor must emit difference plus one borrow bit")
        a, b = operands[:, 0], operands[:, 1]
        b_ext = torch.cat([b, b.new_zeros(batch, 1)], dim=1)
        one, zero = a.new_ones(batch, 1), a.new_zeros(batch, 1)
        rem = a.new_zeros(batch, width)
        quotient: list[torch.Tensor | None] = [None] * width
        for i in reversed(range(width)):
            shifted = torch.cat([a[:, i : i + 1], rem], dim=1)
            out = self.subtractor.execute(torch.stack([shifted, b_ext], dim=1), quantizer)
            diff, borrow = out[:, : width + 1], out[:, width + 1 :]
            kept = self.mux.execute(
                torch.stack([diff, shifted, borrow.expand(-1, width + 1)], dim=1), quantizer
            )
            rem = kept[:, :width]
            quotient[i] = self.mux.execute(torch.stack([one, zero, borrow], dim=1), quantizer)
        return torch.cat([*quotient, rem], dim=1)

    def children(self) -> Sequence[BitExecutable]:
        return (self.subtractor, self.mux)

    def compiled(self) -> RestoringDivide:
        return RestoringDivide(self.subtractor.compiled(), self.mux.compiled())
