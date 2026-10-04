"""Common interface for anything that executes on bit tensors.

Execution stays in tensor space end to end; integers appear only at the
encode/decode boundary in ``__call__``. Executables form a tree: programs
compose child executables, and leaves are function units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn

from vectorpro.bits import BitCodec
from vectorpro.quantize import HardThreshold, Quantizer


class BitExecutable(ABC):
    arity: int

    @property
    def effects(self) -> bool:
        """Whether execution can observe or change external state."""
        return any(child.effects for child in self.children())

    @abstractmethod
    def output_width(self, width: int) -> int: ...

    @abstractmethod
    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        """``operands: (B, arity, W)`` -> ``(B, output_width(W))``."""

    @abstractmethod
    def compiled(self) -> BitExecutable:
        """Same structure with every learned cell replaced by its binary table."""

    def children(self) -> Sequence[BitExecutable]:
        return ()

    def parameters(self) -> list[nn.Parameter]:
        """Trainable parameters of this executable and its children, deduplicated."""
        seen: dict[int, nn.Parameter] = {}
        for child in self.children():
            for p in child.parameters():
                seen.setdefault(id(p), p)
        return list(seen.values())

    def __call__(
        self,
        operand_tuples: Sequence[Sequence[int]],
        width: int,
        quantizer: Quantizer | None = None,
    ) -> list[int]:
        quantizer = quantizer or HardThreshold()
        operands = BitCodec.encode_operands(operand_tuples, width)
        with torch.no_grad():
            return BitCodec.decode(self.execute(operands, quantizer))


def fit_width(bits: torch.Tensor, width: int) -> torch.Tensor:
    """Truncate or zero-extend ``(B, n)`` bits to ``width``: how results land in a W-bit register."""
    if bits.shape[1] >= width:
        return bits[:, :width]
    return torch.cat([bits, bits.new_zeros(bits.shape[0], width - bits.shape[1])], dim=1)


class Fitted(BitExecutable):
    """An executable whose result is fitted to the operand width (``W -> W``)."""

    def __init__(self, inner: BitExecutable) -> None:
        self.inner = inner
        self.arity = inner.arity

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        return fit_width(self.inner.execute(operands, quantizer), operands.shape[-1])

    def children(self) -> Sequence[BitExecutable]:
        return (self.inner,)

    def compiled(self) -> Fitted:
        return Fitted(self.inner.compiled())
