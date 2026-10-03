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
