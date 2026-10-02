"""Common interface for anything that executes on bit tensors.

Execution stays in tensor space end to end; integers appear only at the
encode/decode boundary in ``__call__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from vectorpro.bits import BitCodec
from vectorpro.quantize import HardThreshold, Quantizer


class BitExecutable(ABC):
    arity: int

    @abstractmethod
    def output_width(self, width: int) -> int: ...

    @abstractmethod
    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        """``operands: (B, arity, W)`` -> ``(B, output_width(W))``."""

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
