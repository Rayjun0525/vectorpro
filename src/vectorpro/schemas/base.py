"""Schemas: the iteration structure that applies a cell across bit positions.

A schema is the (currently human-provided) structural hypothesis about how a
function decomposes into repeated local steps. It owns iteration order, state
threading and output layout; the cell owns the per-step rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vectorpro.cells.base import Cell, CellSignature
from vectorpro.quantize import Quantizer


class Schema(ABC):
    arity: int

    @abstractmethod
    def check(self, signature: CellSignature) -> None:
        """Raise ``ValueError`` if a cell with this signature cannot be used."""

    @abstractmethod
    def output_width(self, width: int, signature: CellSignature) -> int:
        """Result width in bits for ``width``-bit operands."""

    @abstractmethod
    def run(self, cell: Cell, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        """``operands: (B, arity, W)`` -> ``(B, output_width(W))``."""

    @abstractmethod
    def to_spec(self) -> dict:
        """JSON-serializable description; inverse of ``schema_from_spec``."""
