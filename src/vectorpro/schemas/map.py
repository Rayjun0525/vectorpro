from __future__ import annotations

from dataclasses import dataclass

import torch

from vectorpro.cells.base import Cell, CellSignature
from vectorpro.quantize import Quantizer
from vectorpro.schemas.base import Schema


@dataclass(frozen=True)
class MapSchema(Schema):
    """Stateless, position-wise application (bitwise ops). Positions run in parallel."""

    arity: int

    def check(self, signature: CellSignature) -> None:
        if signature.n_inputs != self.arity:
            raise ValueError(f"cell takes {signature.n_inputs} inputs, schema arity {self.arity}")
        if signature.n_state != 0:
            raise ValueError("MapSchema requires a stateless cell")

    def output_width(self, width: int, signature: CellSignature) -> int:
        return width * signature.n_outputs

    def run(self, cell: Cell, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        self.check(cell.signature)
        batch, arity, width = operands.shape
        x = operands.permute(0, 2, 1).reshape(batch * width, arity)
        y, _ = cell.step(x, x.new_zeros(batch * width, 0))
        return quantizer(y).reshape(batch, -1)
