"""Function units: a schema bound to a cell, plus what is known about it."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from vectorpro.cells import Cell, TableCell
from vectorpro.execution import BitExecutable
from vectorpro.quantize import Quantizer
from vectorpro.schemas import Schema
from vectorpro.verification import VerificationRecord


@dataclass
class FunctionUnit(BitExecutable):
    name: str
    schema: Schema
    cell: Cell
    verification: VerificationRecord | None = None

    def __post_init__(self) -> None:
        self.schema.check(self.cell.signature)

    @property
    def arity(self) -> int:  # type: ignore[override]
        return self.schema.arity

    def output_width(self, width: int) -> int:
        return self.schema.output_width(width, self.cell.signature)

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        return self.schema.run(self.cell, operands, quantizer)

    def compiled(self) -> FunctionUnit:
        """Same unit with its cell replaced by the tabulated binary rule."""
        return replace(self, name=f"{self.name}[table]", cell=TableCell.from_cell(self.cell))
