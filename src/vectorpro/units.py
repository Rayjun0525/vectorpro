"""Function units: a schema bound to a cell, plus what is known about it."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import torch
from torch import nn

from vectorpro.cells import Cell, TableCell
from vectorpro.execution import BitExecutable
from vectorpro.quantize import Quantizer
from vectorpro.schemas import Schema
from vectorpro.verification import VerificationRecord


@dataclass(eq=False)
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

    def parameters(self) -> list[nn.Parameter]:
        return list(self.cell.parameters()) if isinstance(self.cell, nn.Module) else []

    def compiled(self) -> FunctionUnit:
        """Same unit with its cell replaced by the tabulated binary rule."""
        return replace(self, name=f"{self.name}[table]", cell=TableCell.from_cell(self.cell))


def iter_units(executable: BitExecutable) -> Iterator[FunctionUnit]:
    """All distinct function units in an executable tree, depth first."""
    seen: set[int] = set()
    stack = [executable]
    while stack:
        node = stack.pop()
        if isinstance(node, FunctionUnit):
            if id(node) not in seen:
                seen.add(id(node))
                yield node
        else:
            stack.extend(reversed(node.children()))
