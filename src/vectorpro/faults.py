"""Fault injection for negative controls: flip one entry of a unit's binary table."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from vectorpro.cells import TableCell
from vectorpro.cells.base import binary_domain
from vectorpro.units import FunctionUnit


def with_fault(unit: FunctionUnit, row: int, column: int) -> FunctionUnit:
    """Table-compiled copy of ``unit`` with ``table[row, column]`` flipped."""
    table = TableCell.from_cell(unit.cell)
    flipped = table.table.clone()
    flipped[row, column] = 1 - flipped[row, column]
    return replace(unit, name=f"{unit.name}[fault {row},{column}]", cell=TableCell(table.signature, flipped))


def single_faults(unit: FunctionUnit) -> Iterator[tuple[str, FunctionUnit]]:
    """Every single-entry fault of ``unit``, labelled by the (input, state) row and output column."""
    sig = unit.cell.signature
    for row, bits in enumerate(binary_domain(sig).long().tolist()):
        for column in range(sig.n_out_total):
            label = f"in={bits[: sig.n_inputs]} state={bits[sig.n_inputs:]} out[{column}]"
            yield label, with_fault(unit, row, column)
