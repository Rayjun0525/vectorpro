"""Exhaustive verification of a cell's binary local rule.

Because a cell's binary domain is finite, it can be checked completely. If the
quantized cell matches the reference rule on every entry, then any width run
by a correct schema is correct by induction over positions. That guarantee
covers quantized execution only; continuous execution is tested, not proven.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vectorpro.cells.base import Cell, binary_domain
from vectorpro.cells.table import TableCell
from vectorpro.quantize import HardThreshold, Quantizer
from vectorpro.tasks.base import LocalRule


@dataclass(frozen=True)
class Mismatch:
    inputs: tuple[int, ...]
    state: tuple[int, ...]
    expected: tuple[int, ...]
    actual: tuple[int, ...]


@dataclass(frozen=True)
class VerificationRecord:
    rule: str
    domain_size: int
    mismatches: tuple[Mismatch, ...]

    @property
    def exact(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {**asdict(self), "exact": self.exact}


def verify_local_rule(
    cell: Cell, rule: LocalRule, quantizer: Quantizer | None = None
) -> VerificationRecord:
    sig = cell.signature
    if sig != rule.signature:
        raise ValueError(f"cell signature {sig} != rule signature {rule.signature}")
    actual = TableCell.from_cell(cell, quantizer or HardThreshold()).table.long()
    expected = TableCell.from_rule(rule).table.long()
    mismatches = []
    for row, bits in enumerate(binary_domain(sig).long().tolist()):
        if not (actual[row] == expected[row]).all():
            mismatches.append(
                Mismatch(
                    inputs=tuple(bits[: sig.n_inputs]),
                    state=tuple(bits[sig.n_inputs :]),
                    expected=tuple(expected[row].tolist()),
                    actual=tuple(actual[row].tolist()),
                )
            )
    return VerificationRecord(rule.name, sig.domain_size, tuple(mismatches))
