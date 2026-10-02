"""Lookup-table cells: the exact, inspectable form of a binary local rule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vectorpro.cells.base import Cell, CellSignature, binary_domain
from vectorpro.quantize import HardThreshold, Quantizer

if TYPE_CHECKING:
    from vectorpro.tasks.base import LocalRule


class TableCell(Cell):
    """Cell defined by a ``(domain_size, n_out_total)`` table of 0/1.

    Inputs are thresholded to bits and looked up by their canonical index
    (see ``binary_domain``).
    """

    def __init__(self, signature: CellSignature, table: torch.Tensor) -> None:
        expected = (signature.domain_size, signature.n_out_total)
        if tuple(table.shape) != expected:
            raise ValueError(f"table shape {tuple(table.shape)} != {expected}")
        self._signature = signature
        self.table = table.float()
        self._weights = 1 << torch.arange(signature.n_in_total)

    @classmethod
    def from_cell(cls, cell: Cell, quantizer: Quantizer | None = None) -> TableCell:
        """Tabulate any cell over its full binary domain."""
        quantizer = quantizer or HardThreshold()
        sig = cell.signature
        domain = binary_domain(sig)
        with torch.no_grad():
            y, s = cell.step(domain[:, : sig.n_inputs], domain[:, sig.n_inputs :])
            table = quantizer(torch.cat([y, s], dim=-1))
        return cls(sig, table)

    @classmethod
    def from_rule(cls, rule: LocalRule) -> TableCell:
        sig = rule.signature
        rows = []
        for bits in binary_domain(sig).long().tolist():
            out, nxt = rule.fn(tuple(bits[: sig.n_inputs]), tuple(bits[sig.n_inputs :]))
            rows.append(list(out) + list(nxt))
        return cls(sig, torch.tensor(rows))

    @property
    def signature(self) -> CellSignature:
        return self._signature

    def step(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bits = (torch.cat([x, state], dim=-1) >= 0.5).long()
        row = self.table[(bits * self._weights).sum(dim=-1)]
        n_out = self._signature.n_outputs
        return row[..., :n_out], row[..., n_out:]
