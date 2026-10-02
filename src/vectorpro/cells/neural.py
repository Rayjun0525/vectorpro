"""Learned cells."""

from __future__ import annotations

import torch
from torch import nn

from vectorpro.cells.base import Cell, CellSignature


class MLPCell(nn.Module, Cell):
    """Small MLP over ``concat(x, state)`` with sigmoid outputs."""

    def __init__(self, signature: CellSignature, hidden: int = 16, depth: int = 1) -> None:
        super().__init__()
        self._signature = signature
        layers: list[nn.Module] = []
        width = signature.n_in_total
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, signature.n_out_total))
        self.net = nn.Sequential(*layers)

    @property
    def signature(self) -> CellSignature:
        return self._signature

    def step(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.net(torch.cat([x, state], dim=-1)))
        n_out = self._signature.n_outputs
        return z[..., :n_out], z[..., n_out:]
