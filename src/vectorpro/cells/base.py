"""The local state-transition rule abstraction.

A cell is the small, shared rule a schema applies repeatedly. At each step it
reads ``n_inputs`` operand bits plus ``n_state`` carried bits and produces
``n_outputs`` result bits plus the next state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CellSignature:
    n_inputs: int
    n_state: int
    n_outputs: int

    @property
    def n_in_total(self) -> int:
        return self.n_inputs + self.n_state

    @property
    def n_out_total(self) -> int:
        return self.n_outputs + self.n_state

    @property
    def domain_size(self) -> int:
        """Number of distinct binary (input, state) combinations."""
        return 1 << self.n_in_total


def binary_domain(signature: CellSignature) -> torch.Tensor:
    """All binary ``concat(input, state)`` rows, row ``k`` = bits of ``k`` (LSB first).

    The row order defines the canonical table index used by ``TableCell``.
    """
    k = torch.arange(signature.domain_size).unsqueeze(1)
    shifts = torch.arange(signature.n_in_total).unsqueeze(0)
    return ((k >> shifts) & 1).float()


class Cell(ABC):
    @property
    @abstractmethod
    def signature(self) -> CellSignature: ...

    @abstractmethod
    def step(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x: (B, n_inputs)``, ``state: (B, n_state)`` -> ``(outputs, next_state)``.

        All values lie in [0, 1].
        """
