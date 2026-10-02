"""Tasks define reference semantics; they never participate in execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

from vectorpro.cells.base import CellSignature

Bits = tuple[int, ...]


@dataclass(frozen=True)
class LocalRule:
    """Ground-truth local transition, used only as a verification oracle.

    ``fn(inputs, state) -> (outputs, next_state)`` over 0/1 tuples.
    """

    name: str
    signature: CellSignature
    fn: Callable[[Bits, Bits], tuple[Bits, Bits]]


class Task(ABC):
    name: str
    arity: int
    local_rule: LocalRule | None = None

    @abstractmethod
    def reference(self, operands: Sequence[int], width: int) -> int:
        """Ground-truth result for ``width``-bit unsigned operands."""

    @abstractmethod
    def output_width(self, width: int) -> int: ...
