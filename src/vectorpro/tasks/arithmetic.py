from __future__ import annotations

from typing import Sequence

from vectorpro.cells.base import CellSignature
from vectorpro.tasks.base import Bits, LocalRule, Task


def _full_adder(inputs: Bits, state: Bits) -> tuple[Bits, Bits]:
    total = inputs[0] + inputs[1] + state[0]
    return (total & 1,), (total >> 1,)


FULL_ADDER = LocalRule("full_adder", CellSignature(n_inputs=2, n_state=1, n_outputs=1), _full_adder)


class Addition(Task):
    """Unsigned addition of two operands; result is one bit wider."""

    name = "add"
    arity = 2
    local_rule = FULL_ADDER

    def reference(self, operands: Sequence[int], width: int) -> int:
        return operands[0] + operands[1]

    def output_width(self, width: int) -> int:
        return width + 1


class ModularSum(Task):
    """Sum of ``arity`` operands, wrapping modulo ``2**width``."""

    name = "modular_sum"

    def __init__(self, arity: int) -> None:
        self.arity = arity

    def reference(self, operands: Sequence[int], width: int) -> int:
        return sum(operands) % (1 << width)

    def output_width(self, width: int) -> int:
        return width
