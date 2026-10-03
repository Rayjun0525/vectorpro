from __future__ import annotations

from typing import Sequence

from vectorpro.cells.base import CellSignature
from vectorpro.tasks.base import Bits, LocalRule, Task


def _full_adder(inputs: Bits, state: Bits) -> tuple[Bits, Bits]:
    total = inputs[0] + inputs[1] + state[0]
    return (total & 1,), (total >> 1,)


def _full_subtractor(inputs: Bits, state: Bits) -> tuple[Bits, Bits]:
    diff = inputs[0] - inputs[1] - state[0]
    return (diff & 1,), (int(diff < 0),)


def _less_than_lsb_first(inputs: Bits, state: Bits) -> tuple[Bits, Bits]:
    a, b = inputs
    # A higher differing bit overrides everything below it.
    return (), ((int(a < b) if a != b else state[0]),)


FULL_ADDER = LocalRule("full_adder", CellSignature(2, 1, 1), _full_adder)
FULL_SUBTRACTOR = LocalRule("full_subtractor", CellSignature(2, 1, 1), _full_subtractor)
LESS_THAN = LocalRule("less_than_lsb_first", CellSignature(2, 1, 0), _less_than_lsb_first)


class Addition(Task):
    """Unsigned addition of two operands; result is one bit wider."""

    name = "add"
    arity = 2
    local_rule = FULL_ADDER

    def reference(self, operands: Sequence[int], width: int) -> int:
        return operands[0] + operands[1]

    def output_width(self, width: int) -> int:
        return width + 1


class Subtraction(Task):
    """``(a - b) mod 2**W`` in the low bits, borrow-out (``a < b``) in the top bit."""

    name = "sub"
    arity = 2
    local_rule = FULL_SUBTRACTOR

    def reference(self, operands: Sequence[int], width: int) -> int:
        a, b = operands
        return ((a - b) % (1 << width)) | (int(a < b) << width)

    def output_width(self, width: int) -> int:
        return width + 1


class LessThan(Task):
    """Unsigned ``a < b`` as a single bit."""

    name = "lt"
    arity = 2
    local_rule = LESS_THAN

    def reference(self, operands: Sequence[int], width: int) -> int:
        return int(operands[0] < operands[1])

    def output_width(self, width: int) -> int:
        return 1


class Multiplication(Task):
    """Unsigned product; result is twice the operand width."""

    name = "mul"
    arity = 2

    def reference(self, operands: Sequence[int], width: int) -> int:
        return operands[0] * operands[1]

    def output_width(self, width: int) -> int:
        return 2 * width


class ModularSum(Task):
    """Sum of ``arity`` operands, wrapping modulo ``2**width``."""

    name = "modular_sum"

    def __init__(self, arity: int) -> None:
        self.arity = arity

    def reference(self, operands: Sequence[int], width: int) -> int:
        return sum(operands) % (1 << width)

    def output_width(self, width: int) -> int:
        return width
