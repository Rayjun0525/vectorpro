from __future__ import annotations

import operator
from typing import Callable, Sequence

from vectorpro.cells.base import CellSignature
from vectorpro.tasks.base import LocalRule, Task


class Bitwise(Task):
    """Position-wise binary operation; the local rule is the op on single bits."""

    arity = 2

    def __init__(self, name: str, op: Callable[[int, int], int]) -> None:
        self.name = name
        self.op = op
        self.local_rule = LocalRule(
            name, CellSignature(2, 0, 1), lambda x, s: ((op(x[0], x[1]),), ())
        )

    def reference(self, operands: Sequence[int], width: int) -> int:
        return self.op(operands[0], operands[1])

    def output_width(self, width: int) -> int:
        return width


AND = Bitwise("and", operator.and_)
OR = Bitwise("or", operator.or_)
XOR = Bitwise("xor", operator.xor)
