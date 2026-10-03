from __future__ import annotations

from typing import Mapping, Sequence

from vectorpro.expr import Expr, fold_expr, render
from vectorpro.tasks.arithmetic import MODULAR_OPS, OpSemantics
from vectorpro.tasks.base import Task


class ExpressionTask(Task):
    """Reference value of an expression tree under per-operator semantics."""

    def __init__(
        self, expr: Expr, arity: int, semantics: Mapping[str, OpSemantics] = MODULAR_OPS
    ) -> None:
        self.expr = expr
        self.arity = arity
        self.semantics = semantics
        self.name = render(expr)

    def reference(self, operands: Sequence[int], width: int) -> int:
        return fold_expr(
            self.expr, lambda i: operands[i], lambda o, a, b: self.semantics[o](a, b, width)
        )

    def output_width(self, width: int) -> int:
        return width
