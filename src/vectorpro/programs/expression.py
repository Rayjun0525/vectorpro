"""Executes expression trees through bound ``W -> W`` operator executables."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from vectorpro.execution import BitExecutable
from vectorpro.expr import Expr, fold_expr, operators
from vectorpro.quantize import Quantizer


class ExpressionProgram(BitExecutable):
    """Executes an expression tree; each operator symbol is bound to a ``W -> W`` executable."""

    def __init__(self, expr: Expr, ops: Mapping[str, BitExecutable], arity: int) -> None:
        used = operators(expr)
        if missing := used - ops.keys():
            raise ValueError(f"no executable bound for operators {sorted(missing)}")
        self.expr = expr
        self.ops = {o: ops[o] for o in sorted(used)}
        self.arity = arity

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        return fold_expr(
            self.expr,
            lambda i: operands[:, i],
            lambda o, a, b: self.ops[o].execute(torch.stack([a, b], dim=1), quantizer),
        )

    def children(self) -> Sequence[BitExecutable]:
        return tuple(self.ops.values())

    def compiled(self) -> ExpressionProgram:
        return ExpressionProgram(self.expr, {k: v.compiled() for k, v in self.ops.items()}, self.arity)
