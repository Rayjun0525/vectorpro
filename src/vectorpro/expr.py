"""Expression trees: the AST shared by execution (programs) and reference semantics (tasks)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar, Union

T = TypeVar("T")


@dataclass(frozen=True)
class Var:
    index: int


@dataclass(frozen=True)
class BinOp:
    op: str
    left: Expr
    right: Expr


Expr = Union[Var, BinOp]


def fold_expr(expr: Expr, var: Callable[[int], T], op: Callable[[str, T, T], T]) -> T:
    """Evaluate ``expr`` bottom-up; shared by tensor execution and reference semantics."""
    if isinstance(expr, Var):
        return var(expr.index)
    return op(expr.op, fold_expr(expr.left, var, op), fold_expr(expr.right, var, op))


def render(expr: Expr) -> str:
    return fold_expr(expr, lambda i: f"x{i}", lambda o, a, b: f"({a} {o} {b})")


def expr_to_data(expr: Expr) -> dict:
    """JSON-serializable form; inverse of ``expr_from_data``."""
    return fold_expr(expr, lambda i: {"var": i}, lambda o, a, b: {"op": o, "left": a, "right": b})


def expr_from_data(data: dict) -> Expr:
    if "var" in data:
        return Var(data["var"])
    return BinOp(data["op"], expr_from_data(data["left"]), expr_from_data(data["right"]))


def operators(expr: Expr) -> set[str]:
    return fold_expr(expr, lambda _: set(), lambda o, a, b: a | b | {o})


def count_ops(expr: Expr) -> int:
    return fold_expr(expr, lambda _: 0, lambda _, a, b: a + b + 1)


def random_expression(
    rng: random.Random, n_vars: int, depth: int, ops: Sequence[str], p_leaf: float = 0.3
) -> Expr:
    """Random tree of at most ``depth`` operator levels; the root is always an operator."""

    def grow(level: int, root: bool) -> Expr:
        if level == 0 or (not root and rng.random() < p_leaf):
            return Var(rng.randrange(n_vars))
        return BinOp(rng.choice(ops), grow(level - 1, False), grow(level - 1, False))

    return grow(depth, True)
