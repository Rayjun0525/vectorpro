"""Expression trees: the AST shared by execution (programs) and reference semantics (tasks).

Nodes: ``Var`` (an operand), ``Const`` (a width-generic constant such as
``zero`` or ``ones``), ``BinOp`` (a binary operator, rendered infix) and
``Call`` (an operator of any arity).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar, Union

T = TypeVar("T")


@dataclass(frozen=True)
class Var:
    index: int


@dataclass(frozen=True)
class Const:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: str
    left: Expr
    right: Expr

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Call:
    op: str
    args: tuple[Expr, ...]


Expr = Union[Var, Const, BinOp, Call]


def fold_expr(
    expr: Expr,
    var: Callable[[int], T],
    op: Callable[..., T],
    const: Callable[[str], T] | None = None,
) -> T:
    """Evaluate ``expr`` bottom-up as ``op(name, *argument_values)``.

    Shared by tensor execution and reference semantics.
    """
    if isinstance(expr, Var):
        return var(expr.index)
    if isinstance(expr, Const):
        if const is None:
            raise ValueError(f"expression uses constant {expr.name!r} but no const handler given")
        return const(expr.name)
    return op(expr.op, *(fold_expr(a, var, op, const) for a in expr.args))


def render(expr: Expr, var_names: Sequence[str] | None = None) -> str:
    """Text form; operands are ``x0, x1, ...`` unless ``var_names`` says otherwise."""
    if isinstance(expr, Var):
        return var_names[expr.index] if var_names else f"x{expr.index}"
    if isinstance(expr, Const):
        return expr.name
    if isinstance(expr, BinOp):
        return f"({render(expr.left, var_names)} {expr.op} {render(expr.right, var_names)})"
    return f"{expr.op}({', '.join(render(a, var_names) for a in expr.args)})"


def expr_to_data(expr: Expr) -> dict:
    """JSON-serializable form; inverse of ``expr_from_data``."""
    if isinstance(expr, Var):
        return {"var": expr.index}
    if isinstance(expr, Const):
        return {"const": expr.name}
    if isinstance(expr, BinOp):
        return {"op": expr.op, "left": expr_to_data(expr.left), "right": expr_to_data(expr.right)}
    return {"op": expr.op, "args": [expr_to_data(a) for a in expr.args]}


def expr_from_data(data: dict) -> Expr:
    if "var" in data:
        return Var(data["var"])
    if "const" in data:
        return Const(data["const"])
    if "args" in data:
        return Call(data["op"], tuple(expr_from_data(a) for a in data["args"]))
    return BinOp(data["op"], expr_from_data(data["left"]), expr_from_data(data["right"]))


def operators(expr: Expr) -> set[str]:
    return fold_expr(expr, lambda _: set(), lambda o, *a: set().union({o}, *a), lambda _: set())


def count_ops(expr: Expr) -> int:
    return fold_expr(expr, lambda _: 0, lambda _, *a: 1 + sum(a), lambda _: 0)


def random_expression(
    rng: random.Random, n_vars: int, depth: int, ops: Sequence[str], p_leaf: float = 0.3
) -> Expr:
    """Random binary tree of at most ``depth`` operator levels; the root is always an operator."""

    def grow(level: int, root: bool) -> Expr:
        if level == 0 or (not root and rng.random() < p_leaf):
            return Var(rng.randrange(n_vars))
        return BinOp(rng.choice(ops), grow(level - 1, False), grow(level - 1, False))

    return grow(depth, True)
