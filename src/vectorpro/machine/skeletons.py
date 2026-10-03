"""Generic loop skeletons: control structures the machine provides once, for every task.

``BitFold`` walks the bits of one operand, highest first or lowest first,
carrying a result accumulator and optionally companion accumulators, all
updated together from their previous values:

    acc = 0; c1 = 0; ...
    for each bit b of x[over]:
        acc, c1, ... = body(x0.., acc, bit, mask, c1..), companion_1(x0.., c1, bit, mask), ...

``bit`` is 0/1 and ``mask`` is 0 or all ones. With ``headroom`` the
registers carry extra high bits (operands are zero-extended, so the walk also
visits those leading zero bits) and the result is cut back to ``W`` bits.

The bodies are expressions over capabilities; the skeleton fixes only the
iteration. ``compile_bit_fold`` turns a fold into an ordinary vector program
whose loop, bit extraction and branch are tensors calling capabilities by
address. The capabilities it needs for bit extraction (``FoldPrimitives``)
are supplied by the caller, typically found in the registry by behaviour.

Leaf indices in bodies: operands ``0..A-1``, ``acc`` = A, ``bit`` = A+1,
``mask`` = A+2, companion ``i`` = A+3+i. A companion body reads its own
accumulator through its companion index and never reads ``acc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from vectorpro.expr import BinOp, Call, Const, Expr, Var, expr_from_data, expr_to_data, render
from vectorpro.machine.assembler import HALT, Instr, assemble
from vectorpro.machine.program import VectorProgram
from vectorpro.schemas import Direction


@dataclass(frozen=True)
class FoldPrimitives:
    """Capability names used to extract the current bit; resolved to address vectors at assembly."""

    test_and: str    # (x, m) -> x & m
    to_bit: str      # (0, t) -> 1 if t != 0 else 0   (i.e. 0 < t)
    to_mask: str     # (0, bit) -> 0 - bit            (0 or all ones)
    shift_down: str  # m -> m >> 1   (moves the mask for highest-first folds)
    shift_up: str    # m -> m << 1   (moves the mask for lowest-first folds)


def body_leaves(arity: int) -> tuple[int, int, int]:
    """Indices of the acc, bit and mask leaves in a fold body."""
    return arity, arity + 1, arity + 2


def companion_leaf(arity: int, i: int) -> int:
    return arity + 3 + i


def remap(expr: Expr, mapping: Mapping[int, int]) -> Expr:
    """Rename leaves: ``Var(k)`` becomes ``Var(mapping.get(k, k))``."""
    if isinstance(expr, Var):
        return Var(mapping.get(expr.index, expr.index))
    if isinstance(expr, Const):
        return expr
    args = tuple(remap(a, mapping) for a in expr.args)
    return BinOp(expr.op, *args) if isinstance(expr, BinOp) else Call(expr.op, args)


@dataclass(frozen=True)
class BitFold:
    arity: int
    over: int
    direction: Direction
    body: Expr
    headroom: int = 0
    companions: tuple[Expr, ...] = ()

    def leaf_names(self) -> list[str]:
        names = [f"x{i}" for i in range(self.arity)] + ["acc", "bit", "mask"]
        return names + [f"c{i + 1}" for i in range(len(self.companions))]

    def as_companion(self, slot: int) -> Expr:
        """This fold's body rewritten to run as companion ``slot`` of another fold."""
        return remap(self.body, {body_leaves(self.arity)[0]: companion_leaf(self.arity, slot)})

    def describe(self) -> str:
        order = "highest to lowest" if self.direction is Direction.MSB_FIRST else "lowest to highest"
        names = self.leaf_names()
        accs = ["acc"] + [f"c{i + 1}" for i in range(len(self.companions))]
        updates = [f"acc = {render(self.body, names)}"]
        updates += [f"c{i + 1} = {render(c, names)}" for i, c in enumerate(self.companions)]
        room = f" with {self.headroom} extra bit(s)" if self.headroom else ""
        return (f"{' = '.join(accs)} = 0; for each bit of x{self.over} ({order}){room}: "
                + "; ".join(updates))

    def to_data(self) -> dict:
        return {"arity": self.arity, "over": self.over, "direction": self.direction.value,
                "body": expr_to_data(self.body), "headroom": self.headroom,
                "companions": [expr_to_data(c) for c in self.companions]}

    @classmethod
    def from_data(cls, data: dict) -> BitFold:
        return cls(data["arity"], data["over"], Direction(data["direction"]), expr_from_data(data["body"]),
                   data.get("headroom", 0), tuple(expr_from_data(c) for c in data.get("companions", ())))


def _reads(expr: Expr, index: int) -> bool:
    if isinstance(expr, Var):
        return expr.index == index
    if isinstance(expr, Const):
        return False
    return any(_reads(a, index) for a in expr.args)


def compile_bit_fold(
    fold: BitFold, key_of: Callable[[str], torch.Tensor], primitives: FoldPrimitives
) -> VectorProgram:
    arity = fold.arity
    acc_i, bit_i, mask_i = body_leaves(arity)
    inputs = [f"x{i}" for i in range(arity)]
    msb_first = fold.direction is Direction.MSB_FIRST
    registers = {name: "zero" for name in inputs}
    registers.update({"acc": "zero", "bit": "zero", "mask": "zero", "t": "zero",
                      "c_zero": "zero", "m": "top" if msb_first else "one"})
    leaf = {acc_i: "acc", bit_i: "bit", mask_i: "mask"}
    for i in range(len(fold.companions)):
        leaf[companion_leaf(arity, i)] = f"c{i + 1}"
        registers[f"c{i + 1}"] = "zero"
    body: list[Instr] = []
    computed: dict[Expr, str] = {}  # one register per distinct sub-expression in an iteration

    def emit(e: Expr, dest: str | None = None) -> str:
        if isinstance(e, Var):
            return leaf.get(e.index, f"x{e.index}")
        if isinstance(e, Const):
            registers.setdefault(f"c_{e.name}", e.name)
            return f"c_{e.name}"
        if dest is None and e in computed:
            return computed[e]
        args = tuple(emit(a) for a in e.args)
        target = dest or f"t{len(body)}"
        registers.setdefault(target, "zero")
        body.append(Instr(e.op, args, target))
        if dest is None:  # accumulators change within the iteration; temporaries do not
            computed[e] = target
        return target

    for e in (fold.body, *fold.companions):
        if isinstance(e, (Var, Const)):
            raise ValueError("a fold body must call at least one capability")
    # The result body reads companions' previous values, so it runs first;
    # each companion reads only its own accumulator, which it overwrites last.
    emit(fold.body, dest="acc")
    for i, companion in enumerate(fold.companions):
        emit(companion, dest=f"c{i + 1}")

    bodies = (fold.body, *fold.companions)
    uses_mask = any(_reads(e, mask_i) for e in bodies)
    uses_bit = uses_mask or any(_reads(e, bit_i) for e in bodies)
    instrs = [
        Instr(primitives.test_and, (f"x{fold.over}", "m"), "t", label="loop"),
        *([Instr(primitives.to_bit, ("c_zero", "t"), "bit")] if uses_bit else []),
        *([Instr(primitives.to_mask, ("c_zero", "bit"), "mask")] if uses_mask else []),
        *body,
        Instr(primitives.shift_down if msb_first else primitives.shift_up, ("m",), "m",
              test="m", then="loop", otherwise=HALT),
    ]
    return assemble(instrs, inputs, registers, "acc", key_of, fold.headroom)
