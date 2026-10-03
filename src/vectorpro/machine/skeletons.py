"""Generic loop skeletons: control structures the machine provides once, for every task.

``BitFold`` walks the bits of one operand, highest first or lowest first,
carrying one accumulator register:

    acc = 0
    for each bit b of x[over]:
        acc = body(x0, ..., acc, bit, mask)    # bit is 0/1, mask is 0 or all ones

The body is an expression over capabilities; the skeleton fixes only the
iteration. ``compile_bit_fold`` turns a fold into an ordinary vector program
whose loop, bit extraction and branch are tensors calling capabilities by
address. The capabilities it needs for bit extraction (``FoldPrimitives``)
are supplied by the caller, typically found in the registry by behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from vectorpro.expr import Const, Expr, Var, expr_from_data, expr_to_data, render
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


@dataclass(frozen=True)
class BitFold:
    arity: int
    over: int
    direction: Direction
    body: Expr

    def leaf_names(self) -> list[str]:
        return [f"x{i}" for i in range(self.arity)] + ["acc", "bit", "mask"]

    def describe(self) -> str:
        order = "highest to lowest" if self.direction is Direction.MSB_FIRST else "lowest to highest"
        return (f"acc = 0; for each bit of x{self.over} ({order}): "
                f"acc = {render(self.body, self.leaf_names())}")

    def to_data(self) -> dict:
        return {"arity": self.arity, "over": self.over, "direction": self.direction.value,
                "body": expr_to_data(self.body)}

    @classmethod
    def from_data(cls, data: dict) -> BitFold:
        return cls(data["arity"], data["over"], Direction(data["direction"]), expr_from_data(data["body"]))


def body_leaves(arity: int) -> tuple[int, int, int]:
    """Indices of the acc, bit and mask leaves in a fold body."""
    return arity, arity + 1, arity + 2


def compile_bit_fold(
    fold: BitFold, key_of: Callable[[str], torch.Tensor], primitives: FoldPrimitives
) -> VectorProgram:
    acc_i, bit_i, mask_i = body_leaves(fold.arity)
    inputs = [f"x{i}" for i in range(fold.arity)]
    msb_first = fold.direction is Direction.MSB_FIRST
    registers = {name: "zero" for name in inputs}
    registers.update({"acc": "zero", "bit": "zero", "mask": "zero", "t": "zero",
                      "c_zero": "zero", "m": "top" if msb_first else "one"})
    leaf = {acc_i: "acc", bit_i: "bit", mask_i: "mask"}
    body: list[Instr] = []

    def emit(e: Expr, dest: str | None = None) -> str:
        if isinstance(e, Var):
            return leaf.get(e.index, f"x{e.index}")
        if isinstance(e, Const):
            registers.setdefault(f"c_{e.name}", e.name)
            return f"c_{e.name}"
        args = tuple(emit(a) for a in e.args)
        dest = dest or f"t{len(body)}"
        registers.setdefault(dest, "zero")
        body.append(Instr(e.op, args, dest))
        return dest

    if isinstance(fold.body, (Var, Const)):
        raise ValueError("a fold body must call at least one capability")
    emit(fold.body, dest="acc")

    instrs = [
        Instr(primitives.test_and, (f"x{fold.over}", "m"), "t", label="loop"),
        Instr(primitives.to_bit, ("c_zero", "t"), "bit"),
        Instr(primitives.to_mask, ("c_zero", "bit"), "mask"),
        *body,
        Instr(primitives.shift_down if msb_first else primitives.shift_up, ("m",), "m",
              test="m", then="loop", otherwise=HALT),
    ]
    return assemble(instrs, inputs, registers, "acc", key_of)
