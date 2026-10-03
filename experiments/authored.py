"""Hand-authored vector programs (the vector-machine equivalent of assembly).

These exist to show the format can hold loops, branches and calls; they are
written by a person, not learned. They call capabilities only by name here,
which ``assemble`` turns into address vectors; the stored program holds no names.

Capabilities used: and, or, add, sub, lt, shl, shr. Registers are W bits;
results are W-bit (``uintW_t``) semantics, with RISC-V division by zero.
"""

from __future__ import annotations

from typing import Callable

import torch

from vectorpro.machine import HALT, Instr, VectorProgram, assemble

KeyOf = Callable[[str], torch.Tensor]


def mul(key_of: KeyOf) -> VectorProgram:
    """Shift-and-add: while b != 0: if b & 1: acc += a; a <<= 1; b >>= 1."""
    return assemble(
        [
            Instr("and", ("b", "one"), "t", test="t", then="add", otherwise="shift", label="loop"),
            Instr("add", ("acc", "a"), "acc", label="add"),
            Instr("shl", ("a",), "a", label="shift"),
            Instr("shr", ("b",), "b", test="b", then="loop", otherwise=HALT),
        ],
        inputs=["a", "b"],
        registers={"a": "zero", "b": "zero", "acc": "zero", "one": "one", "t": "zero"},
        output="acc",
        key_of=key_of,
    )


def divmod_program(key_of: KeyOf, result: str) -> VectorProgram:
    """Restoring division over a moving one-bit mask, from the top bit down.

    ``carry`` remembers that ``r`` overflowed W bits when shifted; then ``r >= b``
    holds and the W-bit subtraction is still exact.
    """
    if result not in ("q", "r"):
        raise ValueError("result is 'q' (quotient) or 'r' (remainder)")
    return assemble(
        [
            Instr("and", ("r", "top"), "carry", label="loop"),
            Instr("shl", ("r",), "r"),
            Instr("and", ("a", "m"), "t", test="t", then="setbit", otherwise="check"),
            Instr("or", ("r", "one"), "r", label="setbit"),
            Instr(test="carry", then="take", otherwise="compare", label="check"),
            Instr("lt", ("r", "b"), "t", test="t", then="advance", otherwise="take", label="compare"),
            Instr("sub", ("r", "b"), "r", label="take"),
            Instr("or", ("q", "m"), "q"),
            Instr("shr", ("m",), "m", test="m", then="loop", otherwise=HALT, label="advance"),
        ],
        inputs=["a", "b"],
        registers={"a": "zero", "b": "zero", "q": "zero", "r": "zero", "m": "top",
                   "top": "top", "one": "one", "carry": "zero", "t": "zero"},
        output=result,
        key_of=key_of,
    )
