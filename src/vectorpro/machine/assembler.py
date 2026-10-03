"""Turning symbolic descriptions into vector programs, and vector programs back into text.

``assemble`` is to vector programs what an assembler is to machine code: a
convenience for writing them down. ``compile_expression`` is what the learner
uses to store a composition it found. ``disassemble`` reads a program's
tensors back into text for humans; it uses nothing but the tensors and the
resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import torch

from vectorpro.expr import Const, Expr, Var
from vectorpro.machine.kernel import Resolver
from vectorpro.machine.program import CONSTANTS, MAX_ARGS, VectorProgram

NEXT, HALT = "next", "halt"


@dataclass(frozen=True)
class Instr:
    op: str | None = None           # capability name; None for a branch-only step
    args: tuple[str, ...] = ()
    dest: str | None = None
    test: str | None = None         # register tested for non-zero; None = unconditional
    then: str = NEXT                # target label, NEXT or HALT
    otherwise: str = NEXT
    label: str | None = None


def _one_hot(index: int, size: int) -> torch.Tensor:
    v = torch.zeros(size)
    v[index] = 1
    return v


def assemble(
    instrs: Sequence[Instr],
    inputs: Sequence[str],
    registers: Mapping[str, str],
    output: str,
    key_of: Callable[[str], torch.Tensor],
) -> VectorProgram:
    """``registers`` maps every register name to its initial constant (inputs override it)."""
    names = list(registers)
    reg = {n: i for i, n in enumerate(names)}
    n_regs, n_steps = len(names), len(instrs)
    labels = {ins.label: i for i, ins in enumerate(instrs) if ins.label}
    if reserved := labels.keys() & {NEXT, HALT}:
        raise ValueError(f"labels {sorted(reserved)} are reserved")
    named = [i.op for i in instrs if i.op]
    dim = key_of(named[0]).shape[0] if named else 1  # keys of branch-only steps are never read

    def target(t: str, s: int) -> int:
        if t == HALT:
            return n_steps
        if t == NEXT:
            return s + 1
        return labels[t]

    keys, reads, writes, cond, nt, nf = [], [], [], [], [], []
    for s, ins in enumerate(instrs):
        keys.append(key_of(ins.op) if ins.op else torch.zeros(dim))
        r = torch.zeros(MAX_ARGS, n_regs)
        for k, a in enumerate(ins.args):
            r[k, reg[a]] = 1
        reads.append(r)
        writes.append(_one_hot(reg[ins.dest] if ins.dest else n_regs, n_regs + 1))
        cond.append(_one_hot(reg[ins.test] if ins.test else n_regs, n_regs + 1))
        nt.append(_one_hot(target(ins.then, s), n_steps + 1))
        nf.append(_one_hot(target(ins.otherwise, s), n_steps + 1))

    return VectorProgram(
        keys=torch.stack(keys),
        reads=torch.stack(reads),
        writes=torch.stack(writes),
        cond=torch.stack(cond),
        next_true=torch.stack(nt),
        next_false=torch.stack(nf),
        inputs=torch.stack([_one_hot(reg[i], n_regs) for i in inputs]),
        init=torch.stack([_one_hot(CONSTANTS.index(registers[n]), len(CONSTANTS)) for n in names]),
        output=_one_hot(reg[output], n_regs),
    )


def compile_expression(
    expr: Expr,
    arity: int,
    key_of: Callable[[str], torch.Tensor],
) -> VectorProgram:
    """Straight-line program: one register per operand, constant and operator node (post-order)."""
    instrs: list[Instr] = []
    registers = {f"x{i}": "zero" for i in range(arity)}

    def emit(e: Expr) -> str:
        if isinstance(e, Var):
            return f"x{e.index}"
        if isinstance(e, Const):
            registers.setdefault(f"c_{e.name}", e.name)
            return f"c_{e.name}"
        args = tuple(emit(a) for a in e.args)
        dest = f"t{len(instrs)}"
        registers[dest] = "zero"
        instrs.append(Instr(e.op, args, dest))
        return dest

    result = emit(expr)
    if not instrs:  # a bare leaf: a branch-only step so the program has one
        instrs.append(Instr())
    instrs[-1] = replace(instrs[-1], then=HALT, otherwise=HALT)
    return assemble(instrs, [f"x{i}" for i in range(arity)], registers, result, key_of)


def disassemble(program: VectorProgram, resolver: Resolver) -> list[str]:
    """Human-readable listing, reconstructed from the tensors alone."""
    n_regs, n_steps = program.n_registers, program.n_steps
    inputs = {int(row.argmax()): a for a, row in enumerate(program.inputs)}

    def reg_name(i: int) -> str:
        return f"r{i}"

    def step_name(i: int) -> str:
        return "halt" if i == n_steps else f"@{i}"

    lines = []
    for r in range(n_regs):
        start = f"input x{inputs[r]}" if r in inputs else CONSTANTS[int(program.init[r].argmax())]
        lines.append(f"{reg_name(r)} = {start}")
    for s in range(n_steps):
        dest = int(program.writes[s].argmax())
        if dest < n_regs:
            op = resolver.resolve(program.keys[s])
            args = ", ".join(reg_name(int(program.reads[s, k].argmax())) for k in range(op.arity))
            text = f"{reg_name(dest)} = {resolver.name_of(program.keys[s])}({args})"
        else:
            text = "(test only)"
        test = int(program.cond[s].argmax())
        t, f = int(program.next_true[s].argmax()), int(program.next_false[s].argmax())
        if test == n_regs:
            flow = "" if t == s + 1 else f"; goto {step_name(t)}"
        else:
            flow = f"; if {reg_name(test)} != 0 goto {step_name(t)} else {step_name(f)}"
        lines.append(f"@{s}: {text}{flow}")
    lines.append(f"result = {reg_name(int(program.output.argmax()))}")
    return lines
