"""The execution kernel: a fixed, task-agnostic fetch-execute cycle over vector programs.

Per clock, every example sits at one step. The kernel resolves the step's
address vector to a capability, gathers arguments with the read matrix, writes
the result with the write matrix, tests the condition register and moves the
program counter. It holds no knowledge of any task; all of that is in the
program's tensors and in the capabilities they address.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from vectorpro.execution import BitExecutable, fit_width
from vectorpro.machine.program import VectorProgram, constant_patterns
from vectorpro.quantize import Quantizer


class Resolver(Protocol):
    """Maps an address vector to the capability it addresses."""

    def resolve(self, key: torch.Tensor) -> BitExecutable: ...

    def name_of(self, key: torch.Tensor) -> str: ...


@dataclass(frozen=True)
class RunResult:
    output: torch.Tensor   # (B, W)
    halted: torch.Tensor   # (B,) bool
    clocks: int


def run(
    program: VectorProgram,
    operands: torch.Tensor,
    resolver: Resolver,
    quantizer: Quantizer,
    budget: int | None = None,
) -> RunResult:
    """Execute ``program`` on ``operands: (B, arity, W)``."""
    batch, arity, out_width = operands.shape
    width = out_width + program.extra_bits  # register width
    if width != out_width:
        operands = torch.cat([operands, operands.new_zeros(batch, arity, width - out_width)], dim=2)
    n_steps, n_regs = program.n_steps, program.n_registers
    budget = budget if budget is not None else 16 * n_steps * (width + 2)

    regs = (program.init @ constant_patterns(width)).expand(batch, -1, -1).clone()
    loaded = program.inputs.sum(dim=0)  # (R,) 1 where an operand is loaded
    regs = regs * (1 - loaded)[None, :, None] + torch.einsum("ar,baw->brw", program.inputs, operands)

    ops = [resolver.resolve(program.keys[s]) if program.calls[s] > 0.5 else None
           for s in range(n_steps)]
    if batch != 1 and getattr(resolver, "has_effects", True) and any(
        op is not None and op.effects for op in ops
    ):
        raise ValueError("effectful programs require a single execution lane")

    pc = torch.zeros(batch, n_steps + 1)
    pc[:, 0] = 1
    clocks = 0
    while clocks < budget and pc[:, :n_steps].sum() > 0:
        clocks += 1
        new_pc = pc.clone()
        for s in range(n_steps):
            rows = pc[:, s] > 0.5
            if not rows.any():
                continue
            r = regs[rows]
            op = ops[s]
            if op is not None:
                args = torch.einsum("kr,brw->bkw", program.reads[s], r)[:, : op.arity]
                result = fit_width(op.execute(args, quantizer), width)
                w = program.writes[s, :n_regs]
                r = r * (1 - w)[None, :, None] + w[None, :, None] * result[:, None, :]
                regs[rows] = r
            c = program.cond[s]
            tested = torch.einsum("r,brw->bw", c[:n_regs], r).gt(0.5).any(dim=1)
            flag = tested | bool(c[n_regs] > 0.5)
            new_pc[rows] = torch.where(flag[:, None], program.next_true[s], program.next_false[s])
        pc = new_pc

    output = torch.einsum("r,brw->bw", program.output, regs)[:, :out_width]
    return RunResult(output, pc[:, n_steps] > 0.5, clocks)


class BudgetExceeded(RuntimeError):
    pass


class ProgramExecutable(BitExecutable):
    """A vector program bound to a resolver, usable wherever an executable is."""

    def __init__(self, program: VectorProgram, resolver: Resolver, budget: int | None = None) -> None:
        self.program = program
        self.resolver = resolver
        self.budget = budget
        self.arity = program.arity

    def output_width(self, width: int) -> int:
        return width

    def execute(self, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        result = run(self.program, operands, self.resolver, quantizer, self.budget)
        if not bool(result.halted.all()):
            stuck = int((~result.halted).sum())
            raise BudgetExceeded(f"{stuck} example(s) did not halt within {result.clocks} clocks")
        return result.output

    def children(self) -> Sequence[BitExecutable]:
        called = [self.resolver.resolve(k) for k, call in zip(self.program.keys, self.program.calls)
                  if call > 0.5]
        return list({id(c): c for c in called}.values())

    def compiled(self) -> ProgramExecutable:
        return self  # capabilities it calls are already stored in table form
