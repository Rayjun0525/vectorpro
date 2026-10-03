"""The vector machine: programs as tensors, run by a task-agnostic kernel."""

from vectorpro.machine.assembler import HALT, NEXT, Instr, assemble, compile_expression, disassemble
from vectorpro.machine.kernel import BudgetExceeded, ProgramExecutable, Resolver, RunResult, run
from vectorpro.machine.program import CONSTANTS, VectorProgram, constant_patterns

__all__ = [
    "CONSTANTS",
    "HALT",
    "NEXT",
    "BudgetExceeded",
    "Instr",
    "ProgramExecutable",
    "Resolver",
    "RunResult",
    "VectorProgram",
    "assemble",
    "compile_expression",
    "constant_patterns",
    "disassemble",
    "run",
]
