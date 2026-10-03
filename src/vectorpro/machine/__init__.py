"""The vector machine: programs as tensors, run by a task-agnostic kernel."""

from vectorpro.machine.assembler import HALT, NEXT, Instr, assemble, compile_expression, disassemble
from vectorpro.machine.kernel import BudgetExceeded, ProgramExecutable, Resolver, RunResult, run
from vectorpro.machine.program import CONSTANTS, VectorProgram, constant_patterns
from vectorpro.machine.skeletons import BitFold, FoldPrimitives, compile_bit_fold

__all__ = [
    "CONSTANTS",
    "HALT",
    "NEXT",
    "BitFold",
    "BudgetExceeded",
    "FoldPrimitives",
    "Instr",
    "ProgramExecutable",
    "Resolver",
    "RunResult",
    "VectorProgram",
    "assemble",
    "compile_bit_fold",
    "compile_expression",
    "constant_patterns",
    "disassemble",
    "run",
]
