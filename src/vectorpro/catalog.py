"""Standard learnable primitive units and their training specs.

Each spec pairs a fresh, untrained unit with the task it learns from and the
local rule it is verified against.
"""

from __future__ import annotations

from vectorpro.benchmark import OperationSpec, WidthSuite
from vectorpro.cells import CellSignature, MLPCell
from vectorpro.schemas import MapSchema, ScanSchema
from vectorpro.tasks import (
    AND,
    FULL_ADDER,
    FULL_SUBTRACTOR,
    LESS_THAN,
    MUX,
    OR,
    XOR,
    Addition,
    LessThan,
    Subtraction,
    Task,
)
from vectorpro.units import FunctionUnit


def scan_unit(name: str, signature: CellSignature, initial_state: tuple[int, ...]) -> FunctionUnit:
    return FunctionUnit(name, ScanSchema(signature.n_inputs, initial_state), MLPCell(signature))


def map_unit(name: str, arity: int = 2) -> FunctionUnit:
    return FunctionUnit(name, MapSchema(arity), MLPCell(CellSignature(arity, 0, 1)))


def map_spec(task: Task, suite: WidthSuite | None = None) -> OperationSpec:
    return OperationSpec(
        task.name,
        task,
        lambda: map_unit(task.name, task.arity),
        {task.name: task.local_rule},
        suite or WidthSuite(),
    )


UNIT_SPECS: dict[str, OperationSpec] = {
    "add": OperationSpec("add", Addition(), lambda: scan_unit("add", FULL_ADDER.signature, (0,)),
                         {"add": FULL_ADDER}),
    "sub": OperationSpec("sub", Subtraction(), lambda: scan_unit("sub", FULL_SUBTRACTOR.signature, (0,)),
                         {"sub": FULL_SUBTRACTOR}),
    "lt": OperationSpec("lt", LessThan(), lambda: scan_unit("lt", LESS_THAN.signature, (0,)),
                        {"lt": LESS_THAN}),
    "and": map_spec(AND),
    "or": map_spec(OR),
    "xor": map_spec(XOR),
    # Three operands: exhaustive 8-bit would be 2**24 triples, so stop at 6 bits.
    "mux": map_spec(MUX, WidthSuite(exhaustive_width=6)),
}
