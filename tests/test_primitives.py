"""M1 operations and programs, checked with exact (non-learned) rules."""

import random

import pytest

from vectorpro.cells import CellSignature, MLPCell, TableCell
from vectorpro.data import edge_pairs, random_tuples
from vectorpro.programs import Fold, ShiftAddMultiply
from vectorpro.schemas import MapSchema, ScanSchema
from vectorpro.tasks import (
    AND,
    FULL_ADDER,
    FULL_SUBTRACTOR,
    LESS_THAN,
    OR,
    XOR,
    LessThan,
    Multiplication,
    Subtraction,
)
from vectorpro.units import FunctionUnit, iter_units


def exact_scan(name, rule):
    return FunctionUnit(name, ScanSchema(2, (0,)), TableCell.from_rule(rule))


def exact_map(task):
    return FunctionUnit(task.name, MapSchema(2), TableCell.from_rule(task.local_rule))


def operands(width, n=300):
    return random_tuples(2, width, n, random.Random(width)) + edge_pairs(width)


@pytest.mark.parametrize("width", [1, 4, 64])
@pytest.mark.parametrize(
    "unit, task",
    [
        (exact_scan("sub", FULL_SUBTRACTOR), Subtraction()),
        (exact_scan("lt", LESS_THAN), LessThan()),
        (exact_map(AND), AND),
        (exact_map(OR), OR),
        (exact_map(XOR), XOR),
    ],
    ids=["sub", "lt", "and", "or", "xor"],
)
def test_exact_rules_match_reference(unit, task, width):
    ops = operands(width)
    assert unit.output_width(width) == task.output_width(width)
    assert unit(ops, width) == [task.reference(t, width) for t in ops]


@pytest.mark.parametrize("width", [1, 4, 32])
def test_shift_add_multiply(width):
    program = ShiftAddMultiply(exact_map(AND), exact_scan("add", FULL_ADDER))
    ops = operands(width, n=100)
    assert program(ops, width) == [Multiplication().reference(t, width) for t in ops]


def test_programs_expose_units_and_parameters():
    gate = FunctionUnit("gate", MapSchema(2), MLPCell(CellSignature(2, 0, 1)))
    adder = FunctionUnit("adder", ScanSchema(2, (0,)), MLPCell(FULL_ADDER.signature))
    program = Fold(ShiftAddMultiply(gate, adder), 3)

    assert [u.name for u in iter_units(program)] == ["gate", "adder"]
    expected = len(list(gate.cell.parameters())) + len(list(adder.cell.parameters()))
    assert len(program.parameters()) == expected
    assert all(isinstance(u.cell, TableCell) for u in iter_units(program.compiled()))


def test_shared_unit_counted_once():
    adder = FunctionUnit("adder", ScanSchema(2, (0,)), MLPCell(FULL_ADDER.signature))
    program = ShiftAddMultiply(adder, adder)
    assert len(list(iter_units(program))) == 1
    assert len(program.parameters()) == len(list(adder.cell.parameters()))
