"""Four-operation programs and expressions, checked with exact (non-learned) rules."""

import random

import pytest

from vectorpro.cells import TableCell
from vectorpro.data import edge_pairs, random_tuples
from vectorpro.expr import BinOp, Var, count_ops, random_expression, render
from vectorpro.faults import single_faults
from vectorpro.programs import ExpressionProgram, RestoringDivide, ShiftAddMultiply, Window
from vectorpro.schemas import MapSchema, ScanSchema
from vectorpro.tasks import (
    AND,
    FULL_ADDER,
    FULL_SUBTRACTOR,
    MODULAR_OPS,
    MUX,
    DivMod,
    ExpressionTask,
)
from vectorpro.units import FunctionUnit, iter_units


def scan(name, rule):
    return FunctionUnit(name, ScanSchema(2, (0,)), TableCell.from_rule(rule))


def mapped(name, rule, arity):
    return FunctionUnit(name, MapSchema(arity), TableCell.from_rule(rule))


ADD = scan("add", FULL_ADDER)
SUB = scan("sub", FULL_SUBTRACTOR)
GATE = mapped("and", AND.local_rule, 2)
SELECT = mapped("mux", MUX.local_rule, 3)
DIVIDE = RestoringDivide(SUB, SELECT)
OPS = {
    "+": Window(ADD),
    "-": Window(SUB),
    "*": Window(ShiftAddMultiply(GATE, ADD)),
    "/": Window(DIVIDE, word=0),
    "%": Window(DIVIDE, word=1),
}


def operands(width, n=200):
    return random_tuples(2, width, n, random.Random(width)) + edge_pairs(width)


def test_mux_rule():
    ops = random_tuples(3, 16, 200, random.Random(0))
    assert SELECT(ops, 16) == [MUX.reference(t, 16) for t in ops]


@pytest.mark.parametrize("width", [1, 4, 32])
def test_restoring_divide(width):
    ops = operands(width)
    assert any(b == 0 for _, b in ops)
    assert DIVIDE(ops, width) == [DivMod().reference(t, width) for t in ops]


@pytest.mark.parametrize("symbol", sorted(OPS))
def test_windows_give_modular_semantics(symbol):
    ops = operands(16)
    assert OPS[symbol](ops, 16) == [MODULAR_OPS[symbol](a, b, 16) for a, b in ops]


def test_expression_program_matches_task():
    rng = random.Random(7)
    for _ in range(20):
        expr = random_expression(rng, n_vars=3, depth=3, ops=sorted(OPS))
        program, task = ExpressionProgram(expr, OPS, 3), ExpressionTask(expr, 3)
        inputs = random_tuples(3, 12, 50, rng) + random_tuples(3, 3, 50, rng)
        assert program(inputs, 12) == [task.reference(t, 12) for t in inputs], render(expr)


def test_expression_helpers():
    expr = BinOp("/", BinOp("+", Var(0), Var(1)), Var(0))
    assert render(expr) == "((x0 + x1) / x0)"
    assert count_ops(expr) == 2
    assert ExpressionTask(expr, 2).reference((3, 4), 8) == 2
    assert ExpressionTask(expr, 2).reference((0, 4), 8) == 255


def test_expression_rejects_unbound_operator():
    with pytest.raises(ValueError):
        ExpressionProgram(BinOp("^", Var(0), Var(1)), OPS, 2)


@pytest.mark.parametrize(
    "expr, units",
    [
        (BinOp("%", Var(0), Var(1)), {"sub", "mux"}),
        (BinOp("*", Var(0), BinOp("-", Var(1), Var(0))), {"and", "add", "sub"}),
    ],
)
def test_expression_uses_only_its_operators(expr, units):
    assert {u.name for u in iter_units(ExpressionProgram(expr, OPS, 2))} == units


def test_single_faults_cover_every_entry_and_break_the_operation():
    faults = list(single_faults(ADD))
    assert len(faults) == FULL_ADDER.signature.domain_size * FULL_ADDER.signature.n_out_total
    ops = operands(8)
    expected = [a + b for a, b in ops]
    assert all(unit(ops, 8) != expected for _, unit in faults)
    assert ADD(ops, 8) == expected  # the original is untouched
