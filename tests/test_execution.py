"""Schemas, table cells and programs, checked with exact (non-learned) rules."""

import random

import pytest
import torch

from vectorpro.cells import CellSignature, TableCell
from vectorpro.data import edge_pairs, random_tuples
from vectorpro.programs import Fold
from vectorpro.quantize import Identity
from vectorpro.schemas import Direction, MapSchema, ScanSchema
from vectorpro.tasks import FULL_ADDER, LocalRule
from vectorpro.units import FunctionUnit

XOR = LocalRule("xor", CellSignature(2, 0, 1), lambda x, s: ((x[0] ^ x[1],), ()))

# MSB-first comparison: state is (decided, a_less_than_b).
def _less_than(x, s):
    decided, lt = s
    if decided or x[0] == x[1]:
        return (), (decided, lt)
    return (), (1, int(x[0] < x[1]))


LESS_THAN = LocalRule("less_than", CellSignature(2, 2, 0), _less_than)


def exact_adder() -> FunctionUnit:
    return FunctionUnit("add", ScanSchema(2, (0,)), TableCell.from_rule(FULL_ADDER))


@pytest.mark.parametrize("width", [1, 4, 17, 64])
def test_scan_with_exact_rule_adds(width):
    ops = random_tuples(2, width, 500, random.Random(width)) + edge_pairs(width)
    assert exact_adder()(ops, width) == [a + b for a, b in ops]


def test_msb_first_scan_compares():
    unit = FunctionUnit(
        "lt", ScanSchema(2, (0, 0), Direction.MSB_FIRST), TableCell.from_rule(LESS_THAN)
    )
    ops = random_tuples(2, 12, 500, random.Random(0)) + [(5, 5)]
    # Output is the final state (decided, lt); lt is bit 1.
    assert [r >> 1 for r in unit(ops, 12)] == [int(a < b) for a, b in ops]


def test_map_schema_xor():
    unit = FunctionUnit("xor", MapSchema(2), TableCell.from_rule(XOR))
    ops = random_tuples(2, 32, 200, random.Random(1))
    assert unit(ops, 32) == [a ^ b for a, b in ops]


def test_fold_wraps():
    ops = random_tuples(5, 32, 200, random.Random(2))
    assert Fold(exact_adder(), 5)(ops, 32) == [sum(t) % (1 << 32) for t in ops]


def test_table_from_cell_roundtrips():
    table = TableCell.from_rule(FULL_ADDER)
    assert torch.equal(TableCell.from_cell(table).table, table.table)


def test_schema_rejects_incompatible_cell():
    with pytest.raises(ValueError):
        FunctionUnit("bad", MapSchema(2), TableCell.from_rule(FULL_ADDER))


def test_continuous_execution_matches_on_exact_rule():
    ops = random_tuples(2, 16, 100, random.Random(3))
    assert exact_adder()(ops, 16, Identity()) == [a + b for a, b in ops]
