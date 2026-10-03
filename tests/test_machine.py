"""Vector programs on the task-agnostic kernel, with exact (rule-built) capabilities."""

import random
import sys
from pathlib import Path

import pytest
import torch

from vectorpro.bits import BitCodec
from vectorpro.cells import CellSignature, TableCell
from vectorpro.data import all_tuples, edge_pairs, random_tuples
from vectorpro.expr import BinOp, Var
from vectorpro.learning import Capability, LearningPlan, OutputWidth, Registry
from vectorpro.learning.registry import program_provenance, unit_provenance
from vectorpro.machine import (
    HALT,
    BudgetExceeded,
    Instr,
    VectorProgram,
    assemble,
    compile_expression,
    disassemble,
    run,
)
from vectorpro.quantize import HardThreshold
from vectorpro.schemas import Direction, MapSchema, ScanSchema
from vectorpro.tasks import AND, FULL_ADDER, FULL_SUBTRACTOR, LESS_THAN, OR, LocalRule
from vectorpro.units import FunctionUnit

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import authored  # noqa: E402

SHIFT = LocalRule("shift", CellSignature(1, 1, 1), lambda x, s: ((s[0],), (x[0],)))


def register_unit(reg, name, schema, rule, arity, output):
    unit = FunctionUnit(name, schema, TableCell.from_rule(rule))
    plan, prov = LearningPlan(name, name, arity, output), unit_provenance(unit)
    reg.add(Capability(plan, reg.build(plan, prov), prov))


def register_program(reg, name, program, arity=2):
    plan, prov = LearningPlan(name, name, arity, OutputWidth.SAME), program_provenance(program, "test")
    reg.add(Capability(plan, reg.build(plan, prov), prov))


@pytest.fixture
def reg() -> Registry:
    reg = Registry(seed=1)
    register_unit(reg, "and", MapSchema(2), AND.local_rule, 2, OutputWidth.SAME)
    register_unit(reg, "or", MapSchema(2), OR.local_rule, 2, OutputWidth.SAME)
    register_unit(reg, "add", ScanSchema(2, (0,)), FULL_ADDER, 2, OutputWidth.PLUS_ONE)
    register_unit(reg, "sub", ScanSchema(2, (0,)), FULL_SUBTRACTOR, 2, OutputWidth.PLUS_ONE)
    register_unit(reg, "lt", ScanSchema(2, (0,)), LESS_THAN, 2, OutputWidth.BIT)
    register_unit(reg, "shl", ScanSchema(1, (0,), Direction.LSB_FIRST, False), SHIFT, 1, OutputWidth.SAME)
    register_unit(reg, "shr", ScanSchema(1, (0,), Direction.MSB_FIRST, False), SHIFT, 1, OutputWidth.SAME)
    register_program(reg, "mul", authored.mul(reg.key_of))
    register_program(reg, "div", authored.divmod_program(reg.key_of, "q"))
    register_program(reg, "mod", authored.divmod_program(reg.key_of, "r"))
    return reg


REFERENCE = {
    "mul": lambda a, b, w: (a * b) % (1 << w),
    "div": lambda a, b, w: a // b if b else (1 << w) - 1,
    "mod": lambda a, b, w: a % b if b else a,
}


def operands(width):
    if width <= 4:
        return all_tuples(2, width)
    return random_tuples(2, width, 150, random.Random(width)) + edge_pairs(width)


@pytest.mark.parametrize("width", [1, 4, 12])
@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_loop_and_branch_programs(reg, name, width):
    ops = operands(width)
    assert reg.run(name, ops, width) == [REFERENCE[name](a, b, width) for a, b in ops]


def test_program_calls_program_by_address(reg):
    expr = BinOp("add", BinOp("mul", Var(0), Var(1)), Var(2))
    register_program(reg, "mul_add", compile_expression(expr, 3, reg.key_of), arity=3)
    ops = random_tuples(3, 10, 100, random.Random(0))
    assert reg.run("mul_add", ops, 10) == [(a * b + c) % 1024 for a, b, c in ops]


def test_dispatch_follows_address_vectors(reg):
    ops = [(7, 3), (200, 13)]
    assert reg.run("mul", ops, 8) == [21, (200 * 13) % 256]
    add, sub = reg.get("add"), reg.get("sub")
    add.key, sub.key = sub.key, add.key  # same program tensors, swapped addresses
    assert reg.run("mul", ops, 8) != [21, (200 * 13) % 256]


def test_disassembly_comes_from_tensors(reg):
    listing = disassemble(reg.get("mul").executable.program, reg)
    assert listing[0] == "r0 = input x0"
    assert "@1: r2 = add(r2, r0)" in listing
    assert listing[-2] == "@3: r1 = shr(r1); if r1 != 0 goto @0 else halt"


def test_non_halting_program_is_reported_not_answered(reg):
    spin = assemble([Instr("and", ("a", "a"), "a", then="spin", otherwise="spin", label="spin")],
                    ["a"], {"a": "zero"}, "a", reg.key_of)
    result = run(spin, BitCodec.encode_operands([(1,), (2,)], 4), reg, HardThreshold(), budget=50)
    assert not result.halted.any() and result.clocks == 50
    register_program(reg, "spin", spin, arity=1)
    with pytest.raises(BudgetExceeded):
        reg.run("spin", [(1,)], 4)


def test_program_data_roundtrip_and_registry_reload(reg, tmp_path):
    program = reg.get("div").executable.program
    again = VectorProgram.from_data(program.to_data())
    assert all(torch.equal(getattr(program, f), getattr(again, f)) for f in ("keys", "reads", "next_true"))
    reg.save(tmp_path / "r.json")
    loaded = Registry.load(tmp_path / "r.json")
    ops = operands(8)
    assert loaded.run("div", ops, 8) == reg.run("div", ops, 8)


def test_assembler_rejects_reserved_labels(reg):
    with pytest.raises(ValueError):
        assemble([Instr("and", ("a", "a"), "a", label=HALT)], ["a"], {"a": "zero"}, "a", reg.key_of)


def test_compile_bare_variable(reg):
    program = compile_expression(Var(1), 2, reg.key_of)
    register_program(reg, "second", program)
    assert reg.run("second", [(3, 9)], 8) == [9]
