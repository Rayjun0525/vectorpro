"""M2c: wider composition search, generic bit folds, and the learner's loop strategy."""

import random

import pytest

from vectorpro.cells import CellSignature, TableCell
from vectorpro.expr import BinOp, Call, Const, Var, expr_from_data, expr_to_data, render
from vectorpro.learning import (
    Capability,
    ExampleSet,
    ExampleStream,
    Learner,
    LearnerConfig,
    LearningPlan,
    OutputWidth,
    Registry,
    search_composition,
)
from vectorpro.learning.loops import find_primitives, search_bit_fold, simulate
from vectorpro.learning.registry import program_provenance, unit_provenance
from vectorpro.machine import BitFold, compile_bit_fold, compile_expression, disassemble
from vectorpro.schemas import Direction, MapSchema, ScanSchema
from vectorpro.training import TrainConfig
from vectorpro.tasks import AND, FULL_ADDER, FULL_SUBTRACTOR, LESS_THAN, MUX, OR, XOR, LocalRule
from vectorpro.units import FunctionUnit

SHIFT = LocalRule("shift", CellSignature(1, 1, 1), lambda x, s: ((s[0],), (x[0],)))
UNITS = [
    ("and", MapSchema(2), AND.local_rule, 2, OutputWidth.SAME),
    ("or", MapSchema(2), OR.local_rule, 2, OutputWidth.SAME),
    ("xor", MapSchema(2), XOR.local_rule, 2, OutputWidth.SAME),
    ("add", ScanSchema(2, (0,)), FULL_ADDER, 2, OutputWidth.PLUS_ONE),
    ("sub", ScanSchema(2, (0,)), FULL_SUBTRACTOR, 2, OutputWidth.PLUS_ONE),
    ("lt", ScanSchema(2, (0,)), LESS_THAN, 2, OutputWidth.BIT),
    ("shl", ScanSchema(1, (0,), Direction.LSB_FIRST, False), SHIFT, 1, OutputWidth.SAME),
    ("shr", ScanSchema(1, (0,), Direction.MSB_FIRST, False), SHIFT, 1, OutputWidth.SAME),
    ("mux", MapSchema(3), MUX.local_rule, 3, OutputWidth.SAME),
]


def mask(w):
    return (1 << w) - 1


def exact_registry() -> Registry:
    """Exact (rule-built) units, standing in for already-learned ones."""
    reg = Registry(seed=3)
    for name, schema, rule, arity, output in UNITS:
        unit = FunctionUnit(name, schema, TableCell.from_rule(rule))
        plan, prov = LearningPlan(name, name, arity, output), unit_provenance(unit)
        reg.add(Capability(plan, reg.build(plan, prov), prov))
    return reg


@pytest.fixture(scope="module")
def reg() -> Registry:
    return exact_registry()


def example_sets(fn, arity, n=24, seed=0):
    stream = ExampleStream(fn, arity, random.Random(seed))
    return [stream.extend(ExampleSet(4), n), stream.extend(ExampleSet(8), n)]


def test_expr_call_and_const_roundtrip():
    expr = Call("mux", (Var(0), Const("ones"), BinOp("lt", Var(1), Var(0))))
    assert expr_from_data(expr_to_data(expr)) == expr
    assert render(expr) == "mux(x0, ones, (x1 lt x0))"
    assert render(Var(2), ["a", "b", "acc"]) == "acc"


def test_commutativity_is_detected_by_behaviour(reg):
    commutative = {op.name for op in reg.operators() if op.commutative}
    assert commutative == {"and", "or", "xor", "add"}


def test_search_uses_constants_and_ternary_operators(reg):
    expr = search_composition(reg.operators(), 2, example_sets(lambda o, w: max(o), 2))
    assert expr is not None and "mux" in render(expr) and "lt" in render(expr)
    program = compile_expression(expr, 2, reg.key_of)
    plan, prov = LearningPlan("max_t", "max", 2, OutputWidth.SAME), program_provenance(program, "t")
    reg.add(Capability(plan, reg.build(plan, prov), prov))
    ops = [(3, 9), (200, 7), (5, 5), (0, 255)]
    assert reg.run("max_t", ops, 8) == [max(o) for o in ops]


def test_bit_extraction_primitives_are_found_by_behaviour(reg):
    p = find_primitives(reg.operators())
    assert (p.test_and, p.to_bit, p.to_mask, p.shift_down, p.shift_up) == ("and", "lt", "sub", "shr", "shl")


def test_primitives_missing_means_no_loops():
    assert find_primitives([]) is None


@pytest.mark.parametrize(
    "name, arity, fn",
    [
        ("mul", 2, lambda o, w: (o[0] * o[1]) & mask(w)),
        ("popcount", 1, lambda o, w: bin(o[0]).count("1")),
        ("reverse", 1, lambda o, w: int(format(o[0], f"0{w}b")[::-1], 2)),
    ],
)
def test_fold_search_and_compiled_program_agree(reg, name, arity, fn):
    ops = [op for op in reg.operators() if not op.loops]
    fold = search_bit_fold(ops, arity, example_sets(fn, arity))
    assert fold is not None
    program = compile_bit_fold(fold, reg.key_of, find_primitives(ops))
    plan, prov = LearningPlan(f"{name}_t", name, arity, OutputWidth.SAME), program_provenance(program, "t")
    reg.add(Capability(plan, reg.build(plan, prov), prov))
    tuples = [tuple(random.Random(i).getrandbits(16) for _ in range(arity)) for i in range(60)]
    assert reg.run(f"{name}_t", tuples, 16) == [fn(t, 16) for t in tuples]
    assert any("goto @0" in line for line in disassemble(program, reg))  # it really loops


def test_simulation_matches_reference_for_hand_written_fold(reg):
    ops = {o.name: o for o in reg.operators()}
    horner = BitFold(2, 1, Direction.MSB_FIRST,
                     BinOp("add", BinOp("add", Var(2), Var(2)), BinOp("and", Var(0), Var(4))))
    sets = example_sets(lambda o, w: (o[0] * o[1]) & mask(w), 2)
    from vectorpro.bits import BitCodec
    for s in sets:
        got = BitCodec.decode(simulate(horner, ops, BitCodec.encode_operands(s.operands, s.width)))
        assert got == s.targets


def register_program(reg, name, program, arity=1):
    plan, prov = LearningPlan(name, name, arity, OutputWidth.SAME), program_provenance(program, "t")
    reg.add(Capability(plan, reg.build(plan, prov), prov))


def test_loop_detection_reads_the_control_tensors():
    reg = exact_registry()
    straight = compile_expression(BinOp("add", Var(0), Var(1)), 2, reg.key_of)
    looped = compile_bit_fold(BitFold(1, 0, Direction.LSB_FIRST, BinOp("add", Var(1), Var(2))),
                              reg.key_of, find_primitives(reg.operators()))
    assert not straight.has_loop and looped.has_loop

    register_program(reg, "count_t", looped)
    caller = compile_expression(Call("count_t", (Var(0),)), 1, reg.key_of)
    register_program(reg, "caller_t", caller)
    assert not caller.has_loop and reg.loops("caller_t")  # loops by calling a loop
    assert not reg.loops("add")
    ops = {op.name: op for op in reg.operators()}
    assert ops["caller_t"].loops and not ops["add"].loops


def test_fold_data_roundtrip():
    fold = BitFold(1, 0, Direction.LSB_FIRST, BinOp("add", Var(1), Var(2)))
    assert BitFold.from_data(fold.to_data()) == fold
    assert fold.describe() == "acc = 0; for each bit of x0 (lowest to highest): acc = (acc add bit)"


def test_learner_finds_loop_when_reuse_and_units_fail():
    reg = exact_registry()
    learner = Learner(reg, LearnerConfig(train=TrainConfig(steps=200), restarts=1))
    plan = LearningPlan("popcount_l", "number of set bits", 1, OutputWidth.SAME, rounds=(16,))
    outcome = learner.learn(plan, lambda o, w: bin(o[0]).count("1"), random.Random(1))
    assert outcome.learned and outcome.history[-1]["strategy"] == "loop"
    assert reg.run("popcount_l", [(0xFFFF,), (0x8001,)], 16) == [16, 2]


def register_composition(reg, name, expr, arity=2, headroom=0):
    register_program(reg, name, compile_expression(expr, arity, reg.key_of, headroom), arity)


def division_registry():
    """Exact units plus the helper compositions division builds on."""
    reg = exact_registry()
    x0, x1 = Var(0), Var(1)
    register_composition(reg, "dadd", BinOp("add", x0, BinOp("add", x0, x1)))
    register_composition(reg, "ge", BinOp("lt", BinOp("lt", x0, x1), Const("one")))
    register_composition(reg, "gemask", BinOp("add", Const("ones"), BinOp("lt", x0, x1)))
    register_composition(reg, "csub", Call("mux", (x0, BinOp("sub", x0, x1), BinOp("gemask", x0, x1))))
    return reg


def test_headroom_composition_finds_average():
    from vectorpro.learning.search import find_composition
    reg = exact_registry()
    found = find_composition(reg.operators(), 2, example_sets(lambda o, w: (o[0] + o[1]) >> 1, 2))
    assert found is not None and found[1] == 1  # needs one extra bit for the carry
    expr, headroom = found
    register_composition(reg, "avg_t", expr, headroom=headroom)
    ops = [(255, 255), (200, 100), (1, 2)]
    assert reg.run("avg_t", ops, 8) == [255, 150, 1]


def test_division_is_found_beside_a_learned_remainder_loop():
    reg = division_registry()
    ops = [op for op in reg.operators() if not op.loops]
    mod_fn = lambda o, w: o[0] % o[1] if o[1] else o[0]  # noqa: E731
    div_fn = lambda o, w: o[0] // o[1] if o[1] else mask(w)  # noqa: E731
    mod_fold = search_bit_fold(ops, 2, example_sets(mod_fn, 2), headrooms=(0,))
    assert mod_fold is not None and not mod_fold.companions

    div_fold = search_bit_fold(ops, 2, example_sets(div_fn, 2), headrooms=(), library=[mod_fold])
    assert div_fold is not None and len(div_fold.companions) == 1
    program = compile_bit_fold(div_fold, reg.key_of, find_primitives(ops))
    register_program(reg, "div_t", program, arity=2)
    tuples = [(random.Random(i).getrandbits(16), random.Random(-i).getrandbits(16) >> (i % 16))
              for i in range(80)] + [(5, 0), (0, 0), (65535, 1), (65535, 65535)]
    assert reg.run("div_t", tuples, 16) == [div_fn(t, 16) for t in tuples]
