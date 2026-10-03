import random

import pytest
import torch

from vectorpro.cells import TableCell
from vectorpro.expr import BinOp, Var, expr_from_data, expr_to_data, render
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
    structure_candidates,
)
from vectorpro.learning.examples import ExampleTask
from vectorpro.learning.registry import unit_provenance
from vectorpro.schemas import Direction, MapSchema, ScanSchema, schema_from_spec
from vectorpro.tasks import FULL_ADDER, XOR
from vectorpro.training import TrainConfig
from vectorpro.units import FunctionUnit

ADD_PLAN = LearningPlan("add", "sum with carry out", 2, OutputWidth.PLUS_ONE)
XOR_PLAN = LearningPlan("xor", "bitwise exclusive or", 2, OutputWidth.SAME, tags=("bitwise",))


def seeded_registry() -> Registry:
    """A registry holding exact (rule-built) add and xor, as if already learned."""
    registry = Registry()
    for plan, unit in [
        (ADD_PLAN, FunctionUnit("add", ScanSchema(2, (0,)), TableCell.from_rule(FULL_ADDER))),
        (XOR_PLAN, FunctionUnit("xor", MapSchema(2), TableCell.from_rule(XOR.local_rule))),
    ]:
        provenance = unit_provenance(unit)
        registry.add(Capability(plan, registry.build(plan, provenance), provenance))
    return registry


def examples(fn, arity, width, n, seed=0):
    return ExampleStream(fn, arity, random.Random(seed)).extend(ExampleSet(width), n)


def test_plan_roundtrip_and_defaults():
    plan = LearningPlan.from_dict({"name": "x", "description": "d", "arity": 2, "output": "W+1"})
    assert plan.rounds == LearningPlan.rounds and plan.output is OutputWidth.PLUS_ONE
    assert LearningPlan.from_dict(XOR_PLAN.to_dict()) == XOR_PLAN


@pytest.mark.parametrize("schema", [MapSchema(3), ScanSchema(2, (0, 1), Direction.MSB_FIRST, False)])
def test_schema_spec_roundtrip(schema):
    assert schema_from_spec(schema.to_spec()) == schema


def test_expr_data_roundtrip():
    expr = BinOp("add", Var(0), BinOp("xor", Var(1), Var(2)))
    assert expr_from_data(expr_to_data(expr)) == expr


def test_example_stream_is_distinct_and_capped():
    ex = examples(lambda o, w: o[0], 1, 3, 100)
    assert len(ex) == 8 and len(set(ex.operands)) == 8


def test_example_task_only_answers_handed_over_examples():
    ex = examples(lambda o, w: o[0] + o[1], 2, 4, 4)
    task = ExampleTask(ex, OutputWidth.PLUS_ONE, 2)
    assert task.reference(ex.operands[0], 4) == ex.targets[0]
    unseen = next(t for t in [(a, b) for a in range(16) for b in range(16)] if t not in ex.operands)
    with pytest.raises(KeyError):
        task.reference(unseen, 4)


def test_search_finds_smallest_composition_and_gives_up_on_loops():
    ops = seeded_registry().operators()
    sets = [examples(lambda o, w: (o[0] + o[1] + o[2]) % (1 << w), 3, 4, 16),
            examples(lambda o, w: (o[0] + o[1] + o[2]) % (1 << w), 3, 8, 16, seed=1)]
    assert render(search_composition(ops, 3, sets)) == "(x0 add (x1 add x2))"
    mul = [examples(lambda o, w: (o[0] * o[1]) % (1 << w), 2, 4, 16)]
    assert search_composition(ops, 2, mul) is None


def test_registry_save_load_search_and_describe(tmp_path):
    registry = seeded_registry()
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = Registry.load(path)
    ops = [(3, 5), (255, 1), (0, 0)]
    assert loaded.run("add", ops, 8) == [8, 256, 0]
    assert [c.name for c in loaded.search("bitwise")] == ["xor"]
    xor_examples = examples(lambda o, w: o[0] ^ o[1], 2, 8, 8)
    assert [c.name for c in loaded.find_by_examples(2, xor_examples)] == ["xor"]
    assert "11 | 1 -> 1 | 1" in loaded.explain("add")


def test_structure_candidates_match_requested_shape():
    for output in (OutputWidth.SAME, OutputWidth.PLUS_ONE, OutputWidth.BIT):
        for candidate in structure_candidates("t", 2, output):
            assert candidate.build().output_width(8) == output(8)
    assert structure_candidates("t", 2, OutputWidth.DOUBLE) == []


def test_learner_reuses_registry_without_training():
    registry = seeded_registry()
    plan = LearningPlan("add3", "sum of three", 3, OutputWidth.SAME)
    outcome = Learner(registry).learn(plan, lambda o, w: sum(o) % (1 << w), random.Random(0))
    assert outcome.learned and outcome.history[-1]["strategy"] == "reuse"
    assert registry.run("add3", [(250, 10, 1)], 8) == [5]


def test_learner_learns_a_new_unit_from_examples_only():
    torch.manual_seed(0)
    registry = Registry()
    learner = Learner(registry, LearnerConfig(train=TrainConfig(steps=800), restarts=2))
    outcome = learner.learn(XOR_PLAN, lambda o, w: o[0] ^ o[1], random.Random(0))
    assert outcome.learned and outcome.history[-1]["strategy"] == "learn"
    assert outcome.capability.provenance["table"] == [[0], [1], [1], [0]]


def test_learner_reports_failure_within_budget():
    registry = seeded_registry()
    plan = LearningPlan("mul", "product", 2, OutputWidth.DOUBLE, rounds=(8, 16))
    outcome = Learner(registry).learn(plan, lambda o, w: o[0] * o[1], random.Random(0))
    assert not outcome.learned and len(outcome.history) == 2 and "mul" not in registry
