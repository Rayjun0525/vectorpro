"""Already learned numeric functions and state procedures participate in search."""
import json
from pathlib import Path

import pytest
import torch

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


@pytest.fixture
def runtime(tmp_path):
    data = json.loads((Path(__file__).resolve().parents[1] / "experiments/requests/learn_xor.json").read_text())
    data["plan"]["name"] = "combine"
    torch.manual_seed(0)
    model = VectorRuntime(host=HostContext(tmp_path))
    result = model.request("combine", [(1, 2)], 16,
                           plan=LearningPlan.from_dict(data["plan"]),
                           lesson=ExampleLesson.from_dict(data["lesson"]))
    assert result.status == "learned_and_executed"
    model.provide_host_operations()
    return model


def observation(name, content):
    return StateExample((name,), {name: content}, {name: content}, content[0])


def numeric_case(name, content, mask):
    return StateExample((name, mask), {name: content}, {name: content}, content[0] ^ mask)


def combined_lesson(steps=3):
    return StateLesson(("path", "value"),
                       [numeric_case("a", b"\x07\x00", 3), numeric_case("b", b"\xa5\x11", 15)],
                       [numeric_case("c", b"\xff\x01", 170), numeric_case("d", b"\x10\x12", 9)],
                       max_steps=steps, candidate_budget=15000)


def test_state_search_calls_actually_learned_numeric_unit(runtime, tmp_path):
    example = combined_lesson()
    outcome = learn_stateful(runtime.registry, "file_compute", example)
    assert outcome.capability is not None
    listing = runtime.registry.explain("file_compute")
    assert "combine(" in listing
    assert "file.read(" in listing and "buffer.get(" in listing
    assert outcome.capability.history[-1]["steps"] == 3
    case = numeric_case("unseen", b"\x63\x00\xff", 513)
    assert evaluate(outcome.capability.executable, case, example)
    (tmp_path / "native").write_bytes(b"\x63\x00\xff")
    path = runtime.host.put(b"native")
    assert runtime.request("file_compute", [(path, 513)], 16).outputs == [610]
    assert (tmp_path / "native").read_bytes() == b"\x63\x00\xff"
    runtime.save(tmp_path / "program.json")
    context = HostContext(tmp_path)
    restored = VectorRuntime.load(tmp_path / "program.json", host=context)
    assert restored.request("file_compute", [(context.put(b"native"), 513)], 16).outputs == [610]


def test_learned_state_function_is_reused_in_later_search(runtime):
    observations = StateLesson(("path",), [observation("a", b"\x07\x01"), observation("b", b"\xa5\x11")],
                               [observation("c", b"\xff\x03"), observation("d", b"\x10\x12")], max_steps=2)
    first = learn_stateful(runtime.registry, "observe", observations)
    assert first.capability is not None
    assert first.capability.provenance["output_type"] == "value"
    second = learn_stateful(runtime.registry, "combined", combined_lesson(steps=2))
    assert second.capability is not None
    listing = runtime.registry.explain("combined")
    assert "observe(" in listing and "combine(" in listing
    assert second.capability.history[-1]["steps"] == 2
    assert evaluate(second.capability.executable, numeric_case("new", b"\x42\xff", 1024), combined_lesson())


def test_no_numeric_unit_means_goal_cannot_be_met_by_host_primitives():
    model = VectorRuntime()
    model.provide_host_operations()
    example = combined_lesson(steps=2)
    outcome = learn_stateful(model.registry, "missing", example)
    assert outcome.capability is None and "missing" not in model.registry
