"""Control tensors are discovered from state goals and optional demonstrations."""
import pytest
import torch

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateExample, StateLesson, control_candidates, evaluate, learn_stateful
from vectorpro.learning.registry import program_provenance
from vectorpro.machine import BudgetExceeded, Instr, assemble
from vectorpro.runtime import VectorRuntime


def conditional_case(source, destination, flag, content, old=b"old"):
    before = {source: content, destination: old, "keep": b"unchanged"}
    return StateExample((source, destination, flag), before,
                        before | ({destination: content} if flag else {}),
                        len(content) if flag else 0)


def conditional_lesson():
    return StateLesson(("path", "path", "value"),
                       [conditional_case("a", "b", 0, b"abc"),
                        conditional_case("c", "d", 1, b"xyz")],
                       [conditional_case("e", "f", 0, b"long"),
                        conditional_case("g", "h", 9, b"new\x00")],
                       max_steps=2, candidate_budget=20000, control_flow=True)


def test_learn_conditional_copy_native_and_reload(tmp_path):
    runtime = VectorRuntime(host=HostContext(tmp_path))
    runtime.provide_host_operations()
    lesson = conditional_lesson()
    outcome = learn_stateful(runtime.registry, "conditional", lesson)
    assert outcome.capability is not None
    assert outcome.history[-1]["control"] == "branch"
    for flag in (0, 1, 2, 255):
        assert evaluate(outcome.capability.executable,
                        conditional_case("unseen", "target", flag, bytes(range(256))), lesson)
    (tmp_path / "native").write_bytes(b"new content")
    (tmp_path / "target").write_bytes(b"keep me")
    operands = (runtime.host.put(b"native"), runtime.host.put(b"target"), 0)
    assert runtime.request("conditional", [operands], 16).outputs == [0]
    assert (tmp_path / "target").read_bytes() == b"keep me"
    runtime.save(tmp_path / "program.json")
    host = HostContext(tmp_path)
    restored = VectorRuntime.load(tmp_path / "program.json", host=host)
    assert restored.request("conditional", [(host.put(b"native"), host.put(b"target"), 7)], 16).outputs == [11]
    assert (tmp_path / "target").read_bytes() == b"new content"


def loop_case(path, count):
    files = {path: b"unchanged"}
    # Demonstrated effects distinguish true repetition from simply returning zero.
    return StateExample((path, count), files, dict(files), 0,
                        operations=("file.read",) * count.bit_length())


def loop_lesson():
    return StateLesson(("path", "value"),
                       [loop_case("a", 0), loop_case("b", 3), loop_case("c", 9)],
                       [loop_case("d", 1), loop_case("e", 31)],
                       max_steps=2, candidate_budget=20000, control_flow=True)


def learned_shift(runtime):
    torch.manual_seed(0)
    data = {"training": {"width": 4, "operands": [[x] for x in range(16)],
                         "targets": [x >> 1 for x in range(16)]},
            "validation": {"width": 8, "operands": [[x] for x in (0, 1, 2, 127, 128, 255)],
                           "targets": [x >> 1 for x in (0, 1, 2, 127, 128, 255)]}}
    plan = LearningPlan.from_dict({"name": "advance", "description": "example-defined step",
                                   "arity": 1, "output": "W", "rounds": [16],
                                   "train_width": 4, "validation_width": 8, "validation_examples": 6})
    assert runtime.request("advance", [(9,)], 16, plan=plan,
                           lesson=ExampleLesson.from_dict(data)).outputs == [4]


def test_discover_variable_iteration_loop_with_learned_numeric_feedback(tmp_path):
    runtime = VectorRuntime(host=HostContext(tmp_path))
    learned_shift(runtime)
    runtime.provide_host_operations()
    lesson = loop_lesson()
    outcome = learn_stateful(runtime.registry, "repeat", lesson)
    assert outcome.capability is not None
    assert outcome.history[-1]["control"] == "while"
    assert runtime.registry.loops("repeat")
    for count in (0, 1, 2, 7, 16, 127, 1024, 65535):
        assert evaluate(outcome.capability.executable, loop_case("unseen", count), lesson)
    (tmp_path / "native").write_bytes(b"unchanged")
    runtime.save(tmp_path / "program.json")
    host = HostContext(tmp_path)
    restored = VectorRuntime.load(tmp_path / "program.json", host=host)
    restored.request("repeat", [(host.put(b"native"), 1024)], 16)
    assert [e["operation"] for e in host.events] == ["file.read"] * 11
    assert (tmp_path / "native").read_bytes() == b"unchanged"
    cap = restored.registry.get("repeat")
    cap.executable.budget = 1
    with host.activate(), pytest.raises(BudgetExceeded):
        cap.run([(host.put(b"native"), 1024)], 16)


def test_control_evidence_failure_and_json_contract():
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    lesson = conditional_lesson()
    lesson.validation[0].after["extra"] = b"impossible"
    lesson.candidate_budget = 100
    assert learn_stateful(runtime.registry, "invalid", lesson).capability is None
    assert "invalid" not in runtime.registry
    data = {"input_types": ["path", "value"], "control_flow": True,
            "training": [{"inputs": ["a", 0], "before": {"a": ""}, "after": {"a": ""}, "operations": []}],
            "validation": [{"inputs": ["b", 1], "before": {"b": ""}, "after": {"b": ""}, "operations": ["file.read"]}]}
    parsed = StateLesson.from_dict(data)
    parsed.validate()
    assert parsed.control_flow and parsed.training[0].operations == ()
    parsed.control_flow = "yes"
    with pytest.raises(ValueError, match="boolean"):
        parsed.validate()


def test_nonterminating_effectful_candidate_is_rejected_in_memory(monkeypatch):
    def forbidden(*args):
        raise AssertionError("candidate must not access native files")
    monkeypatch.setattr(HostContext, "_read_file", forbidden)
    monkeypatch.setattr(HostContext, "_write_file", forbidden)
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    # Setting a byte returns that same value, so feedback of a nonzero byte
    # cannot terminate. This is a negative candidate, not a learned task recipe.
    instructions = (Instr("file.read", ("x0",), "t1"),
                    Instr("buffer.set", ("t1", "zero", "x1"), "t2"))
    variants = control_candidates(instructions, ("path", "value"), ("buffer", "value"), True)
    candidate, output, _ = next(v for v in variants if v[2] == "while" and v[1] == "x1")
    program = assemble(candidate, ["x0", "x1"],
                       {"x0": "zero", "x1": "zero", "zero": "zero", "t1": "zero", "t2": "zero"},
                       output, runtime.registry.key_of)
    plan = LearningPlan.from_dict({"name": "negative", "description": "nonterminating control",
                                   "arity": 2, "output": "W"})
    executable = runtime.registry.build(plan, program_provenance(program, "negative control"))
    files = {"a": b"\x01"}
    case = StateExample(("a", 1), files, dict(files), 0)
    lesson = StateLesson(("path", "value"), [case], [StateExample(("b", 1), {"b": b"\x01"}, {"b": b"\x01"}, 0)], width=4)
    assert program.has_loop
    assert not evaluate(executable, case, lesson)
    assert "negative" not in runtime.registry
