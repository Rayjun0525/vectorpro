import pytest

from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


def transfer_case(source, destination, content, previous=None):
    before = {source: content, "untouched": b"keep"}
    if previous is not None:
        before[destination] = previous
    return StateExample((source, destination), before, before | {destination: content}, len(content))


def lesson():
    return StateLesson(("path", "path"),
                       [transfer_case("a", "b", b"abc"), transfer_case("c", "d", b"\x00\xff")],
                       [transfer_case("empty", "out", b"", b"old"),
                        transfer_case("long", "new", b"longer content", b"x")], max_steps=2)


def test_discover_argument_routing_and_native_execution_then_reload(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    (tmp_path / "unseen").write_bytes(b"new content\x00\xff")
    args = (host.put(b"unseen"), host.put(b"result"))
    result = runtime.request("arbitrary_name", [args], 16, state_lesson=lesson())
    assert result.status == "learned_and_executed" and result.outputs == [13]
    assert (tmp_path / "result").read_bytes() == b"new content\x00\xff"
    cap = runtime.registry.get("arbitrary_name")
    assert cap.provenance["origin"] == "state example search"
    assert cap.history[-1]["steps"] == 2
    assert "2 training cases" in runtime.registry.explain("arbitrary_name")
    assert cap.provenance["input_types"] == ["path", "path"]
    for i, content in enumerate((b"", b"hello", bytes(range(256)), b"x" * 4096)):
        case = transfer_case(f"s{i}", f"d{i}", content, b"replace me")
        assert evaluate(cap.executable, case, lesson())
    path = tmp_path / "program.json"
    runtime.save(path)
    fresh = HostContext(tmp_path)
    restored = VectorRuntime.load(path, host=fresh)
    args = (fresh.put(b"unseen"), fresh.put(b"reloaded"))
    assert restored.request("arbitrary_name", [args], 16).status == "executed"
    assert (tmp_path / "reloaded").read_bytes() == b"new content\x00\xff"


def test_virtual_candidates_never_invoke_native_backend(monkeypatch):
    def forbidden(*args):
        raise AssertionError("synthesis must not access native files")
    monkeypatch.setattr(HostContext, "_read_file", forbidden)
    monkeypatch.setattr(HostContext, "_write_file", forbidden)
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    outcome = learn_stateful(runtime.registry, "transfer", lesson())
    assert outcome.capability is not None


def test_budget_failure_does_not_register_task_or_touch_native_files(tmp_path):
    (tmp_path / "marker").write_bytes(b"unchanged")
    runtime = VectorRuntime(host=HostContext(tmp_path))
    runtime.provide_host_operations()
    examples = lesson()
    examples.max_steps = 1
    outcome = learn_stateful(runtime.registry, "impossible", examples)
    assert outcome.capability is None and "impossible" not in runtime.registry
    assert (tmp_path / "marker").read_bytes() == b"unchanged"
    examples.max_steps = 2
    examples.candidate_budget = 1
    assert learn_stateful(runtime.registry, "limited", examples).capability is None


def test_validation_counterexample_rejects_training_fit():
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    examples = lesson()
    examples.validation[0].after["extra"] = b"cannot produce"
    assert learn_stateful(runtime.registry, "bad", examples).capability is None
    assert "bad" not in runtime.registry


def test_goal_checks_unintended_file_changes():
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    examples = lesson()
    cap = learn_stateful(runtime.registry, "transfer", examples).capability
    case = transfer_case("a", "b", b"abc")
    del case.after["untouched"]  # Exact target snapshot requests a deletion too.
    assert not evaluate(cap.executable, case, examples)


def test_virtual_paths_and_validation_overlap_are_rejected():
    with pytest.raises(ValueError):
        MemoryHostContext({"../outside": b"x"})
    examples = lesson()
    examples.validation[0] = examples.training[0]
    with pytest.raises(ValueError, match="distinct"):
        examples.validate()


def test_same_search_learns_a_different_observation_procedure():
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    def case(name, content):
        files = {name: content}
        return StateExample((name,), files, dict(files), content[0])
    examples = StateLesson(("path",), [case("a", b"\x07\x01"), case("b", b"\xff\x00")],
                           [case("c", b"\x80\x02"), case("d", b"\x2a\x03")], max_steps=2)
    outcome = learn_stateful(runtime.registry, "observe", examples)
    assert outcome.capability is not None
    assert evaluate(outcome.capability.executable, case("unseen", b"\x63\x12\x13"), examples)
