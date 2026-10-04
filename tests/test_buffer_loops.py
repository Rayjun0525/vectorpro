from pathlib import Path

import pytest

from vectorpro.host import HostContext
from vectorpro.learning import Capability, Registry
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


def runtime(root):
    # Reuse data tables from the earlier, actually learned arithmetic registry.
    acquired = Registry.load(Path(__file__).resolve().parents[1] / "results/four_ops_registry_20261004.json")
    model = VectorRuntime(host=HostContext(root))
    for name in ("sub", "xor"):
        cap = acquired.get(name)
        model.registry.add(Capability(cap.plan, model.registry.build(cap.plan, cap.provenance), cap.provenance, cap.history))
    model.provide_host_operations()
    return model


def case(path, value, content, fill=False):
    before = {path: content, "keep": b"untouched"}
    after = bytes([value]) * len(content) if fill else bytes(b ^ value for b in content)
    return StateExample((path, value), before, before | {path: after}, len(content))


def lesson(fill=False):
    return StateLesson(("path", "value"),
                       [case("a", 7, b"\x01\x02\x03", fill), case("b", 0, b"", fill)],
                       [case("c", 31, b"\xff\x00\x80\x05", fill), case("d", 2, b"\x10", fill)],
                       max_steps=5 if fill else 7, candidate_budget=5000, buffer_loops=True,
                       execution_budget=20000)


@pytest.mark.parametrize("fill", [False, True])
def test_learn_mutating_file_loop_from_final_states_native_and_reload(tmp_path, fill):
    model = runtime(tmp_path)
    examples = lesson(fill)
    outcome = learn_stateful(model.registry, "transform", examples)
    assert outcome.capability is not None
    assert model.registry.loops("transform")
    assert outcome.history[-1]["control"] == ("buffer-fill" if fill else "buffer-map")
    for size in (0, 1, 2, 31, 257, 2048):
        content = bytes(i % 256 for i in range(size))
        assert evaluate(outcome.capability.executable, case("unseen", 29, content, fill), examples)
    (tmp_path / "native").write_bytes(bytes(range(256)))
    assert model.request("transform", [(model.host.put(b"native"), 53)], 16).outputs == [256]
    expected = bytes([53]) * 256 if fill else bytes(x ^ 53 for x in range(256))
    assert (tmp_path / "native").read_bytes() == expected
    model.save(tmp_path / "program.json")
    host = HostContext(tmp_path)
    restored = VectorRuntime.load(tmp_path / "program.json", host=host)
    assert restored.registry.get("transform").executable.budget == 20000
    restored.request("transform", [(host.put(b"native"), 53)], 16)
    assert (tmp_path / "native").read_bytes() == (expected if fill else bytes(range(256)))


def test_empty_iteration_and_unsatisfiable_snapshot_do_not_escape_memory(tmp_path, monkeypatch):
    model = runtime(tmp_path)
    monkeypatch.setattr(HostContext, "_read_file", lambda *args: pytest.fail("native read during learning"))
    monkeypatch.setattr(HostContext, "_write_file", lambda *args: pytest.fail("native write during learning"))
    examples = lesson(True)
    assert learn_stateful(model.registry, "fill", examples).capability is not None
    examples.validation[0].after["impossible"] = b"extra"
    examples.candidate_budget = 100
    assert learn_stateful(model.registry, "bad", examples).capability is None
    assert "bad" not in model.registry


def test_conditional_and_multiple_learned_loops_are_composed(tmp_path):
    model = runtime(tmp_path)
    assert learn_stateful(model.registry, "map", lesson()).capability is not None

    def guarded(path, mask, flag, content):
        before = {path: content}
        return StateExample((path, mask, flag), before,
                            {path: bytes(b ^ mask for b in content) if flag else content}, len(content) if flag else 0)

    guards = StateLesson(("path", "value", "value"),
                         [guarded("a", 7, 0, b"abc"), guarded("b", 5, 1, b"xyz")],
                         [guarded("c", 9, 2, b"\x00\xff"), guarded("d", 8, 0, b"\x80")],
                         max_steps=1, control_flow=True)
    conditional = learn_stateful(model.registry, "guarded_map", guards)
    assert conditional.capability is not None and model.registry.loops("guarded_map")
    assert evaluate(conditional.capability.executable, guarded("unseen", 53, 7, bytes(range(256))), guards)

    def pair(a, b, mask, first, second):
        before = {a: first, b: second}
        return StateExample((a, b, mask), before,
                            {a: bytes(v ^ mask for v in first), b: bytes(v ^ mask for v in second)}, len(second))

    pairs = StateLesson(("path", "path", "value"),
                        [pair("a", "b", 7, b"abc", b"xyz"), pair("c", "d", 5, b"x", b"yz")],
                        [pair("e", "f", 9, b"\x00", b"\xff\x80"), pair("g", "h", 8, b"", b"xx")],
                        max_steps=2, candidate_budget=20000)
    outcome = learn_stateful(model.registry, "pair_map", pairs)
    assert outcome.capability is not None and model.registry.loops("pair_map")
    assert evaluate(outcome.capability.executable, pair("new-a", "new-b", 53, bytes(range(256)), b""), pairs)
