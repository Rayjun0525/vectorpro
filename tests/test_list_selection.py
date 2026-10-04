import pytest
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.runtime import VectorRuntime
from experiments.list_selection import SOURCE, selection_request, reduction_request, teach_add

@pytest.mark.parametrize("text,suffix,expected", [("abc.txt", ".txt", 1), ("한글", "글", 1), ("abc", "", 1), ("x", "xx", 0)])
def test_utf8_suffix_predicate(text, suffix, expected):
    host = MemoryHostContext({})
    assert host.invoke("text.ends_with", (host.put(text.encode()), host.put(suffix.encode())), 16) == expected


def test_acquire_filtered_copy_and_preserve_unselected_files(tmp_path):
    r = VectorRuntime.load(SOURCE, host=HostContext(tmp_path))
    outcome = r.teach_contract(**selection_request())
    assert outcome["status"] == "registered", outcome
    assert not list(tmp_path.iterdir()) and not r.host.events
    path = tmp_path / "program.pt"
    r.save(path)
    root = tmp_path / "native"
    (root / "source").mkdir(parents=True)
    (root / "target").mkdir()
    for name, data in {"a.new": b"", "b.txt": b"ignore", "c.new": b"new", "새 파일.new": b"unicode"}.items():
        (root / "source" / name).write_bytes(data)
    (root / "target/keep").write_bytes(b"keep")
    loaded = VectorRuntime.load(path, host=HostContext(root))
    loaded.call_contract(outcome["contract"]["id"], {"source": "source", "destination": "target", "suffix": b".new".hex()}, 16)
    assert {p.name: p.read_bytes() for p in (root / "target").iterdir()} == {"keep": b"keep", "a.new": b"", "c.new": b"new", "새 파일.new": b"unicode"}
    before = {p.name: p.read_bytes() for p in (root / "target").iterdir()}
    assert loaded.call_contract(outcome["contract"]["id"], {"source": "source", "destination": "target", "suffix": b".absent".hex()}, 16)["outputs"] == [0]
    assert {p.name: p.read_bytes() for p in (root / "target").iterdir()} == before


def test_acquire_reduction_using_learned_arithmetic(tmp_path):
    r = VectorRuntime.load(SOURCE, host=HostContext(tmp_path))
    teach_add(r)
    result = r.teach_contract(**reduction_request())
    assert result["status"] == "registered", result
    assert r.registry.get("total_bytes").provenance["control"] == "list-reduction"
    assert not list(tmp_path.iterdir()) and not r.host.events
    path = tmp_path / "program.pt"
    r.save(path)
    root = tmp_path / "native"
    (root / "source").mkdir(parents=True)
    for i in range(9):
        (root / "source" / str(i)).write_bytes(bytes([i]) * (i + 1))
    loaded = VectorRuntime.load(path, host=HostContext(root))
    assert loaded.call_contract(result["contract"]["id"], {"source": "source"}, 16)["outputs"] == [45]
    assert all(e["operation"] != "file.write" for e in loaded.host.events)


@pytest.mark.parametrize("flag,correct", [(0, True), (1, True), (0, False), (1, False)])
def test_candidate_prefilter_agrees_with_tensor_and_discarded_writes(flag, correct):
    from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, prefilter_instructions
    from vectorpro.learning.plan import LearningPlan, OutputWidth
    from vectorpro.learning.registry import program_provenance
    from vectorpro.machine import Instr, assemble
    runtime = VectorRuntime.load(SOURCE)
    instructions = (Instr(test="x2", then="write", otherwise="done"),
                    Instr("file.read", ("x0",), "data", label="write"),
                    Instr("file.write", ("x1", "data"), None), Instr(label="done"))
    registers = {n: "zero" for n in ("x0", "x1", "x2", "data", "result")}
    before = {"source": b"", "keep": b"keep"}
    after = before | ({"target": b""} if flag else {})
    if not correct:
        after = after | {"unexpected": b"x"}
    case = StateExample(("source", "target", flag), before, after, 0)
    lesson = StateLesson(("path", "path", "value"), [case], [], execution_budget=50)
    program = assemble(instructions, ["x0", "x1", "x2"], registers, "result", runtime.registry.key_of)
    executable = runtime.registry.build(LearningPlan("check", "check", 3, OutputWidth.SAME),
        program_provenance(program, "test", execution_budget=50))
    assert prefilter_instructions(runtime.registry, instructions, registers, "result", case, lesson, lambda: False) == correct
    assert evaluate(executable, case, lesson) == correct
    assert not prefilter_instructions(runtime.registry, instructions, registers, "result", case, lesson, lambda: True)

