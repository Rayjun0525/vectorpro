import pytest
import torch

from vectorpro.host import ARITIES, BASE_OPERATIONS, HostContext, MemoryHostContext
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate
from vectorpro.runtime import VectorRuntime


def snapshot(root):
    return ({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()},
            {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()})


def invoke(context, op, *paths):
    return context.invoke(op, tuple(context.put(p.encode("utf-8")) for p in paths), 16)


def test_native_memory_parity_for_directory_and_file_transitions(tmp_path):
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "a").write_bytes(b"a")
    (tmp_path / "tree" / "줄\n바꿈").write_bytes(b"z")
    native = HostContext(tmp_path)
    memory = MemoryHostContext(snapshot(tmp_path)[0], directories=["tree"])
    transitions = [("path.exists", ("missing",)), ("path.exists", ("tree",)),
                   ("directory.create", ("tree/empty",)), ("directory.list", ("tree",)),
                   ("file.move", ("tree/a", "tree/new")), ("file.remove", ("tree/new",)),
                   ("directory.remove", ("tree/empty",))]
    for op, args in transitions:
        a, b = invoke(native, op, *args), invoke(memory, op, *args)
        if op == "directory.list":
            assert bytes(native.buffers[a]) == bytes(memory.buffers[b]) == "a\0empty\0줄\n바꿈\0".encode()
        else:
            assert a == b
        assert snapshot(tmp_path) == (memory.files, memory.directories)


@pytest.mark.parametrize("operation,args", [
    ("file.remove", ("tree",)), ("directory.remove", ("tree",)),
    ("directory.create", ("tree",)), ("directory.create", ("missing/child",)),
    ("file.move", ("tree/a", "missing/new")), ("file.move", ("tree", "new")),
    ("directory.list", ("tree/a",)), ("file.remove", ("absent",)),
])
def test_invalid_filesystem_operations_have_native_memory_parity(tmp_path, operation, args):
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/a").write_bytes(b"keep")
    contexts = [HostContext(tmp_path), MemoryHostContext({"tree/a": b"keep"})]
    before = snapshot(tmp_path)
    for context in contexts:
        with pytest.raises(OSError):
            invoke(context, operation, *args)
        assert context.events == []
    assert snapshot(tmp_path) == before
    assert (contexts[1].files, contexts[1].directories) == before


def test_extensions_are_opt_in_and_existing_ids_stay_stable():
    r = VectorRuntime()
    r.provide_host_operations()
    assert {c["name"] for c in r.contracts()} == set(BASE_OPERATIONS)
    before = r.contract("file.read")
    r.provide_host_operations(ARITIES)
    assert r.contract("file.read") == before
    assert r.contract("directory.list")["output"]["type"] == "buffer"
    with pytest.raises(ValueError):
        r.provide_host_operations(["directory.unknown"])


def draft(name, parameters, operations):
    return {"version": 1, "name": name, "description": name,
            "parameters": [{"name": n, "type": "path", "role": n} for n in parameters],
            "output": {"type": "value", "width": "W"}, "allowed_operations": operations}


def directory_case(folder, source, destination, data):
    before = {source: data.hex(), "preserved": b"keep".hex()}
    return {"inputs": [folder, source, destination], "before": before,
            "after": before | {destination: data.hex()}, "after_directories": [folder], "output": len(data)}


def lesson():
    return {"input_types": ["path", "path", "path"], "max_steps": 3, "candidate_budget": 20000,
            "training": [directory_case("box", "src", "box/out", b"abc"),
                         directory_case("new", "input", "new/result", b"\x00\xff")],
            "validation": [directory_case("empty", "blank", "empty/copy", b""),
                           directory_case("other", "long", "other/copied", b"unseen bytes")]}


def test_learn_directory_creation_and_copy_without_native_learning_then_reload(tmp_path):
    r = VectorRuntime(host=HostContext(tmp_path))
    outcome = r.teach_contract(draft("make_copy", ["folder", "source", "destination"],
                                   ["directory.create", "file.read", "file.write"]), state_lesson=lesson())
    assert outcome["status"] == "registered", outcome
    assert not list(tmp_path.iterdir()) and not r.host.events
    assert r.registry.get("make_copy").history[-1]["steps"] == 3
    program = tmp_path / "program.pt"
    r.save(program)
    root = tmp_path / "native"
    root.mkdir()
    (root / "source").write_bytes(bytes(range(256)))
    (root / "preserved").write_bytes(b"keep")
    loaded = VectorRuntime.load(program, host=HostContext(root))
    c = loaded.contract("make_copy")
    assert c["id"] == outcome["contract"]["id"]
    result = loaded.call_contract(c["id"], {"folder": "새 폴더", "source": "source", "destination": "새 폴더/result"}, 16)
    assert result["outputs"] == [256]
    assert snapshot(root) == ({"source": bytes(range(256)), "preserved": b"keep", "새 폴더/result": bytes(range(256))}, {"새 폴더"})


def test_directory_changes_are_part_of_exact_goal_not_ignored():
    r = VectorRuntime()
    r.provide_host_operations(["directory.create"])
    cap = r.registry.get("directory.create")
    case = StateExample(("new",), {}, {}, 1)
    settings = StateLesson(("path",), [case], [StateExample(("other",), {}, {})])
    assert not evaluate(cap.executable, case, settings)
    case.after_directories = ("new",)
    assert evaluate(cap.executable, case, settings)


def test_path_escape_and_listing_limit_are_rejected_before_side_effects(tmp_path):
    r = VectorRuntime(host=HostContext(tmp_path))
    r.provide_host_operations(ARITIES)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="path"):
        r.call_contract(r.contract("directory.create")["id"], {"x0": "../escape"}, 16)
    assert snapshot(tmp_path) == before and not r.host.buffers and not r.host.events
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree/longname").write_bytes(b"")
    for context in (HostContext(tmp_path, max_buffer_bytes=5), MemoryHostContext({"tree/longname": b""}, max_buffer_bytes=5)):
        with pytest.raises(ValueError, match="listing"):
            invoke(context, "directory.list", "tree")
        assert not context.events


def test_catalog_checks_full_directory_snapshots_without_native_access(tmp_path):
    from vectorpro.semantic_catalog import TensorCatalog
    r = VectorRuntime(host=HostContext(tmp_path))
    r.provide_host_operations(["directory.create"])
    catalog = TensorCatalog(r, lambda texts: torch.ones(len(texts), 4), {"test": "uniform"})
    request = {"query": "create directory", "rows": [[{"utf8": "native"}]], "width": 16,
               "state_validation": [{"inputs": [{"utf8": "probe"}], "before": {}, "after": {}, "output": 1}]}
    assert catalog.resolve(**request)["status"] == "needs_learning_examples"
    request["state_validation"][0]["after_directories"] = ["probe"]
    assert catalog.resolve(**request)["status"] == "matched"
    assert not list(tmp_path.iterdir()) and not r.host.events and not r.host.buffers


def test_directory_state_validation_rejects_conflicts_and_incomplete_goals():
    with pytest.raises(ValueError, match="both"):
        MemoryHostContext({"parent": b"file", "parent/child": b"impossible"})
    with pytest.raises(ValueError, match="both"):
        MemoryHostContext({"same": b"file"}, directories=["same"])
    context = MemoryHostContext({})
    data = context.put(b"bytes")
    path = context.put(b"missing/child")
    with pytest.raises(FileNotFoundError):
        context.invoke("file.write", (path, data), 16)
    assert not context.files and not context.directories and not context.events
