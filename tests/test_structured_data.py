import json
from pathlib import Path
import pytest
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.runtime import VectorRuntime

SOURCE = Path(__file__).resolve().parents[1] / "results/initial_model/program.json"


def call(host, name, *values):
    args = tuple(host.put(v) if isinstance(v, bytes) else v for v in values)
    result = host.invoke(name, args, 16)
    return bytes(host.buffers[result]) if name not in ("list.length",) else result


def test_list_text_json_operations_are_portable(tmp_path):
    for host in (HostContext(tmp_path), MemoryHostContext({})):
        assert call(host, "list.length", b"") == 0
        assert call(host, "list.length", b"a\0\0") == 2
        assert call(host, "list.get", "한글\0line\nbreak\0".encode(), 1) == b"line\nbreak"
        assert call(host, "list.append", b"a\0", b"b") == b"a\0b\0"
        assert call(host, "path.join", b"folder", "새 파일".encode()) == "folder/새 파일".encode()
        assert call(host, "text.concat", "한".encode(), "글".encode()) == "한글".encode()
        assert call(host, "json.get", b'{"x":[3,null,true]}', b"x") == b"[3,null,true]"
        assert call(host, "json.get", b"[3,null]", b"1") == b"null"
        assert call(host, "json.text", b'"a\\nb"') == b"a\nb"


@pytest.mark.parametrize("op,values", [("list.length", [b"unterminated"]),
    ("list.append", [b"", b"bad\0item"]), ("list.get", [b"a\0", 1]),
    ("path.join", [b"folder", b"../escape"]), ("text.concat", [b"\xff", b"a"]),
    ("json.get", [b'{"a":1,"a":2}', b"a"]), ("json.get", [b"[1]", b"-1"]),
    ("json.text", [b"NaN"]), ("json.text", [b"1"])])
def test_invalid_data_never_changes_files(op, values):
    host = MemoryHostContext({"keep": b"keep"})
    with pytest.raises((ValueError, IndexError, KeyError)):
        call(host, op, *values)
    assert host.files == {"keep": b"keep"} and not host.events


def batch_case(source, target, names):
    before = {f"{source}/{name}": data.hex() for name, data in names.items()} | {"keep": b"keep".hex()}
    return {"inputs": [source, target], "before": before,
            "after": before | {f"{target}/{name}": data.hex() for name, data in names.items()},
            "before_directories": [source, target], "after_directories": [source, target]}


def lesson():
    return {"input_types": ["path", "path"], "list_loops": True, "max_steps": 8,
            "candidate_budget": 20000, "time_budget_seconds": 60, "execution_budget": 20000,
            "training": [batch_case("src", "out", {"a": b"abc", "b": b"\xff"}),
                         batch_case("in", "to", {"z": b"different"})],
            "validation": [batch_case("empty", "result", {}),
                           batch_case("unicode", "copied", {"새 파일": b"", "q": b"longer\0bytes", "line\nbreak": b"x"})]}


def draft():
    return {"version": 1, "name": "batch_copy", "description": "Copy every file in a flat directory",
        "parameters": [{"name": n, "type": "path", "role": n} for n in ("source", "destination")],
        "output": {"type": "value", "width": "W"},
        "allowed_operations": ["directory.list", "list.length", "list.get", "path.join", "file.read", "file.write"]}


def test_learn_list_body_and_reload_on_unseen_native_directory(tmp_path):
    r = VectorRuntime.load(SOURCE, host=HostContext(tmp_path))
    result = r.teach_contract(draft(), state_lesson=lesson())
    assert result["status"] == "registered", result
    assert not list(tmp_path.iterdir()) and not r.host.events
    assert r.registry.get("batch_copy").provenance["control"] == "list-iteration"
    program = tmp_path / "program.pt"
    r.save(program)
    root = tmp_path / "native"
    (root / "source").mkdir(parents=True)
    (root / "target").mkdir()
    contents = {f"unseen-{i}": bytes([i]) * (i + 1) for i in range(7)}
    for name, data in contents.items():
        (root / "source" / name).write_bytes(data)
    loaded = VectorRuntime.load(program, host=HostContext(root))
    assert loaded.contract("batch_copy")["id"] == result["contract"]["id"]
    loaded.call_contract(result["contract"]["id"], {"source": "source", "destination": "target"}, 16)
    assert {p.name: p.read_bytes() for p in (root / "target").iterdir()} == contents


def test_buffer_examples_acquire_json_field_to_file(tmp_path):
    def case(name, value):
        return {"inputs": [json.dumps({"content": value}).encode().hex(), b"content".hex(), name],
                "before": {"keep": "aa"}, "after": {"keep": "aa", name: value.encode().hex()}, "output": len(value.encode())}
    d = {"version": 1, "name": "save_field", "description": "Save JSON string field",
         "parameters": [{"name": "record", "type": "buffer", "role": "JSON bytes"},
                        {"name": "key", "type": "buffer", "role": "field UTF-8 bytes"},
                        {"name": "destination", "type": "path", "role": "destination"}],
         "output": {"type": "value", "width": "W"}, "allowed_operations": ["json.get", "json.text", "file.write"]}
    r = VectorRuntime(host=HostContext(tmp_path))
    evidence = {"input_types": ["buffer", "buffer", "path"], "max_steps": 3,
                "training": [case("a", "abc"), case("b", "line\nbreak")],
                "validation": [case("c", ""), case("d", "한글")]}
    result = r.teach_contract(d, state_lesson=evidence)
    assert result["status"] == "registered", result
    assert not list(tmp_path.iterdir())
    r.call_contract(result["contract"]["id"], {"record": b'{"other":3,"content":"unseen"}'.hex(),
                    "key": b"content".hex(), "destination": "out"}, 16)
    assert (tmp_path / "out").read_bytes() == b"unseen"


def test_failed_list_search_keeps_native_and_registry_unchanged(tmp_path):
    r = VectorRuntime.load(SOURCE, host=HostContext(tmp_path))
    before = r.registry.to_data()
    evidence = lesson()
    evidence["candidate_budget"] = 1
    result = r.teach_contract(draft(), state_lesson=evidence)
    assert result["status"] == "learning_failed" and not result["registered"]
    assert r.registry.to_data() == before and not list(tmp_path.iterdir())
    assert not r.host.buffers and not r.host.events
