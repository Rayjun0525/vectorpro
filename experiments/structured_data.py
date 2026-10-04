"""Learn a typed list body and JSON extraction, then execute one saved tensor file."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def case(source, target, files):
    before = {source + "/" + name: data.hex() for name, data in files.items()} | {"keep": b"preserved".hex()}
    return {"inputs": [source, target], "before": before,
            "after": before | {target + "/" + name: data.hex() for name, data in files.items()},
            "before_directories": [source, target], "after_directories": [source, target]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/structured_data"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("preserve results; choose another output root")
    root.mkdir(parents=True)
    draft = {"version": 1, "name": "batch_copy", "description": "Copy all files of a flat directory",
        "parameters": [{"name": n, "type": "path", "role": n} for n in ("source", "destination")],
        "output": {"type": "value", "width": "W"},
        "allowed_operations": ["directory.list", "list.length", "list.get", "path.join", "file.read", "file.write"]}
    evidence = {"input_types": ["path", "path"], "list_loops": True, "max_steps": 8,
        "candidate_budget": 20000, "execution_budget": 20000,
        "training": [case("src", "out", {"a": b"abc", "b": b"\xff"}), case("in", "to", {"z": b"different"})],
        "validation": [case("empty", "result", {}), case("unicode", "copy", {"새 파일": b"", "q": b"longer\0bytes", "line\nbreak": b"x"})]}
    def field_case(name, value):
        return {"inputs": [json.dumps({"content": value}).encode().hex(), b"content".hex(), name],
                "before": {"keep": "aa"}, "after": {"keep": "aa", name: value.encode().hex()}, "output": len(value.encode())}
    field_draft = {"version": 1, "name": "save_field", "description": "Write a JSON string field to a file",
        "parameters": [{"name": "record", "type": "buffer", "role": "JSON bytes"}, {"name": "key", "type": "buffer", "role": "field UTF-8 bytes"},
                       {"name": "destination", "type": "path", "role": "destination"}],
        "output": {"type": "value", "width": "W"}, "allowed_operations": ["json.get", "json.text", "file.write"]}
    field_lesson = {"input_types": ["buffer", "buffer", "path"], "max_steps": 3, "candidate_budget": 20000,
        "training": [field_case("a", "abc"), field_case("b", "line\nbreak")],
        "validation": [field_case("c", ""), field_case("d", "한글")]}
    curriculum = [{"draft": draft, "state_lesson": evidence}, {"draft": field_draft, "state_lesson": field_lesson}]
    (root / "curriculum.json").write_text(json.dumps(curriculum, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = VectorRuntime.load("results/initial_model/program.json")
    learned = [runtime.teach_contract(**request) for request in curriculum]
    assert all(r["status"] == "registered" for r in learned), learned
    program = root / "program.pt"
    runtime.save(program)
    heldout = []
    for n in (0, 1, 7, 17):
        native = root / f"native-{n}"
        (native / "source").mkdir(parents=True)
        (native / "target").mkdir()
        (native / "keep").write_bytes(b"preserved")
        contents = {f"새 파일-{i}": bytes([i]) * (i + 1) for i in range(n)}
        for name, data in contents.items():
            (native / "source" / name).write_bytes(data)
        loaded = VectorRuntime.load(program, host=HostContext(native))
        for result in learned:
            assert loaded.contract(result["contract"]["name"])["id"] == result["contract"]["id"]
        batch = loaded.call_contract(learned[0]["contract"]["id"], {"source": "source", "destination": "target"}, 16)
        field = loaded.call_contract(learned[1]["contract"]["id"],
            {"record": json.dumps({"extra": [1, True], "message": "새 결과\n"}).encode().hex(),
             "key": b"message".hex(), "destination": "metadata.txt"}, 16)
        files = {p.relative_to(native).as_posix(): p.read_bytes().hex() for p in native.rglob("*") if p.is_file()}
        expected = {"keep": b"preserved".hex(), "metadata.txt": "새 결과\n".encode().hex()}
        expected.update({folder + "/" + name: data.hex() for folder in ("source", "target") for name, data in contents.items()})
        directories = sorted(p.relative_to(native).as_posix() for p in native.rglob("*") if p.is_dir())
        assert files == expected and directories == ["source", "target"]
        heldout.append({"items": n, "batch": batch, "field": field, "files": files, "directories": directories, "passed": True})
    result = {"learned": learned, "heldout": heldout, "passed": len(heldout),
              "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
              "llm_imported": "llama_cpp" in sys.modules, "encoder_imported": "transformers" in sys.modules,
              "scope": "flat files only, reverse list iterator with a searched typed expression body; not general list processing"}
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "program_sha256")}))


if __name__ == "__main__":
    main()
