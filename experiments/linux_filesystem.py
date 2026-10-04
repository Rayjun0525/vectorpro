"""Learn filesystem procedures from states, then use one tensor file on Linux."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def case(inputs, before, after, output, before_dirs=(), after_dirs=()):
    return {"inputs": inputs, "before": {p: data.hex() for p, data in before.items()},
            "after": {p: data.hex() for p, data in after.items()}, "output": output,
            "before_directories": list(before_dirs), "after_directories": list(after_dirs)}


def curricula():
    keep = {"keep": b"preserved"}
    def make_copy(folder, source, target, data):
        return case([folder, source, target], keep | {source: data},
                    keep | {source: data, target: data}, len(data), after_dirs=[folder])
    def listing(folder, target, filename):
        files = keep | {folder + "/" + filename: b"data"}
        data = filename.encode() + b"\0"
        return case([folder, target], files, files | {target: data}, len(data), [folder], [folder])
    def remove(name):
        return case([name], keep | {name: b"delete"}, keep, 1)
    def move(source, target):
        return case([source, target], keep | {source: b"move", target: b"old"}, keep | {target: b"move"}, 1)
    def create(name):
        return case([name], keep, keep, 1, after_dirs=[name])
    def rmdir(name):
        return case([name], keep, keep, 1, before_dirs=[name])
    def exists(name, present):
        files = keep | ({name: b"exists"} if present else {})
        return case([name], files, files, int(present))
    definitions = [
        ("make_copy", ["folder", "source", "destination"], ["directory.create", "file.read", "file.write"], 3,
         [make_copy("box", "src", "box/out", b"abc"), make_copy("other", "input", "other/result", b"\0\xff")],
         [make_copy("empty", "blank", "empty/copy", b""), make_copy("long", "data", "long/copied", b"longer bytes")]),
        ("save_listing", ["folder", "destination"], ["directory.list", "file.write"], 2,
         [listing("one", "a.txt", "x"), listing("two", "b.txt", "two names")],
         [listing("three", "c.txt", "새 파일"), listing("four", "d.txt", "line\nbreak")]),
        ("remove_file", ["target"], ["file.remove"], 1, [remove("a"), remove("b")], [remove("c"), remove("d")]),
        ("move_file", ["source", "destination"], ["file.move"], 1, [move("a", "b"), move("c", "d")], [move("e", "f"), move("g", "h")]),
        ("create_folder", ["folder"], ["directory.create"], 1, [create("a"), create("b")], [create("c"), create("d")]),
        ("remove_folder", ["folder"], ["directory.remove"], 1, [rmdir("a"), rmdir("b")], [rmdir("c"), rmdir("d")]),
        ("exists", ["target"], ["path.exists"], 1, [exists("a", True), exists("b", False)], [exists("c", False), exists("d", True)]),
    ]
    for name, parameters, operations, steps, training, validation in definitions:
        yield {"draft": {"version": 1, "name": name, "description": name,
                         "parameters": [{"name": p, "type": "path", "role": p} for p in parameters],
                         "output": {"type": "value", "width": "W"}, "allowed_operations": operations},
               "state_lesson": {"input_types": ["path"] * len(parameters), "max_steps": steps,
                                "candidate_budget": 20000, "training": training, "validation": validation}}


def snapshot(root):
    return {"files": {p.relative_to(root).as_posix(): p.read_bytes().hex() for p in sorted(root.rglob("*")) if p.is_file()},
            "directories": sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/linux_filesystem"))
    root = parser.parse_args().output
    if platform.system() != "Linux":
        raise RuntimeError("run this experiment inside the existing Linux test container")
    if root.exists():
        raise RuntimeError("preserve prior results; select another --output directory")
    root.mkdir(parents=True)
    curriculum = list(curricula())
    (root / "curriculum.json").write_text(json.dumps(curriculum, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime = VectorRuntime()
    learned = []
    stable_ids = {}
    for request in curriculum:
        result = runtime.teach_contract(**request)
        if result["status"] != "registered":
            raise AssertionError(result)
        assert all(runtime.contract(n)["id"] == identity for n, identity in stable_ids.items())
        stable_ids[result["contract"]["name"]] = result["contract"]["id"]
        learned.append(result)
    program = root / "program.pt"
    runtime.save(program)
    cases = []
    for index, data in enumerate((b"", b"unseen content\0\xff", bytes(range(256)), b"a" * 4096)):
        native = root / f"native-{index}"
        native.mkdir()
        (native / "source.bin").write_bytes(data)
        (native / "keep").write_bytes(b"preserved")
        loaded = VectorRuntime.load(program, host=HostContext(native))
        assert all(loaded.contract(n)["id"] == identity for n, identity in stable_ids.items())
        calls = []
        def call(name, arguments, expected):
            result = loaded.call_contract(stable_ids[name], arguments, 16)
            assert result["outputs"] == [expected]
            calls.append({"name": name, "arguments": arguments, "result": result})
        folder = "새 폴더"
        destination = folder + "/copied.bin"
        call("make_copy", {"folder": folder, "source": "source.bin", "destination": destination}, len(data))
        assert (native / destination).read_bytes() == data
        call("save_listing", {"folder": folder, "destination": "listing.bin"}, len(b"copied.bin\0"))
        assert (native / "listing.bin").read_bytes() == b"copied.bin\0"
        call("move_file", {"source": destination, "destination": folder + "/moved.bin"}, 1)
        assert not (native / destination).exists() and (native / folder / "moved.bin").read_bytes() == data
        call("exists", {"target": folder + "/moved.bin"}, 1)
        call("remove_file", {"target": folder + "/moved.bin"}, 1)
        call("exists", {"target": folder + "/moved.bin"}, 0)
        call("remove_folder", {"folder": folder}, 1)
        call("create_folder", {"folder": "empty-new"}, 1)
        call("remove_folder", {"folder": "empty-new"}, 1)
        final = snapshot(native)
        assert final == {"files": {"source.bin": data.hex(), "keep": b"preserved".hex(), "listing.bin": b"copied.bin\0".hex()}, "directories": []}
        cases.append({"root": str(native), "calls": calls, "final_snapshot": final, "passed": True})
    report = {"platform": platform.platform(), "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
              "learned": learned, "held_out": cases, "passed": len(cases),
              "llm_imported": "llama_cpp" in sys.modules, "encoder_imported": "transformers" in sys.modules,
              "scope": "filesystem examples only; not general Linux or natural-language correctness"}
    (root / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(cases), "learned": len(learned), "program_sha256": report["program_sha256"]}))


if __name__ == "__main__":
    main()
