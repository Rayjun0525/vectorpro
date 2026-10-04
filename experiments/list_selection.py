"""Learn guarded list traversal and reduction from independent state examples."""
import json
from pathlib import Path
import random
import torch
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime

SOURCE = Path(__file__).resolve().parents[1] / "results/initial_model/program.json"


def filtered_case(src, dst, suffix, files):
    before = {src + "/" + n: v.hex() for n, v in files.items()} | {"keep": "aa", dst + "/preserved": "bb"}
    return {"inputs": [src, dst, suffix.encode().hex()], "before": before,
            "after": before | {dst + "/" + n: v.hex() for n, v in files.items() if n.endswith(suffix)},
            "before_directories": [src, dst], "after_directories": [src, dst]}


def selection_request():
    return {"draft": {"version": 1, "name": "selected_copy", "description": "Copy filenames matching supplied suffix",
        "parameters": [{"name": "source", "type": "path", "role": "source"},
                       {"name": "destination", "type": "path", "role": "destination"},
                       {"name": "suffix", "type": "buffer", "role": "UTF-8 suffix"}],
        "output": {"type": "value", "width": "W"},
        "allowed_operations": ["directory.list", "list.length", "list.get", "path.join", "file.read", "file.write", "text.ends_with"]},
      "state_lesson": {"input_types": ["path", "path", "buffer"], "list_loops": True,
        "control_flow": True, "max_steps": 9, "candidate_budget": 20000, "execution_budget": 20000,
        "training": [filtered_case("src", "out", ".txt", {"a.txt": b"yes", "b.bin": b"no"}),
                     filtered_case("in", "to", ".dat", {"a.log": b"ignore", "b.dat": b"", "c.dat": b"data"})],
        "validation": [filtered_case("empty", "result", ".txt", {}),
                       filtered_case("other", "copied", ".log", {"one.txt": b"skip", "two.log": b"yes", "three.log": b"more"})]}}


def reduction_request():
    def case(src, files):
        snapshot = {src + "/" + n: v.hex() for n, v in files.items()} | {"keep": "aa"}
        return {"inputs": [src], "before": snapshot, "after": snapshot,
                "before_directories": [src], "after_directories": [src], "output": sum(map(len, files.values()))}
    return {"draft": {"version": 1, "name": "total_bytes", "description": "Sum byte lengths in a flat directory",
        "parameters": [{"name": "source", "type": "path", "role": "source"}],
        "output": {"type": "value", "width": "W"},
        "allowed_operations": ["directory.list", "list.length", "list.get", "path.join", "file.read", "buffer.length"]},
      "state_lesson": {"input_types": ["path"], "list_loops": True, "list_reduction": True,
        "max_steps": 8, "candidate_budget": 20000, "execution_budget": 20000,
        "training": [case("a", {"first": b"abc", "second": b"ab"}), case("b", {"x": b"abcd", "y": b"", "z": b"a"})],
        "validation": [case("empty", {}), case("other", {"one": b"0123456", "two": b"12", "three": b"xxx"})]}}


def teach_add(runtime):
    torch.manual_seed(0)
    rows = [(a, b) for a in range(16) for b in range(16)]
    random.Random(0).shuffle(rows)
    def split(part):
        return {"width": 4, "operands": [list(p) for p in part], "targets": [a + b for a, b in part]}
    draft = {"version": 1, "name": "plus", "description": "Add two unsigned integers",
             "parameters": [{"name": n, "type": "value", "role": n} for n in ("left", "right")],
             "output": {"type": "value", "width": "W+1"}, "allowed_operations": []}
    result = runtime.teach_contract(draft, lesson={"training": split(rows[:64]), "validation": split(rows[64:128])})
    assert result["status"] == "registered", result
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/list_selection_verified"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Choose a new output root; preserve prior evidence")
    root.mkdir(parents=True)
    runtime = VectorRuntime.load(SOURCE, host=HostContext(root / "unused"))
    requests = [selection_request(), reduction_request()]
    (root / "curriculum.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    add = teach_add(runtime)
    outcomes = [runtime.teach_contract(**request) for request in requests]
    assert all(result["status"] == "registered" for result in outcomes), outcomes
    runtime.save(root / "program.pt")
    checks = []
    for count in (0, 1, 7, 17):
        native = root / f"native-{count}"
        (native / "source").mkdir(parents=True)
        (native / "target").mkdir()
        files = {f"{i}.{'new' if i % 2 == 0 else 'old'}": bytes([65 + i]) * i for i in range(count)}
        for name, data in files.items():
            (native / "source" / name).write_bytes(data)
        (native / "keep").write_bytes(b"keep")
        loaded = VectorRuntime.load(root / "program.pt", host=HostContext(native))
        selected = loaded.call_contract(outcomes[0]["contract"]["id"], {"source": "source", "destination": "target", "suffix": b".new".hex()}, 16)
        total = loaded.call_contract(outcomes[1]["contract"]["id"], {"source": "source"}, 16)
        snapshot = {str(p.relative_to(native)): p.read_bytes().hex() for p in native.rglob("*") if p.is_file()}
        expected = {"source/" + n: d.hex() for n, d in files.items()} | {"target/" + n: d.hex() for n, d in files.items() if n.endswith(".new")} | {"keep": b"keep".hex()}
        assert snapshot == expected and total["outputs"] == [sum(map(len, files.values()))]
        assert sorted(str(p.relative_to(native)) for p in native.rglob("*") if p.is_dir()) == ["source", "target"]
        checks.append({"count": count, "passed": True, "sum": total["outputs"], "selection": selected, "snapshot": snapshot})
    summary = {"arithmetic": add, "learning": outcomes, "native_checks": checks}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(checks), "candidates": [o["history"] for o in outcomes]}))


if __name__ == "__main__":
    main()
