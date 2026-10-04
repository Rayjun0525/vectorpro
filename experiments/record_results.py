"""Acquire a file payload/length result and connect it to another learned function."""
import argparse
import json
from pathlib import Path
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def packed(data, value):
    # Evidence oracle, never used by the vector execution backend.
    return (b"VPR1" + len(data).to_bytes(8, "little") + value.to_bytes(8, "little") + data).hex()


def requests():
    def read_case(path, data):
        files = {path: data.hex(), "keep": "aa"}
        return {"inputs": [path], "before": files, "after": files, "output": packed(data, len(data))}
    def write_case(path, data):
        return {"inputs": [packed(data, len(data)), path], "before": {"keep": "aa"},
                "after": {"keep": "aa", path: data.hex()}, "output": len(data)}
    def draft(name, parameters, output, operations):
        return {"version": 1, "name": name, "description": name,
                "parameters": [{"name": n, "type": t, "role": n} for n, t in parameters],
                "output": {"type": output, "width": "W"}, "allowed_operations": operations}
    return [
        {"draft": draft("read_result", [("source", "path")], "buffer", ["file.read", "buffer.length", "record.pack"]),
         "state_lesson": {"input_types": ["path"], "output_type": "buffer", "max_steps": 3,
              "training": [read_case("a", b"ab"), read_case("b", b"xyz")],
              "validation": [read_case("empty", b""), read_case("binary", b"\xff\0more")]}},
        {"draft": draft("write_result", [("result", "buffer"), ("destination", "path")], "value", ["record.buffer", "file.write"]),
         "state_lesson": {"input_types": ["buffer", "path"], "max_steps": 2,
              "training": [write_case("a", b"ab"), write_case("b", b"xyz")],
              "validation": [write_case("empty", b""), write_case("binary", b"\xff\0more")]}}
    ]


def verify(runtime, outcomes, root):
    checks = []
    for i, data in enumerate((b"", b"\0\xff\t\n", "새로운 자료".encode(), b"x" * 2048)):
        native = root / f"native-{i}"
        native.mkdir(parents=True)
        (native / "source").write_bytes(data)
        (native / "keep").write_bytes(b"keep")
        runtime.host = HostContext(native)
        output = runtime.call_contract(outcomes[0]["contract"]["id"], {"source": "source"}, 16)
        record = output["outputs"][0]["hex"]
        assert record == packed(data, len(data))
        # Transfer portable bytes to a fresh context: raw handles do not cross calls.
        runtime.host = HostContext(native)
        extracted = runtime.call_contract(runtime.contract("record.value")["id"], {"x0": record}, 16)
        written = runtime.call_contract(outcomes[1]["contract"]["id"], {"result": record, "destination": "target"}, 16)
        assert extracted["outputs"] == written["outputs"] == [len(data)]
        assert {p.name: p.read_bytes() for p in native.iterdir()} == {"source": data, "keep": b"keep", "target": data}
        checks.append({"bytes": len(data), "passed": True, "record": record})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/record_results_verified"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Choose a new output directory")
    root.mkdir(parents=True)
    runtime = VectorRuntime.load("results/list_selection_verified/program.pt", host=HostContext(root / "unused"))
    evidence = requests()
    (root / "curriculum.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    outcomes = [runtime.teach_contract(**r) for r in evidence]
    assert all(o["status"] == "registered" for o in outcomes), outcomes
    runtime.provide_host_operations(["record.value"])
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt", host=HostContext(root))
    summary = {"learning": outcomes, "native": verify(loaded, outcomes, root)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(summary["native"]), "history": [o["history"] for o in outcomes]}))


if __name__ == "__main__":
    main()
