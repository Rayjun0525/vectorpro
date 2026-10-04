"""Learn captured-process-output routing from recorded evidence, then run on Linux."""
import argparse
import json
from pathlib import Path
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def request_bytes(spec):
    return json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()


def lesson_request():
    def case(name, data, spec):
        request = request_bytes(spec).hex()
        return {"inputs": [request, data.hex(), name], "before": {"keep": "aa"},
                "after": {"keep": "aa", name: data.upper().hex()}, "output": len(data),
                "processes": [{"request": request, "stdin": data.hex(), "stdout": data.upper().hex(), "stderr": "", "code": 0}]}
    return {"draft": {"version": 1, "name": "capture_to_file", "description": "Run supplied process and save its stdout",
                "parameters": [{"name": n, "type": t, "role": n} for n, t in
                    (("request", "buffer"), ("stdin", "buffer"), ("destination", "path"))],
                "output": {"type": "value", "width": "W"},
                "allowed_operations": ["process.run", "process.stdout", "file.write"]},
            "state_lesson": {"input_types": ["buffer", "buffer", "path"], "max_steps": 3, "candidate_budget": 20000,
                "training": [case("a", b"abc", {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}),
                             case("b", b"longer\0data", {"argv": ["/usr/bin/tr", "a-z", "A-Z"], "env": {"DEMO": "two"}})],
                "validation": [case("empty", b"", {"argv": ["/usr/bin/tr", "a-z", "A-Z"], "timeout": 2}),
                               case("binary", b"\xff\0more\n", {"argv": ["/usr/bin/tr", "a-z", "A-Z"], "env": {"DEMO": "validation"}})]}}


def verify(runtime, contract, root):
    checks = []
    for i, data in enumerate((b"", b"\xff\0\t\n", "다른 입력".encode(), b"x" * 4096)):
        native = root / f"native-{i}"
        native.mkdir(parents=True)
        (native / "keep").write_bytes(b"keep")
        runtime.host = HostContext(native)
        result = runtime.call_contract(contract["id"], {"request": request_bytes({"argv": ["/bin/cat"], "timeout": 3}).hex(),
                                       "stdin": data.hex(), "destination": "output"}, 16)
        assert result["outputs"] == [len(data)]
        assert {p.name: p.read_bytes() for p in native.iterdir()} == {"keep": b"keep", "output": data}
        checks.append({"bytes": len(data), "passed": True, "call": result})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/process_results"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Preserve evidence; select another output directory")
    root.mkdir(parents=True)
    evidence = lesson_request()
    (root / "curriculum.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    runtime = VectorRuntime.load("results/record_results_verified/program.pt", host=HostContext(root / "unused"))
    outcome = runtime.teach_contract(**evidence)
    assert outcome["status"] == "registered", outcome
    runtime.provide_host_operations(["process.stderr", "process.code"])
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt", host=HostContext(root))
    summary = {"learning": outcome, "native": verify(loaded, outcome["contract"], root)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(summary["native"]), "history": outcome["history"]}))


if __name__ == "__main__":
    main()
