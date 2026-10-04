"""Acquire process stdout and then compose two acquired calls from observations."""
import argparse
import json
import sys
from pathlib import Path
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime
from vectorpro.process_host import encode
from experiments.process_results import request_bytes

SOURCE = "results/process_results_verified_ordered/program.pt"


def evidence(requests, stdin, outputs, name):
    current = stdin
    fixtures = []
    for spec, output in zip(requests, outputs):
        fixtures.append({"request": request_bytes(spec).hex(), "stdin": current.hex(),
                         "stdout": output.hex(), "stderr": "", "code": 0})
        current = output
    return {"inputs": [*(request_bytes(spec).hex() for spec in requests), stdin.hex()],
            "before": {name: "aa"}, "after": {name: "aa"}, "output": current.hex(), "processes": fixtures}


def lessons():
    upper = {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}
    replace = {"argv": ["/usr/bin/tr", "AB", "XY"]}
    def draft(name, params):
        return {"version": 1, "name": name, "description": name,
                "parameters": [{"name": p, "type": "buffer", "role": p} for p in params],
                "output": {"type": "buffer", "width": "W"},
                "allowed_operations": ["process.run", "process.stdout"]}
    common = {"output_type": "buffer", "max_steps": 2, "candidate_budget": 20000}
    return [
        {"draft": draft("run_stdout", ["request", "stdin"]), "state_lesson": {
            **common, "input_types": ["buffer", "buffer"],
            "training": [evidence([upper], b"abc", [b"ABC"], "one"),
                         evidence([upper], b"different", [b"DIFFERENT"], "two")],
            "validation": [evidence([upper], b"", [b""], "empty"),
                           evidence([upper], b"more\0\xff", [b"MORE\0\xff"], "binary")]}},
        {"draft": draft("chain_stdout", ["first", "second", "stdin"]), "state_lesson": {
            **common, "input_types": ["buffer", "buffer", "buffer"],
            "training": [evidence([upper, replace], b"abc", [b"ABC", b"XYC"], "one"),
                         evidence([upper, replace], b"baba", [b"BABA", b"YXYX"], "two")],
            "validation": [evidence([upper, replace], b"", [b"", b""], "empty"),
                           evidence([upper, replace], b"ab\0\xff", [b"AB\0\xff", b"XY\0\xff"], "binary")]}}
    ]


def guarded_lesson():
    def case(name, data, code):
        before = {"keep": "aa"}
        return {"inputs": [encode(data, b"diagnostic", code).hex(), name], "before": before,
                "after": before | ({name: data.hex()} if code == 0 else {}), "output": len(data) if code == 0 else 0}
    return {"draft": {"version": 1, "name": "save_success", "description": "Save stdout only when process exit code is zero",
                 "parameters": [{"name": "result", "type": "buffer", "role": "process result"},
                                {"name": "destination", "type": "path", "role": "destination"}],
                 "output": {"type": "value", "width": "W"},
                 "allowed_operations": ["process.code", "process.stdout", "file.write"]},
            "state_lesson": {"input_types": ["buffer", "path"], "max_steps": 2, "control_flow": True, "branches_only": True,
                 "candidate_budget": 20000,
                 "training": [case("ok", b"abc", 0), case("fail", b"should-not-save", 7)],
                 "validation": [case("empty", b"", 0), case("signal", b"no", -15)]}}


def saving_lesson():
    request = guarded_lesson()
    request["draft"]["name"] = "save_stdout"
    request["draft"]["description"] = "Save stdout from a supplied process result"
    request["draft"]["allowed_operations"] = ["process.stdout", "file.write"]
    lesson = request["state_lesson"]
    lesson["control_flow"] = False
    lesson["branches_only"] = False
    for case in lesson["training"] + lesson["validation"]:
        from vectorpro.process_host import decode
        data = bytes.fromhex(decode(bytes.fromhex(case["inputs"][0]))["stdout"])
        case["after"] = case["before"] | {case["inputs"][1]: data.hex()}
        case["output"] = len(data)
    return request


def verify(runtime, outcome, root):
    cat = {"argv": ["/bin/cat"]}
    upper = {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}
    cases = [(cat, cat, b"", b""), (cat, cat, b"\xff\0\t\n", b"\xff\0\t\n"),
             (cat, upper, "새 입력 abc".encode(), "새 입력 ABC".encode()),
             (upper, cat, b"abc" * 2048, b"ABC" * 2048)]
    checks = []
    for i, (first, second, data, expected) in enumerate(cases):
        native = root / f"native-{i}"
        native.mkdir(parents=True)
        (native / "keep").write_bytes(b"keep")
        runtime.host = HostContext(native)
        result = runtime.call_contract(outcome["contract"]["id"], {
            "first": request_bytes(first).hex(), "second": request_bytes(second).hex(), "stdin": data.hex()}, 16)
        assert result["outputs"] == [{"hex": expected.hex()}]
        assert [e["operation"] for e in result["effects"]] == ["process.run", "process.stdout"] * 2
        assert {p.name: p.read_bytes() for p in native.iterdir()} == {"keep": b"keep"}
        assert not any(p.is_dir() for p in native.iterdir())
        checks.append({"bytes": len(data), "passed": True, "call": result})
    return checks


def verify_guard(runtime, outcome, root):
    checks = []
    for i, (code, data, previous) in enumerate(((0, b"", b"old"), (0, "새 자료".encode(), b"old"),
                                               (19, b"do-not-save", b"existing"), (-15, b"signal-output", None))):
        native = root / f"guard-{i}"
        native.mkdir(parents=True)
        (native / "keep").write_bytes(b"keep")
        if previous is not None:
            (native / "target").write_bytes(previous)
        runtime.host = HostContext(native)
        script = ("import os,signal,sys;sys.stdout.buffer.write(sys.stdin.buffer.read());sys.stdout.flush();"
                  "sys.stderr.write('diagnostic');sys.stderr.flush();"
                  "code=int(sys.argv[1]);os.kill(os.getpid(),-code) if code<0 else sys.exit(code)")
        result = runtime.call_contract(runtime.contract("process.run")["id"], {
            "x0": request_bytes({"argv": [sys.executable, "-c", script, str(code)]}).hex(), "x1": data.hex()}, 16)
        guarded = runtime.call_contract(outcome["contract"]["id"], {
            "result": result["outputs"][0]["hex"], "destination": "target"}, 16)
        expected = {"keep": b"keep"}
        if code == 0:
            expected["target"] = data
        elif previous is not None:
            expected["target"] = previous
        assert {p.name: p.read_bytes() for p in native.iterdir()} == expected
        assert not any(p.is_dir() for p in native.iterdir())
        assert guarded["outputs"] == [len(data) if code == 0 else 0]
        assert sum(e["operation"] == "file.write" for e in guarded["effects"]) == int(code == 0)
        checks.append({"code": code, "passed": True, "result": result, "guard": guarded})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/process_chain_verified"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Choose another output directory; preserve evidence")
    root.mkdir(parents=True)
    requests = [*lessons(), saving_lesson(), guarded_lesson()]
    (root / "curriculum.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
    runtime = VectorRuntime.load(SOURCE, host=HostContext(root / "unused"))
    outcomes = [runtime.teach_contract(**request) for request in requests]
    assert all(o["status"] == "registered" for o in outcomes), outcomes
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt", host=HostContext(root))
    summary = {"learning": outcomes, "native": verify(loaded, outcomes[1], root),
               "guard": verify_guard(loaded, outcomes[3], root)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(summary["native"]), "guard_passed": len(summary["guard"]),
                      "history": [o["history"] for o in outcomes]}))


if __name__ == "__main__":
    main()
