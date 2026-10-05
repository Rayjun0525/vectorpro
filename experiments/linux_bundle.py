"""Acquire and verify bounded Linux pipelines, process states, HTTP and system queries."""
import argparse
import json
import platform
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime
from experiments.process_results import request_bytes

SOURCE = "results/process_chain_final_verified/program.pt"


def draft(name, parameters, output, operations):
    return {"version": 1, "name": name, "description": name,
            "parameters": [{"name": n, "type": t, "role": n} for n, t in parameters],
            "output": {"type": output, "width": "W"}, "allowed_operations": operations}


def lesson(parameters, training, validation, output="value"):
    return {"input_types": [t for _, t in parameters], "training": training, "validation": validation,
            "max_steps": 2, "candidate_budget": 20000, "output_type": output}


def pipeline_spec(fail=False):
    first = {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}
    if fail:
        first = {"argv": [sys.executable, "-c", "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read());sys.exit(7)"]}
    return {"stages": [first, {"argv": ["/bin/cat"]}], "timeout": 3}


def pipeline_case(name, data, fail=False):
    spec = pipeline_spec(fail)
    output = data if fail else data.upper()
    request = request_bytes(spec).hex()
    before = {"keep": "aa"}
    return {"inputs": [request, data.hex(), name], "before": before,
            "after": before | ({} if fail else {name: output.hex()}), "output": 0 if fail else len(output),
            "processes": [{"operation": "process.pipeline", "request": request, "stdin": data.hex(),
                           "stdout": output.hex(), "stderr": "", "code": 7 if fail else 0}]}


WRITE_SCRIPT = "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);pathlib.Path('empty').mkdir(exist_ok=True);p.write_bytes(sys.stdin.buffer.read())"


def created_case(name, data):
    spec = {"argv": [sys.executable, "-c", WRITE_SCRIPT, name]}
    request = request_bytes(spec).hex()
    before = {"keep": "aa"}
    after = before | {name: data.hex()}
    directories = ["empty", str(Path(name).parent)]
    return {"inputs": [request, data.hex(), name], "before": before, "after": after,
            "before_directories": [], "after_directories": directories, "output": data.hex(),
            "processes": [{"request": request, "stdin": data.hex(), "stdout": "", "stderr": "", "code": 0,
                           "before": before, "before_directories": [], "after": after, "after_directories": directories}]}


def http_case(name, data, status=200):
    spec = {"url": "http://recorded.invalid/" + name}
    request = request_bytes(spec).hex()
    return {"inputs": [request], "before": {name: "aa"}, "after": {name: "aa"}, "output": data.hex(),
            "observations": [{"operation": "network.http", "request": request,
                              "response": request_bytes({"status": status, "body": data.hex()}).hex()}]}


def system_case(name, query, value):
    return {"inputs": [query.encode().hex()], "before": {name: "aa"}, "after": {name: "aa"},
            "output": value.encode().hex(), "observations": [{"operation": "system.info", "request": query.encode().hex(),
                                                                  "response": json.dumps(value).encode().hex()}]}


def integrated_case(name, data, fail=False):
    pipe = pipeline_case(name, data, fail)
    http = http_case(name, data)
    pipe["inputs"] = [http["inputs"][0], pipe["inputs"][0], name]
    pipe["observations"] = http["observations"]
    return pipe


def curriculum():
    pipe_ops = ["process.pipeline", "process.code", "process.stdout", "file.write"]
    http_ops = ["network.http", "network.require_body"]
    p = [("request", "buffer"), ("stdin", "buffer"), ("destination", "path")]
    c = [("request", "buffer"), ("stdin", "buffer"), ("source", "path")]
    h = [("request", "buffer")]
    s = [("query", "buffer")]
    i = [("http", "buffer"), ("pipeline", "buffer"), ("destination", "path")]
    return [
        {"draft": draft("pipeline_save", p, "value", pipe_ops), "state_lesson": lesson(p,
            [pipeline_case("a", b"abc"), pipeline_case("b", b"no", True)],
            [pipeline_case("empty", b""), pipeline_case("binary", b"\xffx", True)])},
        {"draft": draft("created_content", c, "buffer", ["process.run", "file.read"]), "state_lesson": lesson(c,
            [created_case("folder/a", b"abc"), created_case("other/b", b"different")],
            [created_case("new/empty", b""), created_case("files/binary", b"\xff\0")], "buffer")},
        {"draft": draft("http_checked", h, "buffer", http_ops), "state_lesson": lesson(h,
            [http_case("a", b"abc"), http_case("b", b"\xffmore", 201)],
            [http_case("empty", b"", 204), http_case("unicode", "자료".encode())], "buffer")},
        {"draft": draft("system_text", s, "buffer", ["system.info", "json.text"]), "state_lesson": lesson(s,
            [system_case("a", "platform", "Linux"), system_case("b", "machine", "x86_64")],
            [system_case("c", "cwd", "/other/root"), system_case("d", "platform", "OtherOS")], "buffer")},
        {"draft": draft("download_transform", i, "value", http_ops + pipe_ops), "state_lesson": lesson(i,
            [integrated_case("a", b"abc"), integrated_case("b", b"do-not-save", True)],
            [integrated_case("empty", b""), integrated_case("binary", b"\xffx", True)])}
    ]


class FixtureServer:
    """Local test peer, never invoked during acquisition."""
    def __init__(self, values):
        owner = self
        self.values = values
        self.requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.path)
                status, data = owner.values.get(self.path, (404, b"missing"))
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_POST(self):
                data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(201)
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def __enter__(self):
        self.thread.start()
        return self
    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"
    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def verify(runtime, outcomes, root):
    contracts = {o["contract"]["name"]: o["contract"] for o in outcomes}
    checks = []
    payloads = (b"", b"\xff\0\t\n", "새 자료 abc".encode(), b"ab" * 4096)
    with FixtureServer({"/" + str(i): (200, data) for i, data in enumerate(payloads)}) as peer:
        for i, data in enumerate(payloads):
            native = root / f"native-{i}"
            native.mkdir(parents=True)
            (native / "keep").write_bytes(b"keep")
            runtime.host = HostContext(native)
            result = runtime.call_contract(contracts["download_transform"]["id"], {
                "http": request_bytes({"url": peer.url + "/" + str(i)}).hex(),
                "pipeline": request_bytes(pipeline_spec()).hex(), "destination": "output"}, 16)
            assert result["outputs"] == [len(data.upper())]
            assert {p.name: p.read_bytes() for p in native.iterdir()} == {"keep": b"keep", "output": data.upper()}
            assert not any(p.is_dir() for p in native.iterdir())
            checks.append({"kind": "integration", "bytes": len(data), "passed": True, "call": result})
        for failure in ("http", "pipeline"):
            native = root / ("fail-" + failure)
            native.mkdir()
            (native / "output").write_bytes(b"preserved")
            runtime.host = HostContext(native)
            arguments = {"http": request_bytes({"url": peer.url + ("/missing" if failure == "http" else "/2")}).hex(),
                         "pipeline": request_bytes(pipeline_spec(failure == "pipeline")).hex(), "destination": "output"}
            if failure == "http":
                try:
                    runtime.call_contract(contracts["download_transform"]["id"], arguments, 16)
                except OSError:
                    pass
                else:
                    raise AssertionError("HTTP failure was accepted")
                assert all(e["operation"] != "process.pipeline" for e in runtime.host.events)
            else:
                assert runtime.call_contract(contracts["download_transform"]["id"], arguments, 16)["outputs"] == [0]
                assert all(e["operation"] != "file.write" for e in runtime.host.events)
            assert {p.name: p.read_bytes() for p in native.iterdir()} == {"output": b"preserved"}
            checks.append({"kind": failure + " failure", "passed": True})
    for i, data in enumerate(payloads):
        native = root / f"created-{i}"
        native.mkdir()
        (native / "keep").write_bytes(b"keep")
        runtime.host = HostContext(native)
        name = f"new-{i}/새 파일"
        result = runtime.call_contract(contracts["created_content"]["id"], {
            "request": request_bytes({"argv": [sys.executable, "-c", WRITE_SCRIPT, name]}).hex(),
            "stdin": data.hex(), "source": name}, 16)
        assert result["outputs"] == [{"hex": data.hex()}]
        assert {p.relative_to(native).as_posix(): p.read_bytes() for p in native.rglob("*") if p.is_file()} == {"keep": b"keep", name: data}
        assert sorted(p.relative_to(native).as_posix() for p in native.rglob("*") if p.is_dir()) == ["empty", f"new-{i}"]
        checks.append({"kind": "process filesystem", "bytes": len(data), "passed": True})
    runtime.host = HostContext(root)
    for query, value in (("platform", platform.system()), ("machine", platform.machine()), ("cwd", str(root.resolve()))):
        result = runtime.call_contract(contracts["system_text"]["id"], {"query": query.encode().hex()}, 16)
        assert result["outputs"] == [{"hex": value.encode().hex()}]
        checks.append({"kind": "system", "query": query, "passed": True})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/linux_bundle_verified"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    evidence = curriculum()
    (root / "curriculum.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    runtime = VectorRuntime.load(SOURCE, host=HostContext(root / "unused"))
    outcomes = []
    for request in evidence:
        outcome = runtime.teach_contract(**request)
        outcomes.append(outcome)
        (root / "learning.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        if outcome["status"] != "registered":
            raise RuntimeError(str(outcome))
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt", host=HostContext(root))
    checks = verify(loaded, outcomes, root)
    summary = {"learning": outcomes, "native": checks}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(checks), "history": [o["history"] for o in outcomes]}))


if __name__ == "__main__":
    main()
