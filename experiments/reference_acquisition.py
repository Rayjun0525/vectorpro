"""Automatically select installed references, collect examples, learn and reuse."""
import argparse
import json
from pathlib import Path
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime


def manifests():
    parameters = [{"name": "source", "type": "path", "role": "source file"},
                  {"name": "destination", "type": "path", "role": "destination file"}]
    common = {"seed_files": {"$source": "$payload", "$destination": "6f6c64", "keep": "aa"}, "directories": ["empty"]}
    return [{**common, "id": "copy", "description": "Copy file contents to another path, keeping source and replacing destination",
        "interface": {"parameters": parameters, "output": {"type": "value", "width": "W"}, "allowed_operations": ["file.read", "file.write"]},
        "argv": ["/bin/cp", "--", "$source", "$destination"], "return": {"kind": "file_length", "parameter": "destination"}, "max_steps": 2},
        {**common, "id": "move", "description": "Move or rename a file: remove source and replace destination",
        "interface": {"parameters": parameters, "output": {"type": "value", "width": "W"}, "allowed_operations": ["file.move"]},
        "argv": ["/bin/mv", "--", "$source", "$destination"], "return": {"kind": "success"}, "max_steps": 1}]


class ProtocolModel:
    def __init__(self, provider, known=None):
        self.provider, self.known, self.calls = provider, known, 0
    def complete(self, messages, tools):
        self.calls += 1
        if self.known is not None:
            name, args = "call_contract", {"contract_id": self.known["id"], "arguments": {"source": "input.bin", "destination": "output.bin"}, "width": 16}
        elif self.calls == 1:
            name, args = "collect_evidence", {"provider_id": self.provider}
        elif self.calls == 2:
            source = json.loads(messages[-1]["content"])["sources"][0]
            name, args = "learn_verified_contract", {"source_id": source["id"], "name": "observed_" + self.provider}
        else:
            contract = json.loads(messages[-1]["content"])["contract"]
            name, args = "call_contract", {"contract_id": contract["id"], "arguments": {"source": "input.bin", "destination": "output.bin"}, "width": 16}
        return {"role": "assistant", "tool_calls": [{"id": str(self.calls), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/reference_acquisition"))
    parser.add_argument("--live-gemma", action="store_true")
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    definitions = manifests()
    (root / "providers.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    cases = [("copy", "Copy input.bin to output.bin, keeping input.bin."),
             ("move", "Move input.bin to output.bin, removing input.bin."),
             ("unsupported", "Encrypt input.bin into output.bin with a secret key.")]
    (root / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    live = None
    if args.live_gemma:
        from experiments.verified_acquisition_gemma import GemmaModel
        live = GemmaModel("/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    runtime = VectorRuntime()
    checks = []
    for kind, intent in cases:
        native = root / kind
        native.mkdir()
        payload = b"unseen\0\xff" + "실제 입력".encode()
        (native / "input.bin").write_bytes(payload)
        (native / "keep").write_bytes(b"keep")
        (native / "empty").mkdir()
        runtime.host = HostContext(native)
        if not live and kind == "unsupported":
            continue
        session = AgentSession(runtime, live or ProtocolModel(kind), root / "program.pt", reference_providers=ReferenceProviders(definitions))
        result = session.run(intent)
        if session.collected_evidence:
            (root / (kind + "-evidence.json")).write_text(json.dumps(session.collected_evidence, indent=2), encoding="utf-8")
        files = {p.name: p.read_bytes().hex() for p in native.iterdir() if p.is_file()}
        expected = {"keep": b"keep".hex()}
        if kind != "move":
            expected["input.bin"] = payload.hex()
        if kind != "unsupported":
            expected["output.bin"] = payload.hex()
        passed = files == expected and [p.name for p in native.iterdir() if p.is_dir()] == ["empty"]
        passed = passed and (result["status"] in ("needs_input", "not_executed") if kind == "unsupported" else result["status"] == "executed")
        checks.append({"kind": kind, "passed": passed, "result": result, "files": files})
        (root / "summary.json").write_text(json.dumps({"protocol": "live first-use" if live else "scripted protocol", "cases": checks}, indent=2), encoding="utf-8")
        if live:
            (root / "raw_calls.json").write_text(json.dumps(live.raw, indent=2), encoding="utf-8")
        print(json.dumps({"kind": kind, "passed": passed, "status": result["status"]}), flush=True)
    loaded = VectorRuntime.load(root / "program.pt") if (root / "program.pt").exists() else None
    if not live:
        assert all(c["passed"] for c in checks)
        assert loaded is not None


if __name__ == "__main__":
    main()
