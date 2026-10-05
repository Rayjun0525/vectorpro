"""Observe an external reference, acquire privately, accept, save and reuse."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime

INTENT = "파일 input.bin을 output.bin으로 복사해줘."


def draft(name="observed_transfer"):
    return {"version": 1, "name": name, "description": "Copy source bytes to destination",
            "parameters": [{"name": "source", "type": "path", "role": "source file"},
                           {"name": "destination", "type": "path", "role": "destination file"}],
            "output": {"type": "value", "width": "W"}, "allowed_operations": ["file.read", "file.write"]}


def observe(source, destination, data):
    # This executable is an independent reference, never installed in the kernel.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / source).write_bytes(data)
        (root / destination).write_bytes(b"old")
        (root / "keep").write_bytes(b"preserved")
        (root / "empty").mkdir()
        snapshot = lambda: {p.name: p.read_bytes().hex() for p in root.iterdir() if p.is_file()}
        before = snapshot()
        subprocess.run(["/bin/cp", "--", source, destination], cwd=root, check=True, timeout=5)
        return {"inputs": [source, destination], "before": before, "after": snapshot(),
                "before_directories": ["empty"], "after_directories": ["empty"],
                "output": len((root / destination).read_bytes())}


def record():
    interface = {k: draft()[k] for k in ("parameters", "output", "allowed_operations")}
    samples = [observe("source" + str(i), "destination" + str(i), data)
               for i, data in enumerate((b"abc", b"\0\xff", b"", b"validation",
                                          bytes(range(256)), "독립 검증".encode(), b"ab" * 4096))]
    return {"id": "copy-observation-v1", "intent": INTENT,
            "origin": "caller-selected /bin/cp observations, separate temporary roots",
            "interface": interface, "state_lesson": {"input_types": ["path", "path"],
                "training": samples[:2], "validation": samples[2:4], "max_steps": 2},
            "heldout": samples[4:]}


class ProtocolModel:
    """Scripted protocol test, not a live model or intent interpretation score."""
    def __init__(self, known=None):
        self.known = known
        self.calls = 0
        self.messages = []
    def complete(self, messages, tools):
        self.messages.append(json.loads(json.dumps(messages)))
        self.calls += 1
        if self.known is None and self.calls == 1:
            name, args = "learn_verified_contract", {"source_id": "copy-observation-v1", "name": draft()["name"]}
        else:
            contract = self.known or json.loads(messages[-1]["content"])["contract"]
            name, args = "call_contract", {"contract_id": contract["id"],
                "arguments": {"source": "input.bin", "destination": "output.bin"}, "width": 16}
        return {"role": "assistant", "tool_calls": [{"id": str(self.calls), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/verified_acquisition"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    source = record()
    (root / "evidence.json").write_text(json.dumps([source], indent=2), encoding="utf-8")
    native = root / "native"
    native.mkdir()
    payload = b"new payload\0\xff" + "새 입력".encode()
    (native / "input.bin").write_bytes(payload)
    (native / "keep").write_bytes(b"preserved")
    (native / "empty").mkdir()
    runtime = VectorRuntime(host=HostContext(native))
    program = root / "program.pt"
    model = ProtocolModel()
    result = AgentSession(runtime, model, program, evidence_bank=EvidenceBank([source])).run(INTENT)
    assert result["status"] == "executed" and result["outputs"] == [len(payload)]
    assert {p.name: p.read_bytes() for p in native.iterdir() if p.is_file()} == {
        "input.bin": payload, "output.bin": payload, "keep": b"preserved"}
    assert [p.name for p in native.iterdir() if p.is_dir()] == ["empty"]
    loaded = VectorRuntime.load(program, host=HostContext(native))
    contract = loaded.contract("observed_transfer")
    (native / "input.bin").write_bytes(b"second unseen input")
    direct = loaded.call_contract(contract["id"], {"source": "input.bin", "destination": "output.bin"}, 16)
    assert (native / "output.bin").read_bytes() == b"second unseen input"
    before = program.read_bytes()
    known = AgentSession(loaded, ProtocolModel(contract), program, evidence_bank=EvidenceBank([]), allow_learning=False).run(INTENT)
    assert known["status"] == "executed" and len(known["tools"]) == 1 and program.read_bytes() == before
    summary = {"protocol": "scripted adapter, external reference observations; not live model quality",
               "acquisition": result, "direct_reuse": direct, "known_request": known}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "model_messages.json").write_text(json.dumps(model.messages, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "hidden_cases": 3, "direct_reuse": direct["status"], "known_reuse": known["status"]}))


if __name__ == "__main__":
    main()
