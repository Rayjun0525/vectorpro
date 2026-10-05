"""Evaluate new request wording against existing tensor contracts without teaching."""
import argparse
import json
from pathlib import Path
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime
from experiments.reference_acquisition import manifests

SOURCE = "results/reference_acquisition_gemma/program.pt"


class ProtocolModel:
    def __init__(self, provider, contract=None):
        self.provider, self.contract, self.calls = provider, contract, 0
    def complete(self, messages, tools):
        self.calls += 1
        if self.contract is None and self.calls == 1:
            name, args = "collect_evidence", {"provider_id": self.provider}
        else:
            contract = self.contract or json.loads(messages[-1]["content"])["contract"]
            name, args = "call_contract", {"contract_id": contract["id"],
                "arguments": {"source": "source.bin", "destination": "target.bin"}, "width": 16}
        return {"role": "assistant", "tool_calls": [{"id": str(self.calls), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/verified_reuse"))
    parser.add_argument("--live-gemma", action="store_true")
    parser.add_argument("--caller-goals", action="store_true", help="supply independently fixed state predicates; measure clean rejection separately")
    parser.add_argument("--run-kind", choices=("first-use", "replay"), default="first-use")
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    definitions = manifests()
    cases = [("copy", "Please duplicate source.bin into target.bin and keep source.bin."),
             ("move", "Rename source.bin to target.bin, so source.bin no longer exists."),
             ("repeat", "Please duplicate source.bin into target.bin and keep source.bin.")]
    if args.live_gemma and not args.caller_goals:
        cases.append(("unsupported", "Encrypt source.bin into target.bin using a secret key."))
    (root / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (root / "providers.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    goals = {kind: {"intent": intent, "rules": [
        {"kind": "equals_initial", "parameter": "destination", "source": "source"},
        {"kind": "absent" if kind == "move" else "unchanged", "parameter": "source"},
        {"kind": "unchanged_except", "parameters": ["source", "destination"]}]}
        for kind, intent in cases} if args.caller_goals else {}
    if goals:
        (root / "caller_goals.json").write_text(json.dumps(goals, indent=2), encoding="utf-8")
    runtime = VectorRuntime.load(SOURCE)
    initial_registry, initial_contracts = runtime.registry.to_data(), runtime.contracts()
    (root / "initial_contracts.json").write_text(json.dumps(initial_contracts, indent=2), encoding="utf-8")
    model = None
    if args.live_gemma:
        from experiments.verified_acquisition_gemma import GemmaModel
        model = GemmaModel("/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    checks, copy_contract = [], None
    program = root / "program.pt"
    for kind, intent in cases:
        if program.exists():
            runtime = VectorRuntime.load(program)
        native = root / kind
        native.mkdir()
        data = b"new\0\xff" + "미사용 입력".encode()
        (native / "source.bin").write_bytes(data)
        (native / "keep").write_bytes(b"keep")
        (native / "empty").mkdir()
        runtime.host = HostContext(native)
        selected = model or ProtocolModel(kind, copy_contract if kind == "repeat" else None)
        session = AgentSession(runtime, selected, program, reference_providers=ReferenceProviders(definitions),
                               allow_learning=kind != "repeat", request_goal=goals.get(kind))
        result = session.run(intent)
        if kind == "copy" and result["status"] == "executed":
            copy_contract = next(c for c in runtime.contracts() if c["id"] == result["contract_id"])
        if session.collected_evidence:
            (root / (kind + "-evidence.json")).write_text(json.dumps(session.collected_evidence, indent=2), encoding="utf-8")
        files = {p.name: p.read_bytes().hex() for p in native.iterdir() if p.is_file()}
        directories = [p.name for p in native.iterdir() if p.is_dir()]
        expected = {"keep": b"keep".hex()}
        if kind != "move":
            expected["source.bin"] = data.hex()
        if kind != "unsupported":
            expected["target.bin"] = data.hex()
        passed = files == expected and directories == ["empty"]
        if kind == "unsupported":
            passed = passed and result["status"] in ("needs_input", "not_executed")
            passed = passed and all(e["name"] != "collect_evidence" for e in result["tools"])
        else:
            passed = passed and result["status"] == "executed"
            if kind == "repeat":
                passed = passed and len(result["tools"]) == 1
            else:
                passed = passed and any(e["result"].get("status") == "reused" for e in result["tools"])
        passed = passed and runtime.registry.to_data() == initial_registry and runtime.contracts() == initial_contracts
        rejected_cleanly = (result["status"] in ("needs_input", "not_executed")
            and files == {"keep": b"keep".hex(), "source.bin": data.hex()} and directories == ["empty"]
            and all(e["name"] != "call_contract" for e in result["tools"])
            and runtime.registry.to_data() == initial_registry)
        checks.append({"kind": kind, "passed": passed, "rejected_cleanly": rejected_cleanly,
            "result": result, "files": files, "directories": directories})
        (root / "summary.json").write_text(json.dumps({"protocol": "live wording evaluation" if model else "scripted protocol", "run_kind": args.run_kind,
            "cases": checks, "unchanged_contracts": runtime.contracts() == initial_contracts}, indent=2), encoding="utf-8")
        if model:
            (root / "raw_calls.json").write_text(json.dumps(model.raw, indent=2), encoding="utf-8")
        print(json.dumps({"kind": kind, "passed": passed, "status": result["status"]}), flush=True)
    if not model:
        assert all(c["passed"] for c in checks)


if __name__ == "__main__":
    main()
