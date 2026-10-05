"""Replay a scripted review-to-execution chain; not a human-review accuracy test."""
import json
from pathlib import Path
from vectorpro.goal_memory import GoalMemory
from vectorpro.goal_draft import accept_goal, propose_goal
from vectorpro.semantic_catalog import Encoder
from vectorpro.runtime import VectorRuntime
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.agent import AgentSession
from experiments.reference_acquisition import manifests
from experiments.verified_reuse import ProtocolModel


def main():
    root = Path("results/goal_memory_execution")
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    intent = "Put the original bytes of source.bin in target.bin and delete source.bin afterward."
    (root / "intent.json").write_text(json.dumps({"intent": intent, "expected": "source absent; target initial source bytes; keep and empty unchanged"}), encoding="utf-8")
    directory = Path("/opt/vectorpro-models/multilingual-minilm")
    encoder, identity = Encoder(directory), json.loads((directory / "download.json").read_text())
    runtime = VectorRuntime.load("results/goal_memory_validation/program.pt")
    before_contracts = runtime.contracts()
    memory = GoalMemory.from_runtime(runtime, encoder, identity)
    providers = ReferenceProviders(manifests())
    draft = propose_goal(None, providers, intent, goal_memory=memory)
    assert draft["status"] == "needs_goal_review"
    expected = [{"kind": "absent", "parameter": "source"},
                {"kind": "equals_initial", "parameter": "destination", "source": "source"},
                {"kind": "unchanged_except", "parameters": ["source", "destination"]}]
    assert draft["goal"]["rules"] == expected  # scripted caller review, not a model voting yes
    goal = accept_goal(draft, draft["proposal_sha256"])
    (root / "draft.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
    program, outcomes = root / "program.pt", []
    for index, payload in enumerate((b"new\0\xff", b"second unseen bytes")):
        native = root / f"case{index}"
        native.mkdir()
        (native / "source.bin").write_bytes(payload)
        (native / "keep").write_bytes(b"keep")
        (native / "empty").mkdir()
        if index:
            runtime = VectorRuntime.load(program)
        runtime.host = HostContext(native)
        contract = None if not index else next(c for c in runtime.contracts() if c["id"] == outcomes[0]["result"]["contract_id"])
        result = AgentSession(runtime, ProtocolModel("move", contract), program,
            reference_providers=providers, request_goal=goal, allow_learning=not index).run(intent)
        files = {p.name: p.read_bytes().hex() for p in native.iterdir() if p.is_file()}
        passed = result["status"] == "executed" and files == {"keep": b"keep".hex(), "target.bin": payload.hex()}
        passed = passed and [p.name for p in native.iterdir() if p.is_dir()] == ["empty"] and runtime.contracts() == before_contracts
        passed = passed and runtime._goal_memory is not None and (not index or len(result["tools"]) == 1)
        outcomes.append({"index": index, "passed": passed, "result": result, "files": files})
    (root / "summary.json").write_text(json.dumps({"protocol": "scripted caller review, observed reuse, native execution and saved repeat", "cases": outcomes}, indent=2), encoding="utf-8")
    assert all(o["passed"] for o in outcomes)
    print(json.dumps({"passed": len(outcomes), "total": len(outcomes)}))


if __name__ == "__main__":
    main()
