"""Replay a scripted review-to-execution chain; not a human-review accuracy test."""
import json
import argparse
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


class AcquisitionProtocol(ProtocolModel):
    def complete(self, messages, tools):
        if len(tools) == 1 and tools[0]['function']['name'] == 'learn_verified_contract':
            self.calls += 1
            source = json.loads(messages[-1]['content'])['sources'][0]
            return {'role': 'assistant', 'tool_calls': [{'id': str(self.calls), 'type': 'function',
                'function': {'name': 'learn_verified_contract', 'arguments': json.dumps({'source_id': source['id'], 'name': 'goal_observed_' + self.provider})}}]}
        try:
            return super().complete(messages, tools)
        except KeyError as error:
            raise RuntimeError('Protocol fixture cannot continue: ' + messages[-1]['content']) from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/goal_memory_execution'))
    parser.add_argument('--program', type=Path, default=Path('results/goal_memory_validation/program.pt'))
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    intent = "Put the original bytes of source.bin in target.bin and delete source.bin afterward."
    if args.copy:
        intent = 'Duplicate source.bin into target.bin, preserving source.bin.'
    (root / "intent.json").write_text(json.dumps({"intent": intent, "expected": "source " + ('unchanged' if args.copy else 'absent') + "; target initial source bytes; keep and empty unchanged"}), encoding="utf-8")
    directory = Path("/opt/vectorpro-models/multilingual-minilm")
    encoder, identity = Encoder(directory), json.loads((directory / "download.json").read_text())
    runtime = VectorRuntime.load(args.program)
    before_contracts = runtime.contracts()
    memory = GoalMemory.from_runtime(runtime, encoder, identity)
    providers = ReferenceProviders(manifests())
    draft = propose_goal(None, providers, intent, goal_memory=memory)
    assert draft["status"] == "needs_goal_review"
    expected = [{"kind": "absent", "parameter": "source"},
                {"kind": "equals_initial", "parameter": "destination", "source": "source"},
                {"kind": "unchanged_except", "parameters": ["source", "destination"]}]
    if args.copy:
        expected[0]['kind'] = 'unchanged'
        expected[2]['parameters'] = ['destination']
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
        result = AgentSession(runtime, AcquisitionProtocol("copy" if args.copy else "move", contract), program,
            reference_providers=providers, request_goal=goal, allow_learning=not index).run(intent)
        files = {p.name: p.read_bytes().hex() for p in native.iterdir() if p.is_file()}
        expected_files = {"keep": b"keep".hex(), "target.bin": payload.hex()}
        if args.copy:
            expected_files['source.bin'] = payload.hex()
        passed = result["status"] == "executed" and files == expected_files
        passed = passed and [p.name for p in native.iterdir() if p.is_dir()] == ["empty"]
        if not index:
            assert all(c in runtime.contracts() for c in before_contracts)
            after_contracts = runtime.contracts()
        passed = passed and runtime.contracts() == after_contracts
        passed = passed and runtime._goal_memory is not None and (not index or len(result["tools"]) == 1)
        outcomes.append({"index": index, "passed": passed, "result": result, "files": files})
    (root / "summary.json").write_text(json.dumps({"protocol": "scripted caller review, observed reuse, native execution and saved repeat", "cases": outcomes}, indent=2), encoding="utf-8")
    assert all(o["passed"] for o in outcomes)
    print(json.dumps({"passed": len(outcomes), "total": len(outcomes)}))


if __name__ == "__main__":
    main()
