import json
import pytest
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.goal_evidence import validate_goal
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime
from experiments.reference_acquisition import manifests

INTENT = "Rename source.bin to target.bin, removing source.bin."


def goal(intent=INTENT, remove=True):
    return {"intent": intent, "rules": [
        {"kind": "equals_initial", "parameter": "destination", "source": "source"},
        {"kind": "absent" if remove else "unchanged", "parameter": "source"},
        {"kind": "unchanged_except", "parameters": ["source", "destination"]}]}


def test_copy_rejected_move_accepted_all_cases():
    providers = ReferenceProviders(manifests())
    with pytest.raises(ValueError, match="contradict"):
        EvidenceBank([providers.collect("copy", INTENT)], goal=goal())
    record = providers.collect("move", INTENT)
    assert EvidenceBank([record], goal=goal()).describe(INTENT)
    record["heldout"][0]["after"]["surprise"] = "00"
    with pytest.raises(ValueError, match="contradict"):
        EvidenceBank([record], goal=goal())


@pytest.mark.parametrize("change", ["directory", "return_source", "wrong_bytes"])
def test_hidden_effect_discrepancies_rejected(change):
    record = ReferenceProviders(manifests()).collect("move", INTENT)
    case = record["heldout"][0]
    if change == "directory":
        case["after_directories"].append("unexpected")
    elif change == "return_source":
        case["after"][case["inputs"][0]] = case["before"][case["inputs"][0]]
    else:
        case["after"][case["inputs"][1]] = "00"
    with pytest.raises(ValueError, match="contradict"):
        EvidenceBank([record], goal=goal())


@pytest.mark.parametrize("suffix", [".pt", ".json"])
def test_goal_hash_persists_and_legacy_binding_cannot_bypass(tmp_path, suffix):
    runtime = VectorRuntime.load("results/reference_acquisition_gemma/program.pt")
    record = ReferenceProviders(manifests()).collect("move", INTENT)
    old = EvidenceBank([record]).reuse(runtime, INTENT, record["id"])
    cap = runtime.registry.get(old["contract"]["name"])
    bank = EvidenceBank([], goal=goal())
    assert not bank.permits(cap, INTENT, runtime, old["contract"]["id"])
    program = tmp_path / ("program" + suffix)
    result = EvidenceBank([record], goal=goal()).reuse(runtime, INTENT, record["id"], program_path=program)
    assert "goal_sha256" in result["binding"]
    loaded = VectorRuntime.load(program)
    assert bank.permits(loaded.registry.get(cap.name), INTENT, loaded, result["contract"]["id"])
    assert not EvidenceBank([], goal=goal(remove=False)).permits(cap, INTENT, runtime, result["contract"]["id"])


def test_wrong_model_choice_cannot_save_or_execute(tmp_path):
    runtime = VectorRuntime.load("results/reference_acquisition_gemma/program.pt", host=HostContext(tmp_path))
    before = runtime.registry.to_data()
    (tmp_path / "source.bin").write_bytes(b"actual")
    program = tmp_path / "program.pt"
    runtime.save(program)
    original = program.read_bytes()
    class WrongModel:
        def complete(self, messages, tools):
            name = "collect_evidence" if any(t["function"]["name"] == "collect_evidence" for t in tools) else "ask_user"
            args = {"provider_id": "copy"} if name == "collect_evidence" else {"question": "Need matching evidence"}
            return {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    session = AgentSession(runtime, WrongModel(), program, reference_providers=ReferenceProviders(manifests()), request_goal=goal())
    result = session.run(INTENT)
    assert result["status"] == "needs_input"
    assert result["tools"][0]["result"]["status"] == "error"
    assert (tmp_path / "source.bin").read_bytes() == b"actual" and not (tmp_path / "target.bin").exists()
    assert runtime.registry.to_data() == before and program.read_bytes() == original
    assert not runtime._intent_bindings and session.collected_evidence is None


def test_invalid_or_wrong_intent_goal_rejected():
    with pytest.raises(ValueError):
        validate_goal({"intent": INTENT, "rules": [{"kind": "python", "parameter": "source"}]})
    with pytest.raises(ValueError, match="matching"):
        EvidenceBank([ReferenceProviders(manifests()).collect("move", "different")], goal=goal())
    invalid = goal()
    invalid["rules"][0]["source"] = "missing"
    with pytest.raises(ValueError, match="unknown"):
        EvidenceBank([ReferenceProviders(manifests()).collect("move", INTENT)], goal=invalid)


def test_new_learning_acceptance_requires_same_goal(tmp_path):
    intent = "Copy source.bin to target.bin and preserve source.bin."
    record = ReferenceProviders(manifests()).collect("copy", intent)
    bank = EvidenceBank([record], goal=goal(intent, remove=False))
    runtime = VectorRuntime()
    runtime.provide_host_operations()
    result = bank.acquire(runtime, intent, record["id"], bank.proposal(intent, record["id"], "goal_copy"), program_path=tmp_path / "program.pt")
    assert result["status"] == "registered"
    loaded = VectorRuntime.load(tmp_path / "program.pt")
    cap = loaded.registry.get("goal_copy")
    assert bank.permits(cap, intent, loaded, result["contract"]["id"])
    assert not EvidenceBank([], goal=goal(intent, remove=True)).permits(cap, intent, loaded, result["contract"]["id"])
