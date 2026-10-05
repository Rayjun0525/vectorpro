import copy
import json
import pytest
from vectorpro.acquisition import EvidenceBank, digest
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime
from experiments.reference_acquisition import manifests, ProtocolModel

OLD = "Copy input.bin to output.bin."
NEW = "Please duplicate input.bin into output.bin while preserving the source."


@pytest.fixture(scope="module")
def acquired(tmp_path_factory):
    root = tmp_path_factory.mktemp("reuse")
    record = ReferenceProviders(manifests()).collect("copy", OLD)
    runtime = VectorRuntime(host=HostContext(root))
    bank = EvidenceBank([record])
    result = bank.acquire(runtime, OLD, record["id"], bank.proposal(OLD, record["id"], "copy_unit"))
    assert result["status"] == "registered"
    return runtime.registry.to_data(), record, result["contract"]


def fresh(acquired, tmp_path):
    from vectorpro.learning import Registry
    return VectorRuntime(Registry.from_data(acquired[0]), host=HostContext(tmp_path))


@pytest.mark.parametrize("suffix", [".pt", ".json"])
def test_new_phrase_checked_persisted_without_learning_or_id_change(acquired, tmp_path, monkeypatch, suffix):
    runtime = fresh(acquired, tmp_path)
    record = copy.deepcopy(acquired[1])
    record["intent"] = NEW
    before = runtime.registry.to_data()
    original_registry = runtime.registry
    def forbidden(*args, **kwargs):
        pytest.fail("reuse invoked learning")
    monkeypatch.setattr(VectorRuntime, "teach_contract", forbidden)
    program = tmp_path / ("program" + suffix)
    bank = EvidenceBank([record])
    outcome = bank.reuse(runtime, NEW, record["id"], program_path=program)
    assert outcome["status"] == "reused" and outcome["binding"]["checked_cases"] == 7
    assert runtime.registry is original_registry and runtime.registry.to_data() == before
    assert runtime.contract("copy_unit") == acquired[2]
    loaded = VectorRuntime.load(program, host=HostContext(tmp_path))
    assert loaded.contract("copy_unit") == acquired[2]
    assert EvidenceBank([]).permits(loaded.registry.get("copy_unit"), NEW, loaded, acquired[2]["id"])
    (tmp_path / "input.bin").write_bytes(b"new bytes")
    result = loaded.call_contract(acquired[2]["id"], {"source": "input.bin", "destination": "output.bin"}, 16)
    assert result["outputs"] == [9]


def test_different_operation_and_changed_goal_cannot_reuse(acquired, tmp_path):
    runtime = fresh(acquired, tmp_path)
    record = ReferenceProviders(manifests()).collect("move", NEW)
    record["interface"]["allowed_operations"] = ["file.read", "file.write", "file.move"]
    before = runtime.registry.to_data()
    result = EvidenceBank([record]).reuse(runtime, NEW, record["id"])
    assert result["status"] == "needs_learning_examples"
    assert not runtime._intent_bindings and runtime.registry.to_data() == before
    wrong = copy.deepcopy(acquired[1])
    wrong["intent"] = NEW
    wrong["heldout"][0]["after"]["unexpected"] = "ff"
    assert EvidenceBank([wrong]).reuse(runtime, NEW, wrong["id"])["status"] == "needs_learning_examples"
    assert not runtime._intent_bindings and not runtime.host.events


def test_multiple_matching_contracts_require_more_evidence(acquired, tmp_path):
    runtime = fresh(acquired, tmp_path)
    record = copy.deepcopy(acquired[1])
    bank = EvidenceBank([record])
    assert bank.acquire(runtime, OLD, record["id"], bank.proposal(OLD, record["id"], "duplicate_copy"))["status"] == "registered"
    record["intent"] = NEW
    result = EvidenceBank([record]).reuse(runtime, NEW, record["id"])
    assert result["status"] == "ambiguous_reuse" and not runtime._intent_bindings


def test_save_failure_preserves_binding_registry_rng_and_program(acquired, tmp_path, monkeypatch):
    runtime = fresh(acquired, tmp_path)
    program = tmp_path / "program.pt"
    runtime.save(program)
    before, registry, rng = program.read_bytes(), runtime.registry, runtime._rng.getstate()
    record = copy.deepcopy(acquired[1])
    record["intent"] = NEW
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(VectorRuntime, "save", fail)
    with pytest.raises(OSError):
        EvidenceBank([record]).reuse(runtime, NEW, record["id"], program_path=program)
    assert runtime.registry is registry and not runtime._intent_bindings
    assert runtime._rng.getstate() == rng and program.read_bytes() == before


def test_stale_binding_and_corrupt_serialized_id_are_rejected(acquired, tmp_path):
    runtime = fresh(acquired, tmp_path)
    record = copy.deepcopy(acquired[1])
    record["intent"] = NEW
    EvidenceBank([record]).reuse(runtime, NEW, record["id"])
    record["heldout"][0]["output"] += 1
    assert not EvidenceBank([record]).permits(runtime.registry.get("copy_unit"), NEW, runtime, acquired[2]["id"])
    assert not EvidenceBank([]).permits(runtime.registry.get("copy_unit"), "another goal", runtime, acquired[2]["id"])
    program = tmp_path / "program.json"
    runtime.save(program)
    data = json.loads(program.read_text())
    data["intent_bindings"][0]["contract_id"] = "vp1:" + "0" * 64
    program.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="current contract"):
        VectorRuntime.load(program)


def test_adapter_reuses_then_repeat_never_collects(acquired, tmp_path, monkeypatch):
    runtime = fresh(acquired, tmp_path)
    (tmp_path / "input.bin").write_bytes(b"another input")
    contract = acquired[2]
    class Model:
        def __init__(self):
            self.calls = 0
        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                name, args = "collect_evidence", {"provider_id": "copy"}
            else:
                name, args = "call_contract", {"contract_id": contract["id"], "arguments": {"source": "input.bin", "destination": "output.bin"}, "width": 16}
            return {"role": "assistant", "tool_calls": [{"id": str(self.calls), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    def forbidden(*args, **kwargs):
        pytest.fail("duplicate teaching")
    monkeypatch.setattr(VectorRuntime, "teach_contract", forbidden)
    program = tmp_path / "program.pt"
    result = AgentSession(runtime, Model(), program, reference_providers=ReferenceProviders(manifests())).run(NEW)
    assert result["status"] == "executed" and result["tools"][0]["result"]["status"] == "reused"
    loaded = VectorRuntime.load(program, host=HostContext(tmp_path))
    before = program.read_bytes()
    providers = ReferenceProviders(manifests())
    monkeypatch.setattr(providers, "collect", forbidden)
    repeat = AgentSession(loaded, ProtocolModel("copy", contract), program, allow_learning=False, reference_providers=providers).run(NEW)
    assert repeat["status"] == "executed" and len(repeat["tools"]) == 1 and program.read_bytes() == before
