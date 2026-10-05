import copy
import json
import pytest
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime
from experiments.verified_acquisition import draft, record, ProtocolModel, INTENT


@pytest.fixture(scope="module")
def source():
    return record()


def test_unknown_learn_hidden_accept_save_execute_and_direct_reuse(source, tmp_path, monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        pytest.fail("acquisition launched a native process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    (tmp_path / "input.bin").write_bytes(b"new\0\xff")
    (tmp_path / "keep").write_bytes(b"preserved")
    (tmp_path / "empty").mkdir()
    runtime = VectorRuntime(host=HostContext(tmp_path))
    program = tmp_path / "program.pt"
    model = ProtocolModel()
    bank = EvidenceBank([source])
    result = AgentSession(runtime, model, program, evidence_bank=bank).run(INTENT)
    assert result["status"] == "executed" and result["outputs"] == [5]
    assert [t["name"] for t in result["tools"]] == ["learn_verified_contract", "call_contract"]
    accepted = result["tools"][0]["result"]
    assert accepted["acceptance"]["passed"] == 3
    assert accepted["intent_independently_verified"] is False
    wire = json.dumps(model.messages)
    assert "heldout" not in json.dumps(model.messages[0])
    assert "source4" not in wire and "source0" not in wire
    assert (tmp_path / "output.bin").read_bytes() == b"new\0\xff"
    assert (tmp_path / "keep").read_bytes() == b"preserved" and (tmp_path / "empty").is_dir()
    loaded = VectorRuntime.load(program, host=HostContext(tmp_path))
    c = loaded.contract("observed_transfer")
    before = program.read_bytes()
    known = AgentSession(loaded, ProtocolModel(c), program, allow_learning=False, evidence_bank=EvidenceBank([])).run(INTENT)
    assert known["status"] == "executed" and len(known["tools"]) == 1
    assert before == program.read_bytes()
    other_intent = AgentSession(loaded, None, evidence_bank=EvidenceBank([]))
    other_intent._intent = "different operation on the same files"
    assert "call_contract" not in [t["function"]["name"] for t in other_intent.available_tools()]
    with pytest.raises(ValueError, match="not available"):
        other_intent.call("call_contract", {"contract_id": c["id"],
            "arguments": {"source": "input.bin", "destination": "output.bin"}, "width": 16})
    changed_source = copy.deepcopy(source)
    changed_source["heldout"][0]["output"] += 1
    stale = AgentSession(loaded, None, evidence_bank=EvidenceBank([changed_source]))
    stale._intent = INTENT
    assert "call_contract" not in [t["function"]["name"] for t in stale.available_tools()]
    (tmp_path / "input.bin").write_bytes(b"other unseen content")
    assert loaded.call_contract(c["id"], {"source": "input.bin", "destination": "output.bin"}, 16)["outputs"] == [20]


def test_hidden_failure_never_registers_or_overwrites_program(source, tmp_path):
    wrong = copy.deepcopy(source)
    wrong["heldout"][0]["after"]["unwanted"] = "ff"
    runtime = VectorRuntime(host=HostContext(tmp_path))
    original_registry = runtime.registry
    program = tmp_path / "program.pt"
    runtime.save(program)
    before, rng = program.read_bytes(), runtime._rng.getstate()
    result = EvidenceBank([wrong]).acquire(runtime, INTENT, wrong["id"], draft(), program_path=program)
    assert result["status"] == "learning_failed" and not result["registered"]
    assert "hidden acceptance" in result["reason"]
    assert runtime.registry is original_registry and not runtime.contracts()
    assert runtime._rng.getstate() == rng and program.read_bytes() == before
    assert not runtime.host.events and not runtime.host.buffers


@pytest.mark.parametrize("mutation", ["training_overlap", "validation_overlap", "missing_output", "interface", "code"])
def test_bad_evidence_is_rejected(source, mutation):
    bad = copy.deepcopy(source)
    if mutation == "training_overlap":
        bad["heldout"][0] = bad["state_lesson"]["training"][0]
    elif mutation == "validation_overlap":
        bad["heldout"][0] = bad["state_lesson"]["validation"][0]
    elif mutation == "missing_output":
        del bad["heldout"][0]["output"]
    elif mutation == "interface":
        bad["interface"]["parameters"][0]["type"] = "value"
    else:
        bad["code"] = "copy(source,destination)"
    with pytest.raises(ValueError):
        EvidenceBank([bad])


def test_model_cannot_create_evidence_change_answers_or_cross_intent(source):
    bank = EvidenceBank([source])
    runtime = VectorRuntime()
    session = AgentSession(runtime, None, evidence_bank=bank)
    session._intent = INTENT
    with pytest.raises(ValueError):
        session.call("learn_verified_contract", {"source_id": source["id"], "draft": draft(), "heldout": []})
    with pytest.raises(ValueError):
        session.call("teach_contract", {"draft": draft(), "state_lesson": source["state_lesson"]})
    assert bank.acquire(runtime, "different intent", source["id"], draft())["status"] == "needs_evidence"
    assert not runtime.contracts()
    source_view = bank.describe(INTENT)
    source_view[0]["interface"]["parameters"][0]["type"] = "value"
    assert bank.describe(INTENT)[0]["interface"]["parameters"][0]["type"] == "path"


def test_no_evidence_no_learning_or_false_success():
    class Model:
        def complete(self, messages, tools):
            assert "learn_verified_contract" not in [t["function"]["name"] for t in tools]
            return {"role": "assistant", "content": "Done"}
    result = AgentSession(VectorRuntime(), Model(), evidence_bank=EvidenceBank([])).run("unknown")
    assert result["status"] == "not_executed"


def test_persistence_failure_keeps_live_registry(source, tmp_path, monkeypatch):
    runtime = VectorRuntime(host=HostContext(tmp_path))
    original = runtime.registry
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(VectorRuntime, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        EvidenceBank([source]).acquire(runtime, INTENT, source["id"], draft(), program_path=tmp_path / "program.pt")
    assert runtime.registry is original and not runtime.contracts() and not runtime.host.events


@pytest.mark.parametrize("wrong_hidden", [False, True])
def test_numeric_hidden_acceptance_is_separate_from_search(wrong_hidden):
    import torch
    torch.manual_seed(0)
    interface = {"parameters": [{"name": n, "type": "value", "role": n} for n in ("left", "right")],
                 "output": {"type": "value", "width": "W"}, "allowed_operations": []}
    source = {"id": "nand-labels", "intent": "learn NAND", "origin": "caller labels", "interface": interface,
        "lesson": {"training": {"width": 4, "operands": [[0, 0], [1, 1], [1, 0], [0, 1]], "targets": [15, 14, 15, 15]},
                   "validation": {"width": 4, "operands": [[2, 1], [3, 1]], "targets": [15, 14]}},
        "heldout": {"width": 8, "operands": [[9, 7], [14, 11]], "targets": [254, 245]}}
    if wrong_hidden:
        source["heldout"]["targets"][0] = 0
    runtime = VectorRuntime()
    proposed = {"version": 1, "name": "accepted_nand", "description": "NAND", **interface}
    result = EvidenceBank([source]).acquire(runtime, "learn NAND", "nand-labels", proposed)
    assert result["registered"] is (not wrong_hidden)
    if wrong_hidden:
        assert not runtime.contracts()
    else:
        assert runtime.call_contract(result["contract"]["id"], {"left": 5, "right": 3}, 16)["outputs"] == [65534]
