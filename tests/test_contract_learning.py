import json
from pathlib import Path

import pytest
import torch

from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def draft():
    return {"version": 1, "name": "new_transfer", "description": "Copy source bytes to destination",
            "parameters": [{"name": "source", "type": "path", "role": "source file"},
                           {"name": "destination", "type": "path", "role": "destination file"}],
            "output": {"type": "value", "width": "W"},
            "allowed_operations": ["file.read", "file.write"]}


def evidence():
    path = Path(__file__).resolve().parents[1] / "experiments/requests/learn_transfer.json"
    return json.loads(path.read_text())["state_lesson"]


def test_missing_evidence_does_not_register():
    r = VectorRuntime()
    before = r.registry.to_data()
    assert r.teach_contract(draft())["status"] == "needs_learning_examples"
    assert r.registry.to_data() == before


def test_numeric_contract_generalizes_to_unseen_width_and_named_inputs():
    torch.manual_seed(0)
    r = VectorRuntime()
    d = draft()
    d.update(name="new_nand", description="Bitwise NAND", allowed_operations=[])
    d["parameters"] = [{"name": n, "type": "value", "role": n} for n in ("left", "right")]
    examples = {"training": {"width": 4, "operands": [[0, 0], [1, 1], [1, 0], [0, 1]],
                             "targets": [15, 14, 15, 15]},
                "validation": {"width": 4, "operands": [[2, 1], [3, 1]], "targets": [15, 14]}}
    result = r.teach_contract(d, lesson=examples)
    assert result["status"] == "registered"
    for a in range(100):
        b = (a * 73 + 11) & 65535
        assert r.call_contract(result["contract"]["id"], {"left": a, "right": b}, 16)["outputs"] == [~(a & b) & 65535]


@pytest.mark.parametrize("suffix", [".json", ".pt"])
def test_learn_named_contract_without_native_effects_and_reload(tmp_path, suffix):
    (tmp_path / "original").write_bytes(b"preserved")
    r = VectorRuntime(host=HostContext(tmp_path))
    result = r.teach_contract(draft(), state_lesson=evidence())
    assert result["status"] == "registered"
    assert result["intent_independently_verified"] is False
    assert r.host.events == [] and r.host.buffers == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["original"]
    c = result["contract"]
    assert [p["name"] for p in c["parameters"]] == ["source", "destination"]
    program = tmp_path / ("program" + suffix)
    r.save(program)
    other = tmp_path / "other"
    other.mkdir()
    (other / "in").write_bytes(bytes(range(100)))
    loaded = VectorRuntime.load(program, host=HostContext(other))
    assert loaded.contract("new_transfer") == c
    answer = loaded.call_contract(c["id"], {"source": "in", "destination": "out"}, 16)
    assert answer["outputs"] == [100]
    assert (other / "out").read_bytes() == bytes(range(100))
    with pytest.raises(ValueError, match="already exists"):
        loaded.teach_contract(draft(), state_lesson=evidence())


@pytest.mark.parametrize("rejection", ["operations", "width", "deadline"])
def test_accepted_candidate_rejected_by_draft_keeps_live_state(tmp_path, monkeypatch, rejection):
    r = VectorRuntime(host=HostContext(tmp_path))
    original_registry = r.registry
    before = r.registry.to_data()
    rng = r._rng.getstate()
    d = draft()
    if rejection == "operations":
        d["allowed_operations"] = ["file.read"]
    elif rejection == "width":
        d["output"]["width"] = "W+1"
    else:
        ticks = iter([0, 61])
        monkeypatch.setattr("vectorpro.runtime.monotonic", lambda: next(ticks))
    result = r.teach_contract(d, state_lesson=evidence())
    assert result["status"] == "learning_failed"
    assert result["registered"] is False
    assert r.registry is original_registry and r.registry.to_data() == before
    assert r._rng.getstate() == rng and not r.host.events and not r.host.buffers
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["code", "duplicate", "type", "certificate"])
def test_invalid_draft_or_evidence_cannot_register(mutation):
    r = VectorRuntime()
    d = draft()
    kwargs = {"state_lesson": evidence()}
    if mutation == "code":
        d["code"] = "copy(a,b)"
    elif mutation == "duplicate":
        d["parameters"][1]["name"] = "source"
    elif mutation == "type":
        d["parameters"][0]["type"] = "value"
    else:
        kwargs["evidence_source"] = "independently_verified"
    with pytest.raises(ValueError):
        r.teach_contract(d, **kwargs)
    assert not r.contracts()


def test_agent_labels_model_examples_and_respects_disabled_learning(tmp_path):
    r = VectorRuntime()
    session = AgentSession(r, None, program_path=tmp_path / "program.pt")
    result = session.call("teach_contract", {"draft": draft(), "state_lesson": evidence()})
    assert result["status"] == "registered"
    assert result["evidence_source"] == "llm_proposed_examples"
    assert session.program_path.exists()
    disabled = AgentSession(VectorRuntime(), None, allow_learning=False)
    assert "teach_contract" not in [t["function"]["name"] for t in disabled.available_tools()]
    with pytest.raises(ValueError, match="disabled"):
        disabled.call("teach_contract", {"draft": draft()})


def test_agent_registration_is_authoritative_and_does_not_execute():
    class Model:
        def complete(self, messages, tools):
            return {"role": "assistant", "tool_calls": [{"id": "one", "type": "function",
                "function": {"name": "teach_contract", "arguments": json.dumps({"draft": draft(), "state_lesson": evidence()})}}]}
    result = AgentSession(VectorRuntime(), Model(), max_calls=1).run("Learn a named copy function")
    assert result["status"] == "registered"
    assert len(result["tools"]) == 1


def test_cli_saves_only_registered_contract(tmp_path, capsys):
    from vectorpro.__main__ import main
    program, request = tmp_path / "program.pt", tmp_path / "request.json"
    args = ["--program", str(program), "--request", str(request)]
    request.write_text(json.dumps({"action": "teach_contract", "draft": draft()}))
    assert main(args) == 2 and not program.exists()
    capsys.readouterr()
    request.write_text(json.dumps({"action": "teach_contract", "draft": draft(), "state_lesson": evidence()}))
    assert main(args) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert VectorRuntime.load(program).contract("new_transfer") == accepted["contract"]
    before = program.read_bytes()
    bad = draft()
    bad["name"] = "rejected"
    bad["allowed_operations"] = []
    request.write_text(json.dumps({"action": "teach_contract", "draft": bad, "state_lesson": evidence()}))
    assert main(args) == 2 and program.read_bytes() == before
