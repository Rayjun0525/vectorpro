import json
import pytest
from vectorpro.goal_draft import propose_goal, accept_goal
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime
from vectorpro.host import HostContext
from experiments.reference_acquisition import manifests
from experiments.verified_reuse import ProtocolModel

INTENT = "Rename source.bin to target.bin and remove source.bin."
RULES = [{"kind": "equals_initial", "parameter": "destination", "source": "source"},
         {"kind": "absent", "parameter": "source"},
         {"kind": "unchanged_except", "parameters": ["source", "destination"]}]


class Model:
    def __init__(self, name="propose_goal", args=None):
        self.name, self.args = name, args if args is not None else {"rules": RULES}
    def complete(self, messages, tools):
        assert {t["function"]["name"] for t in tools} == {"propose_goal", "ask_user"}
        assert messages[-1]["content"] == INTENT
        return {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": self.name, "arguments": json.dumps(self.args)}}]}


def test_draft_is_detached_and_never_collects(monkeypatch):
    providers = ReferenceProviders(manifests())
    monkeypatch.setattr(providers, "collect", lambda *a: pytest.fail("drafting collected evidence"))
    draft = propose_goal(Model(), providers, INTENT)
    assert draft["status"] == "needs_goal_review"
    assert any("없어야" in line for line in draft["review"])
    with pytest.raises(ValueError):
        EvidenceBank([], goal=draft)
    accepted = accept_goal(draft, draft["proposal_sha256"])
    accepted["rules"].clear()
    assert draft["goal"]["rules"] == RULES


@pytest.mark.parametrize("change", ["rules", "intent", "review", "digest", "approval"])
def test_changed_proposal_or_approval_rejected(change):
    draft = propose_goal(Model(), ReferenceProviders(manifests()), INTENT)
    approval = draft["proposal_sha256"]
    if change == "rules":
        draft["goal"]["rules"][1]["kind"] = "unchanged"
    elif change == "intent":
        draft["goal"]["intent"] = "Copy instead"
    elif change == "review":
        draft["review"] = ["All good"]
    elif change == "digest":
        draft["proposal_sha256"] = "0" * 64
    else:
        approval = "0" * 64
    with pytest.raises(ValueError):
        accept_goal(draft, approval)


@pytest.mark.parametrize("name,args", [("accept_goal", {"rules": RULES}),
    ("propose_goal", {"rules": [{"kind": "absent", "parameter": "unknown"}, RULES[-1]]}),
    ("propose_goal", {"rules": RULES[:2]}),
    ("propose_goal", {"rules": RULES, "intent": "substituted"})])
def test_invalid_or_self_approval_tool_fails(name, args):
    assert propose_goal(Model(name, args), ReferenceProviders(manifests()), INTENT)["status"] == "goal_draft_failed"


def test_missing_semantics_returns_question():
    result = propose_goal(Model("ask_user", {"question": "Specify intended effects"}), ReferenceProviders(manifests()), INTENT)
    assert result["status"] == "needs_input" and "goal" not in result


def test_accepted_draft_executes_and_reloads_without_collecting(tmp_path, monkeypatch):
    providers = ReferenceProviders(manifests())
    draft = propose_goal(Model(), providers, INTENT)
    goal = accept_goal(draft, draft["proposal_sha256"])
    runtime = VectorRuntime.load("results/reference_acquisition_gemma/program.pt", host=HostContext(tmp_path))
    (tmp_path / "source.bin").write_bytes(b"unseen")
    program = tmp_path / "program.pt"
    result = AgentSession(runtime, ProtocolModel("move"), program, reference_providers=providers, request_goal=goal).run(INTENT)
    assert result["status"] == "executed" and not (tmp_path / "source.bin").exists()
    assert (tmp_path / "target.bin").read_bytes() == b"unseen"
    loaded = VectorRuntime.load(program, host=HostContext(tmp_path))
    (tmp_path / "source.bin").write_bytes(b"new bytes")
    monkeypatch.setattr(providers, "collect", lambda *a: pytest.fail("repeat collected"))
    contract = next(c for c in loaded.contracts() if c["id"] == result["contract_id"])
    repeat = AgentSession(loaded, ProtocolModel("move", contract), program, reference_providers=providers, request_goal=goal, allow_learning=False).run(INTENT)
    assert repeat["status"] == "executed" and len(repeat["tools"]) == 1
    assert (tmp_path / "target.bin").read_bytes() == b"new bytes"
