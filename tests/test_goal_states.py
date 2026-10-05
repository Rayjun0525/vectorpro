import json
import pytest
from vectorpro.goal_draft import propose_goal, accept_goal
from vectorpro.goal_evidence import validate_goal, check_goal
from vectorpro.reference_evidence import ReferenceProviders
from experiments.reference_acquisition import manifests


class Model:
    def __init__(self, states):
        self.states = states
    def complete(self, messages, tools):
        assert messages[-1]["content"] == "Actual request"
        assert "accept_goal" not in {t["function"]["name"] for t in tools}
        if tools[0]["function"]["name"] == "assess_goal":
            return {"role": "assistant", "tool_calls": [{"id": "triage", "type": "function",
                "function": {"name": "assess_goal", "arguments": json.dumps({"decision": "supported"})}}]}
        assert tools[0]["function"]["parameters"]["properties"]["states"]["required"] == ["source", "destination"]
        return {"role": "assistant", "tool_calls": [{"id": "actual", "type": "function",
            "function": {"name": "propose_goal", "arguments": json.dumps({"states": self.states})}}]}


@pytest.mark.parametrize("states,changes", [
    ({"source": "unchanged", "destination": "initial:source"}, ["destination"]),
    ({"source": "absent", "destination": "initial:source"}, ["source", "destination"]),
    ({"source": "initial:destination", "destination": "unchanged"}, ["source"]),
    ({"source": "unchanged", "destination": "unchanged"}, []),
    ({"source": "unchanged", "destination": "absent"}, ["destination"])])
def test_states_compile_generic_conditions_and_frame(states, changes):
    draft = propose_goal(Model(states), ReferenceProviders(manifests()), "Actual request", encoding="states")
    assert draft["status"] == "needs_goal_review"
    goal = accept_goal(draft, draft["proposal_sha256"])
    assert goal["rules"][-1] == {"kind": "unchanged_except", "parameters": changes}
    for rule, (name, value) in zip(goal["rules"], states.items()):
        assert rule["parameter"] == name
        assert rule["kind"] == ("equals_initial" if value.startswith("initial:") else value)


@pytest.mark.parametrize("states", [{"source": "unchanged"},
    {"source": "unknown", "destination": "absent"},
    {"source": "initial:missing", "destination": "absent"},
    {"source": "unchanged", "destination": "absent", "extra": "present"}])
def test_invalid_complete_states_rejected(states):
    assert propose_goal(Model(states), ReferenceProviders(manifests()), "Actual request", encoding="states")["status"] == "goal_draft_failed"


def test_empty_frame_means_preserve_entire_snapshot():
    goal = validate_goal({"intent": "Keep all", "rules": [{"kind": "unchanged_except", "parameters": []}]})
    case = {"inputs": ["input"], "before": {"input": "ff"}, "after": {"input": "ff"},
            "before_directories": ["empty"], "after_directories": ["empty"]}
    record = {"intent": "Keep all", "interface": {"parameters": [{"name": "source"}]},
              "state_lesson": {"training": [case], "validation": []}, "heldout": []}
    assert check_goal(goal, record) == 1
    case["after_directories"] = []
    with pytest.raises(ValueError, match="contradict"):
        check_goal(goal, record)


@pytest.mark.parametrize("decision", ["unsupported", "ambiguous", "invalid"])
def test_triage_never_produces_draft_for_rejected_or_invalid_request(decision):
    class Triage:
        def __init__(self):
            self.calls = 0
        def complete(self, messages, tools):
            self.calls += 1
            assert self.calls == 1
            return {"role": "assistant", "tool_calls": [{"id": "triage", "type": "function",
                "function": {"name": "assess_goal", "arguments": json.dumps({"decision": decision})}}]}
    model = Triage()
    result = propose_goal(model, ReferenceProviders(manifests()), "Actual request", encoding="states")
    assert result["status"] == ("goal_draft_failed" if decision == "invalid" else "needs_input")
    assert "goal" not in result and model.calls == 1
