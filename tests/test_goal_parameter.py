import json
import pytest
from vectorpro.goal_draft import propose_goal
from vectorpro.reference_evidence import ReferenceProviders
from experiments.reference_acquisition import manifests


class Model:
    def __init__(self, answers):
        self.answers, self.calls = answers, []
    def complete(self, messages, tools):
        assert tools[0]["function"]["name"] == "describe_parameter"
        assert messages[-1]["content"] == "Unseen request"
        name = json.loads(messages[0]["content"].split("Parameter to describe: ")[1])["name"]
        self.calls.append(name)
        return {"role": "assistant", "tool_calls": [{"id": "actual", "type": "function",
            "function": {"name": "describe_parameter", "arguments": json.dumps({"state": self.answers[name]})}}]}


@pytest.mark.parametrize("answers,changes", [
    ({"source": "unchanged", "destination": "initial:source"}, ["destination"]),
    ({"source": "absent", "destination": "initial:source"}, ["source", "destination"]),
    ({"source": "unchanged", "destination": "unchanged"}, []),
    ({"source": "initial:destination", "destination": "unchanged"}, ["source"])])
def test_parameter_answers_are_isolated_then_compiled(answers, changes):
    model = Model(answers)
    draft = propose_goal(model, ReferenceProviders(manifests()), "Unseen request", encoding="per_parameter")
    assert draft["status"] == "needs_goal_review"
    assert draft["goal"]["rules"][-1]["parameters"] == changes
    assert model.calls == ["source", "destination"]


@pytest.mark.parametrize("answer,expected", [("unsupported", "needs_input"), ("unclear", "needs_input"), ("invalid", "goal_draft_failed")])
def test_rejection_or_invalid_first_answer_never_continues(answer, expected):
    model = Model({"source": answer})
    result = propose_goal(model, ReferenceProviders(manifests()), "Unseen request", encoding="per_parameter")
    assert result["status"] == expected and "goal" not in result
    assert model.calls == ["source"]
