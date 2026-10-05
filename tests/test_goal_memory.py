import copy
import json
import pytest
import torch
from vectorpro.goal_memory import GoalMemory
from vectorpro.goal_draft import propose_goal, accept_goal
from vectorpro.runtime import VectorRuntime
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.acquisition import EvidenceBank
from experiments.reference_acquisition import manifests

RULES = [{"kind": "unchanged", "parameter": "source"},
         {"kind": "equals_initial", "parameter": "destination", "source": "source"},
         {"kind": "unchanged_except", "parameters": ["destination"]}]
IDENTITY = {"encoder": "test", "revision": "fixed"}
EXAMPLES = [{"intent": "seed-copy", "rules": RULES},
            {"intent": "seed-move", "rules": [{"kind": "absent", "parameter": "source"}]},
            {"intent": "seed-unsupported", "rules": None}]


class Encoder:
    def __call__(self, texts):
        values = {"seed-copy": [1, 0, 0], "seed-move": [0, 1, 0], "seed-unsupported": [0, 0, 1],
                  "new-copy": [0.99, 0.1, 0], "ambiguous": [1, 1, 0], "unrelated": [1, 1, 1],
                  "unsupported": [0, 0, 1]}
        return torch.tensor([values[t] for t in texts], dtype=torch.float32)


def test_memory_proposes_review_never_approval_and_keeps_original_intent():
    memory = GoalMemory(EXAMPLES, Encoder(), IDENTITY, minimum=0.8, margin=0.1)
    result = propose_goal(None, ReferenceProviders(manifests()), "new-copy", goal_memory=memory)
    assert result["status"] == "needs_goal_review"
    assert result["goal"] == {"intent": "new-copy", "rules": RULES}
    assert result["intent_independently_verified"] is False
    assert accept_goal(result, result["proposal_sha256"]) == result["goal"]
    result["goal"]["rules"].clear()
    assert memory.examples[0]["rules"] == RULES


@pytest.mark.parametrize("intent", ["ambiguous", "unrelated", "unsupported"])
def test_weak_ambiguous_and_negative_match_do_not_propose_goal(intent):
    result = GoalMemory(EXAMPLES, Encoder(), IDENTITY, minimum=0.8, margin=0.1).propose(intent)
    assert result["status"] == "needs_input" and "goal" not in result


@pytest.mark.parametrize("suffix", [".pt", ".json"])
def test_memory_inside_program_reloads_and_survives_verified_reuse(tmp_path, suffix):
    runtime = VectorRuntime.load("results/reference_acquisition_gemma/program.pt")
    contracts = runtime.contracts()
    memory = GoalMemory(EXAMPLES, Encoder(), IDENTITY)
    memory.attach(runtime)
    program = tmp_path / ("program" + suffix)
    runtime.save(program)
    loaded = VectorRuntime.load(program)
    class QueryOnly(Encoder):
        def __call__(self, texts):
            assert texts == ["new-copy"]
            return super().__call__(texts)
    restored = GoalMemory.from_runtime(loaded, QueryOnly(), IDENTITY)
    assert restored.propose("new-copy")["status"] == "needs_goal_review"
    assert loaded.contracts() == contracts
    record = ReferenceProviders(manifests()).collect("copy", "new-copy")
    assert EvidenceBank([record], goal={"intent": "new-copy", "rules": RULES}).reuse(loaded, "new-copy", record["id"], program_path=program)["status"] == "reused"
    assert VectorRuntime.load(program)._goal_memory == runtime._goal_memory
    with pytest.raises(ValueError, match="identity"):
        GoalMemory.from_runtime(loaded, Encoder(), {"revision": "different"})


def test_corrupt_memory_rejected_before_serializing(tmp_path):
    runtime = VectorRuntime()
    GoalMemory(EXAMPLES, Encoder(), IDENTITY).attach(runtime)
    program = tmp_path / "program.json"
    runtime.save(program)
    original = program.read_bytes()
    runtime._goal_memory["vectors"] = [[float("nan")]]
    with pytest.raises(ValueError):
        runtime.save(program)
    assert program.read_bytes() == original
    data = json.loads(original)
    data["goal_memory"]["vectors"] = [[1, 0, 0]]
    program.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        VectorRuntime.load(program)


def test_memory_does_not_cross_interface_names():
    examples = copy.deepcopy(EXAMPLES)
    examples[0]["rules"][0]["parameter"] = "unknown"
    memory = GoalMemory(examples, Encoder(), IDENTITY)
    assert propose_goal(None, ReferenceProviders(manifests()), "new-copy", goal_memory=memory)["status"] == "goal_draft_failed"


@pytest.mark.parametrize("suffix", [".pt", ".json"])
@pytest.mark.parametrize("routing", ["ridge", "ridge_consensus"])
def test_learned_router_reloads_without_retraining(tmp_path, suffix, routing, monkeypatch):
    memory = GoalMemory(EXAMPLES, Encoder(), IDENTITY, routing=routing)
    result = memory.propose("new-copy")
    assert result["status"] == "needs_goal_review" and result["goal"]["rules"] == RULES
    assert result["retrieval"]["score_is_probability"] is False
    assert "similarity" not in result["retrieval"]
    for text in ("ambiguous", "unrelated", "unsupported"):
        assert memory.propose(text)["status"] == "needs_input"
    runtime = VectorRuntime()
    memory.attach(runtime)
    path = tmp_path / ("router" + suffix)
    runtime.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved router must not be fitted again")
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    restored = GoalMemory.from_runtime(VectorRuntime.load(path), Encoder(), IDENTITY)
    assert restored.propose("new-copy") == result


def test_consensus_rejects_confident_disagreement():
    memory = GoalMemory(EXAMPLES, Encoder(), IDENTITY, routing="ridge_consensus")
    # Simulate a strongly wrong learned class; the independent nearest gate must block it.
    memory.router["weights"] = [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
    result = memory.propose("new-copy")
    assert result["status"] == "needs_input"
    assert result["retrieval"]["consensus"] is False


@pytest.mark.parametrize("mutation", ["shape", "nonfinite", "groups", "ridge", "threshold"])
def test_invalid_learned_router_rejected(mutation):
    data = GoalMemory(EXAMPLES, Encoder(), IDENTITY, routing="ridge").to_data()
    if mutation == "shape":
        data["router"]["weights"] = [[1]]
    elif mutation == "nonfinite":
        data["router"]["weights"][0][0] = float("nan")
    elif mutation == "groups":
        data["router"]["groups"] = [1, 0, 2]
    elif mutation == "ridge":
        data["router"]["ridge"] = 0
    else:
        data["router"]["minimum"] = 1.1
    with pytest.raises(ValueError):
        GoalMemory.validate_data(data)
