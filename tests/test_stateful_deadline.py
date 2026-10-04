import pytest

import vectorpro.learning.stateful as stateful
from vectorpro.runtime import VectorRuntime


def lesson():
    return stateful.StateLesson(("value",),
                                [stateful.StateExample((2,), {}, {}, 2)],
                                [stateful.StateExample((3,), {}, {}, 3)])


def test_deadline_returns_failure_without_registration(monkeypatch):
    model = VectorRuntime()
    model.provide_host_operations()
    example = lesson()
    example.time_budget_seconds = 1
    moments = iter((0.0, 2.0))
    monkeypatch.setattr(stateful, "monotonic", lambda: next(moments))
    outcome = stateful.learn_stateful(model.registry, "expired", example)
    assert outcome.capability is None and "expired" not in model.registry
    assert outcome.history == [{"candidates": 0, "accepted": False, "reason": "time budget"}]


@pytest.mark.parametrize("expiry_case", [1, 2])
def test_expiry_after_passing_training_or_validation_cannot_register(monkeypatch, expiry_case):
    model = VectorRuntime()
    model.provide_host_operations()
    example = lesson()
    example.time_budget_seconds = 1
    clock = {"now": 0.0, "evaluations": 0}
    monkeypatch.setattr(stateful, "monotonic", lambda: clock["now"])

    def passing_case(*args):
        clock["evaluations"] += 1
        if clock["evaluations"] == expiry_case:
            clock["now"] = 2.0
        return True

    monkeypatch.setattr(stateful, "evaluate", passing_case)
    outcome = stateful.learn_stateful(model.registry, "expired_fit", example)
    assert outcome.capability is None and "expired_fit" not in model.registry
    assert outcome.history[-1]["reason"] == "time budget"
    assert clock["evaluations"] == expiry_case


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "60"])
def test_invalid_deadline_is_rejected(budget):
    example = lesson()
    example.time_budget_seconds = budget
    with pytest.raises(ValueError, match="positive finite"):
        example.validate()


def test_deadline_json_preserves_old_default_and_explicit_budget():
    data = {"input_types": ["value"],
            "training": [{"inputs": [2], "before": {}, "after": {}, "output": 2}],
            "validation": [{"inputs": [3], "before": {}, "after": {}, "output": 3}]}
    assert stateful.StateLesson.from_dict(data).time_budget_seconds == 60.0
    data["time_budget_seconds"] = 2.5
    parsed = stateful.StateLesson.from_dict(data)
    parsed.validate()
    assert parsed.time_budget_seconds == 2.5
