import copy
import subprocess
import pytest
from vectorpro.host import HostContext
from vectorpro.learning.stateful import StateLesson, control_candidates
from vectorpro.machine import Instr
from vectorpro.runtime import VectorRuntime
from experiments.process_chain import SOURCE, lessons, saving_lesson, guarded_lesson, verify, verify_guard
from experiments.process_results import request_bytes


@pytest.fixture(scope="module")
def acquired(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain-learning")
    runtime = VectorRuntime.load(SOURCE, host=HostContext(root))
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("acquisition launched an OS process")
        patch.setattr(subprocess, "Popen", forbidden)
        outcomes = [runtime.teach_contract(**r) for r in [*lessons(), saving_lesson(), guarded_lesson()]]
    assert all(o["status"] == "registered" for o in outcomes), outcomes
    assert not list(root.iterdir()) and not runtime.host.events
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt", host=HostContext(root))
    assert loaded.contract("capture_to_file")["id"] == VectorRuntime.load(SOURCE).contract("capture_to_file")["id"]
    return loaded, outcomes


def test_chain_discovered_reloaded_native(acquired, tmp_path):
    runtime, outcomes = acquired
    assert len(verify(runtime, outcomes[1], tmp_path)) == 4
    assert outcomes[1]["history"][0]["steps"] == 2
    assert outcomes[1]["contract"]["execution"]["operations"] == ["process.run", "process.stdout"]


def test_learned_exit_guard_empty_success_existing_failure_and_signal(acquired, tmp_path):
    runtime, outcomes = acquired
    assert len(verify_guard(runtime, outcomes[3], tmp_path)) == 4
    assert outcomes[3]["history"][0]["control"] == "branch"
    assert "conditional routing" in outcomes[3]["contract"]["description"]


@pytest.mark.parametrize("failed_stage", [0, 1])
def test_chain_launch_error_stops_at_failing_stage(acquired, tmp_path, monkeypatch, failed_stage):
    runtime, outcomes = acquired
    runtime.host = HostContext(tmp_path)
    specs = [{"argv": ["/bin/cat"]}, {"argv": ["/bin/cat"]}]
    specs[failed_stage] = {"argv": ["/missing/vectorpro"]}
    original = subprocess.Popen
    launched = []
    def observed(*args, **kwargs):
        launched.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", observed)
    with pytest.raises(FileNotFoundError):
        runtime.call_contract(outcomes[1]["contract"]["id"], {
            "first": request_bytes(specs[0]).hex(), "second": request_bytes(specs[1]).hex(), "stdin": b"input".hex()}, 16)
    assert len(launched) == failed_stage + 1
    assert not list(tmp_path.iterdir())


def test_branch_only_grammar_preserves_branches_and_old_loop_default():
    instructions = (Instr("update", ("x0",), "t1"),)
    old = list(control_candidates(instructions, ("value",), ("value",), True))
    narrow = list(control_candidates(instructions, ("value",), ("value",), True, allow_loops=False))
    assert {candidate[2] for candidate in old} == {"sequence", "branch", "while"}
    assert {candidate[2] for candidate in narrow} == {"sequence", "branch"}
    assert [item for item in old if item[2] != "while"] == narrow
    state = copy.deepcopy(guarded_lesson()["state_lesson"])
    assert StateLesson.from_dict(state).branches_only
    state.pop("branches_only")
    assert not StateLesson.from_dict(state).branches_only
    state["branches_only"] = True
    state["control_flow"] = False
    with pytest.raises(ValueError, match="branches_only"):
        StateLesson.from_dict(state).validate()
