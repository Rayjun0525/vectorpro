import json
from pathlib import Path
import random

import pytest
import torch

from vectorpro.__main__ import main
from vectorpro.learning import ExampleLesson, ExampleSet, LearningPlan, OutputWidth
from vectorpro.runtime import VectorRuntime

ROOT = Path(__file__).resolve().parents[1]


def lesson_and_plan():
    data = json.loads((ROOT / "experiments/requests/learn_xor.json").read_text())
    return ExampleLesson.from_dict(data["lesson"]), LearningPlan.from_dict(data["plan"])


def test_data_only_learning_generalizes_beyond_supplied_inputs():
    torch.manual_seed(0)
    lesson, plan = lesson_and_plan()
    runtime = VectorRuntime()
    result = runtime.request("xor", [(4660, 255)], 16, plan=plan, lesson=lesson)
    assert result.status == "learned_and_executed" and result.outputs == [4811]
    rng = random.Random(7)
    operands = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(100)]
    assert runtime.request("xor", operands, 64).outputs == [a ^ b for a, b in operands]
    assert result.history[-1]["provided_train_accuracy"] == 1


@pytest.mark.parametrize("damage", ["empty", "unpaired", "duplicate", "target_range", "arity", "validation_count", "overlap"])
def test_invalid_lessons_cannot_mutate_the_registry(damage):
    lesson, plan = lesson_and_plan()
    if damage == "empty":
        lesson.training.operands.clear()
        lesson.training.targets.clear()
    elif damage == "unpaired":
        lesson.training.targets.pop()
    elif damage == "duplicate":
        lesson.training.operands[1] = lesson.training.operands[0]
    elif damage == "target_range":
        lesson.training.targets[0] = 16
    elif damage == "arity":
        lesson.training.operands[0] = (0,)
    elif damage == "validation_count":
        lesson.validation.operands.pop()
        lesson.validation.targets.pop()
    else:
        plan = LearningPlan("xor", "xor", 2, OutputWidth.SAME, train_width=4,
                            validation_width=4, validation_examples=1)
        lesson.validation = ExampleSet(4, [(0, 0)], [0])
    runtime = VectorRuntime()
    with pytest.raises(ValueError):
        runtime.request("xor", [(1, 2)], 4, plan=plan, lesson=lesson)
    assert len(runtime.registry) == 0


def test_cli_learn_save_run_and_missing_evidence(tmp_path, capsys):
    program = tmp_path / "program.json"
    assert main(["--program", str(program), "--request", str(ROOT / "experiments/requests/run_xor.json")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "needs_learning_examples"
    assert not program.exists()
    assert main(["--program", str(program), "--request", str(ROOT / "experiments/requests/learn_xor.json")]) == 0
    assert json.loads(capsys.readouterr().out)["outputs"] == [4811]
    assert main(["--program", str(program), "--request", str(ROOT / "experiments/requests/run_xor.json")]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "executed"
    assert response["outputs"] == [305441159, 4294967295]
    assert len(VectorRuntime.load(program).registry) == 1


def test_cli_byte_inputs_do_not_persist_transient_handles(tmp_path, capsys):
    program = tmp_path / "program.json"
    (tmp_path / "input.bin").write_bytes(b"hello")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"name": "file.read", "width": 16,
                                   "operands": [[{"utf8": "input.bin"}]], "output_format": "hex"}))
    args = ["--program", str(program), "--request", str(request), "--host-root", str(tmp_path)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["output_buffers_hex"] == [b"hello".hex()]
    (tmp_path / "input.bin").write_bytes(b"changed")
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["output_buffers_hex"] == [b"changed".hex()]
    assert "buffers" not in json.loads(program.read_text())


def test_cli_bad_request_preserves_existing_program(tmp_path, capsys):
    path = tmp_path / "program.json"
    VectorRuntime().save(path)
    before = path.read_bytes()
    request = tmp_path / "bad.json"
    request.write_text('{"name":"x","width":4,"operands":[[999]]}')
    assert main(["--program", str(path), "--request", str(request)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert path.read_bytes() == before
