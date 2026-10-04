"""Learn guards and feedback loops from goals/demonstrations, not authored programs."""
import json
from pathlib import Path
import random

import torch

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


def copy_case(path, target, flag, content):
    before = {path: content, target: b"previous", "keep": b"unchanged"}
    return StateExample((path, target, flag), before,
                        before | ({target: content} if flag else {}), len(content) if flag else 0)


def read_case(path, value):
    files = {path: b"unchanged"}
    return StateExample((path, value), files, dict(files), 0,
                        operations=("file.read",) * value.bit_length())


def main():
    torch.manual_seed(0)
    root = Path("results/stateful_control_model")
    root.mkdir(parents=True, exist_ok=True)
    model = VectorRuntime(host=HostContext(root))
    values = (0, 1, 2, 127, 128, 255)
    numeric = ExampleLesson.from_dict({
        "training": {"width": 4, "operands": [[x] for x in range(16)], "targets": [x >> 1 for x in range(16)]},
        "validation": {"width": 8, "operands": [[x] for x in values], "targets": [x >> 1 for x in values]}})
    plan = LearningPlan.from_dict({"name": "advance", "description": "example-defined step", "arity": 1,
                                   "output": "W", "rounds": [16], "train_width": 4,
                                   "validation_width": 8, "validation_examples": 6})
    assert model.request("advance", [(9,)], 16, plan=plan, lesson=numeric).outputs == [4]
    model.provide_host_operations()
    branch = StateLesson(("path", "path", "value"),
                         [copy_case("a", "b", 0, b"abc"), copy_case("c", "d", 1, b"xyz")],
                         [copy_case("e", "f", 0, b"long"), copy_case("g", "h", 9, b"new\x00")],
                         max_steps=2, candidate_budget=20000, control_flow=True)
    loop = StateLesson(("path", "value"),
                       [read_case("a", 0), read_case("b", 3), read_case("c", 9)],
                       [read_case("d", 1), read_case("e", 31)],
                       max_steps=2, candidate_budget=20000, control_flow=True)
    first = learn_stateful(model.registry, "conditional", branch)
    second = learn_stateful(model.registry, "repeat", loop)
    assert first.capability is not None and second.capability is not None
    rng = random.Random(41)
    branch_passes = loop_passes = 0
    for i in range(100):
        content = bytes(rng.randrange(256) for _ in range(rng.randrange(512)))
        flag = 0 if i % 2 else rng.randrange(1, 65536)
        branch_passes += evaluate(first.capability.executable, copy_case(f"s{i}", f"d{i}", flag, content), branch)
        count = 0 if i == 0 else rng.randrange(65536)
        loop_passes += evaluate(second.capability.executable, read_case(f"r{i}", count), loop)
    assert branch_passes == loop_passes == 100
    (root / "native.bin").write_bytes(b"native\x00\xff")
    (root / "target.bin").write_bytes(b"previous")
    model.request("conditional", [(model.host.put(b"native.bin"), model.host.put(b"target.bin"), 0)], 16)
    assert (root / "target.bin").read_bytes() == b"previous"
    model.save(root / "program.json")
    host = HostContext(root)
    restored = VectorRuntime.load(root / "program.json", host=host)
    assert restored.request("conditional", [(host.put(b"native.bin"), host.put(b"target.bin"), 5)], 16).outputs == [8]
    assert (root / "target.bin").read_bytes() == b"native\x00\xff"
    host.events.clear()
    restored.request("repeat", [(host.put(b"native.bin"), 1024)], 16)
    assert tuple(e["operation"] for e in host.events) == ("file.read",) * 11
    summary = {"branch": first.history, "loop": second.history,
               "held_out_branch": branch_passes, "held_out_loop": loop_passes,
               "native_and_reload": True, "task_program_authored": False,
               "branch_program": model.registry.explain("conditional"),
               "loop_program": model.registry.explain("repeat"),
               "scope": "one guard or one numeric-feedback while region; optional observed host-effect traces",
               "trace_note": "Repeated reads leave the same final files; demonstrated effects supply otherwise missing evidence."}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
