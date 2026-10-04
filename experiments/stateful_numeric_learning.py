"""Learn numeric behaviour from data, then reuse it in a learned file procedure."""
import json
from pathlib import Path
import random

import torch

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/stateful_numeric_model"
    folder.mkdir(parents=True, exist_ok=True)
    context = HostContext(folder)
    runtime = VectorRuntime(host=context)
    torch.manual_seed(0)
    numeric = json.loads((root / "experiments/requests/learn_xor.json").read_text())
    numeric["plan"]["name"] = "combine"
    first = runtime.request("combine", [(1, 2)], 16,
                            plan=LearningPlan.from_dict(numeric["plan"]),
                            lesson=ExampleLesson.from_dict(numeric["lesson"]))
    assert first.status == "learned_and_executed"
    runtime.provide_host_operations()
    def observe(path, content):
        return StateExample((path,), {path: content}, {path: content}, content[0])
    observation = StateLesson(("path",),
                             [observe("a", b"\x07\x00"), observe("b", b"\xa5\x11")],
                             [observe("c", b"\xff\x01"), observe("d", b"\x10\x12")], max_steps=2)
    assert learn_stateful(runtime.registry, "observe", observation).capability is not None
    def combined(path, content, mask):
        return StateExample((path, mask), {path: content}, {path: content}, content[0] ^ mask)
    lesson = StateLesson(("path", "value"),
                         [combined("a", b"\x07\x00", 3), combined("b", b"\xa5\x11", 15)],
                         [combined("c", b"\xff\x01", 170), combined("d", b"\x10\x12", 9)], max_steps=2)
    (folder / "input.bin").write_bytes(b"\x63\x00\xff")
    response = runtime.request("file_compute", [(context.put(b"input.bin"), 513)], 16, state_lesson=lesson)
    assert response.outputs == [610]
    cap = runtime.registry.get("file_compute")
    rng = random.Random(20261004)
    for i in range(100):
        content = bytes(rng.getrandbits(8) for _ in range(rng.randrange(1, 512)))
        assert evaluate(cap.executable, combined(f"unseen{i}", content, rng.randrange(65536)), lesson)
    runtime.save(folder / "program.json")
    fresh = HostContext(folder)
    reloaded = VectorRuntime.load(folder / "program.json", host=fresh)
    assert reloaded.request("file_compute", [(fresh.put(b"input.bin"), 513)], 16).outputs == [610]
    summary = {"numeric_learning": first.history, "state_learning": response.history,
               "native_output": 610, "heldout_cases": 100, "heldout_correct": 100,
               "reload_identical": True, "procedure_authored": False,
               "explanation": runtime.registry.explain("file_compute"),
               "scope": "straight-line reuse of learned arithmetic and typed state procedures"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
