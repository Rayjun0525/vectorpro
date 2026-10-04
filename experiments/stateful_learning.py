"""Initial-model demonstration: only state examples, no authored instruction recipe."""
import json
from pathlib import Path
import random

from vectorpro.host import HostContext
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate
from vectorpro.runtime import VectorRuntime


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/stateful_model"
    folder.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "experiments/requests/learn_transfer.json").read_text())
    lesson = StateLesson.from_dict(data["state_lesson"])
    host = HostContext(folder)
    runtime = VectorRuntime(host=host)
    (folder / "input.bin").write_bytes(bytes([9, 0, 255, 42]))
    inputs = (host.put(b"input.bin"), host.put(b"output.bin"))
    result = runtime.request(data["name"], [inputs], 16, state_lesson=lesson)
    assert result.status == "learned_and_executed" and result.outputs == [4]
    assert (folder / "output.bin").read_bytes() == bytes([9, 0, 255, 42])
    executable = runtime.registry.get(data["name"]).executable
    rng = random.Random(20261004)
    lengths = [0, 1, 2, 256, 4096] + [rng.randrange(1, 1024) for _ in range(15)]
    for i, length in enumerate(lengths):
        source, destination = f"unseen{i}", f"target{i}"
        content = bytes(rng.getrandbits(8) for _ in range(length))
        initial = {source: content, destination: b"old", "protected": b"keep"}
        target = initial | {destination: content}
        assert evaluate(executable, StateExample((source, destination), initial, target, length), lesson)
    runtime.save(folder / "program.json")
    fresh = HostContext(folder)
    restored = VectorRuntime.load(folder / "program.json", host=fresh)
    new_inputs = (fresh.put(b"input.bin"), fresh.put(b"reloaded.bin"))
    assert restored.request(data["name"], [new_inputs], 16).outputs == [4]
    assert (folder / "reloaded.bin").read_bytes() == bytes([9, 0, 255, 42])
    summary = {"status": result.status, "history": result.history,
               "heldout_cases": len(lengths), "heldout_correct": len(lengths),
               "native_output_hex": (folder / "output.bin").read_bytes().hex(),
               "reload_identical": True, "procedure_authored": False,
               "explanation": runtime.registry.explain(data["name"]),
               "scope": "bounded straight-line state example synthesis; no LLM or CPU trace ingestion"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
