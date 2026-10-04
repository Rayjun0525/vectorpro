"""End-to-end initial-model proof: mutable loops, reuse, native I/O and reload."""
import json
from pathlib import Path
import random

from vectorpro.host import HostContext
from vectorpro.learning import Capability, Registry
from vectorpro.learning.stateful import StateExample, StateLesson, evaluate, learn_stateful
from vectorpro.runtime import VectorRuntime


def mutation(path, mask, content, fill=False):
    before = {path: content, "keep": b"untouched"}
    after = bytes([mask]) * len(content) if fill else bytes(v ^ mask for v in content)
    return StateExample((path, mask), before, before | {path: after}, len(content))


def guarded(path, mask, flag, content):
    before = {path: content}
    return StateExample((path, mask, flag), before,
                        {path: bytes(v ^ mask for v in content) if flag else content}, len(content) if flag else 0)


def pair(a, b, mask, first, second):
    before = {a: first, b: second}
    return StateExample((a, b, mask), before,
                        {a: bytes(v ^ mask for v in first), b: bytes(v ^ mask for v in second)}, len(second))


def build(root):
    acquired = Registry.load(Path("results/four_ops_registry_20261004.json"))
    model = VectorRuntime(host=HostContext(root))
    for name in ("sub", "xor"):
        cap = acquired.get(name)
        model.registry.add(Capability(cap.plan, model.registry.build(cap.plan, cap.provenance), cap.provenance, cap.history))
    model.provide_host_operations()
    lessons = {}
    for fill, name in ((True, "fill"), (False, "map")):
        lessons[name] = StateLesson(("path", "value"),
            [mutation("a", 7, b"\x01\x02\x03", fill), mutation("b", 0, b"", fill)],
            [mutation("c", 31, b"\xff\x00\x80\x05", fill), mutation("d", 2, b"\x10", fill)],
            max_steps=5 if fill else 7, buffer_loops=True, execution_budget=20000)
        assert learn_stateful(model.registry, name, lessons[name]).capability is not None
    lessons["guarded_map"] = StateLesson(("path", "value", "value"),
        [guarded("a", 7, 0, b"abc"), guarded("b", 5, 1, b"xyz")],
        [guarded("c", 9, 2, b"\x00\xff"), guarded("d", 8, 0, b"\x80")], max_steps=1, control_flow=True)
    assert learn_stateful(model.registry, "guarded_map", lessons["guarded_map"]).capability is not None
    lessons["pair_map"] = StateLesson(("path", "path", "value"),
        [pair("a", "b", 7, b"abc", b"xyz"), pair("c", "d", 5, b"x", b"yz")],
        [pair("e", "f", 9, b"\x00", b"\xff\x80"), pair("g", "h", 8, b"", b"xx")], max_steps=2, candidate_budget=20000)
    assert learn_stateful(model.registry, "pair_map", lessons["pair_map"]).capability is not None
    return model, lessons


def main():
    root = Path("results/initial_model")
    root.mkdir(parents=True, exist_ok=True)
    model, lessons = build(root)
    rng = random.Random(81)
    passed = {name: 0 for name in lessons}
    for i in range(100):
        content = bytes(rng.randrange(256) for _ in range(rng.randrange(64)))
        other = bytes(rng.randrange(256) for _ in range(rng.randrange(64)))
        mask = rng.randrange(256)
        cases = {"fill": mutation(f"f{i}", mask, content, True),
                 "map": mutation(f"m{i}", mask, content),
                 "guarded_map": guarded(f"g{i}", mask, i % 3, content),
                 "pair_map": pair(f"a{i}", f"b{i}", mask, content, other)}
        for name, case in cases.items():
            passed[name] += evaluate(model.registry.get(name).executable, case, lessons[name])
    assert all(count == 100 for count in passed.values())
    for name, fill in (("fill", True), ("map", False)):
        assert evaluate(model.registry.get(name).executable,
                        mutation("large-unseen", 53, bytes(range(256)) * 8, fill), lessons[name])
    (root / "native.bin").write_bytes(bytes(range(256)))
    args = (model.host.put(b"native.bin"), 53)
    assert model.request("map", [args], 16).outputs == [256]
    expected = bytes(v ^ 53 for v in range(256))
    assert (root / "native.bin").read_bytes() == expected
    model.save(root / "program.json")
    host = HostContext(root)
    restored = VectorRuntime.load(root / "program.json", host=host)
    assert restored.request("map", [(host.put(b"native.bin"), 53)], 16).outputs == [256]
    assert (root / "native.bin").read_bytes() == bytes(range(256))
    summary = {"held_out": passed, "larger_files": 2048, "native_and_reload": True,
               "task_program_authored": False, "effect_trace_required": False,
               "numeric_library": "previously acquired sub/xor tables", "grammar": "generic indexed-buffer fill/map plus learned-procedure composition",
               "capabilities": {name: {"history": model.registry.get(name).history, "listing": model.registry.explain(name)} for name in lessons}}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "capabilities"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
