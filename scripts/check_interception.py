"""Independent output checks after an assistant's real intercepted tool calls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

from vectorpro.runtime import VectorRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    records = [json.loads((root / name).read_text(encoding="utf-8-sig")) for name in
               ("02-unknown.json", "03-teach.json", "04-execute.json", "05-file.json")]
    assert [r["result"]["status"] for r in records] == ["needs_learning_examples", "learned", "executed", "executed"]
    assert records[2]["result"]["outputs"] == [61422]
    assert records[3]["result"]["outputs"] == [4]
    assert (root / "live-input.bin").read_bytes() == b"\x35\x32\xca\xb5"
    model = VectorRuntime.load(root / "assistant-program.json")
    rng = random.Random(112)
    passed = 0
    for i in range(100):
        width = 16 if i % 2 else 32
        a, b = rng.getrandbits(width), rng.getrandbits(width)
        expected = (~(a & b)) & ((1 << width) - 1)
        assert model.request("assistant_nand", [(a, b)], width).outputs == [expected]
        passed += 1
    summary = {"platform": platform.system(), "mode": "assistant_tool_interception",
               "scenario_chosen_by": "assistant", "external_model_api": False,
               "unknown_then_learned_then_executed": True, "held_out_numeric": passed,
               "widths": [16, 32], "native_file_hex": (root / "live-input.bin").read_bytes().hex(),
               "reload_without_llm": True, "original_program_sha256": hashlib.sha256((root / "program.json").read_bytes()).hexdigest()}
    (root / "interception-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
