"""Run the SAME acquired file on any installed Python/PyTorch host, without learning.

Only invoke on an OS where tests are authorized. The current project validates
this checker inside its one existing Linux test container.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile

from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def check(program):
    with tempfile.TemporaryDirectory(prefix="vectorpro-portable-") as directory:
        root = Path(directory)
        (root / "입력.bin").write_bytes(bytes(range(256)))
        (root / "other.bin").write_bytes(b"\x00\xff\x80")
        host = HostContext(root)
        model = VectorRuntime.load(program, host=host)
        source = host.put("입력.bin".encode("utf-8"))
        assert model.request("map", [(source, 53)], 16).outputs == [256]
        assert (root / "입력.bin").read_bytes() == bytes(v ^ 53 for v in range(256))
        assert model.request("guarded_map", [(source, 53, 0)], 16).outputs == [0]
        assert model.request("guarded_map", [(source, 53, 1)], 16).outputs == [256]
        assert (root / "입력.bin").read_bytes() == bytes(range(256))
        assert model.request("pair_map", [(source, host.put(b"other.bin"), 7)], 16).outputs == [3]
        assert (root / "other.bin").read_bytes() == b"\x07\xf8\x87"
        assert model.request("fill", [(source, 31)], 16).outputs == [256]
        assert (root / "입력.bin").read_bytes() == bytes([31]) * 256
        assert model.request("xor", [(12345, 4567)], 16).outputs == [12345 ^ 4567]
        assert model.request("sub", [(65535, 1)], 16).outputs == [65534]
        return {"platform": platform.system(), "python": platform.python_version(), "status": "passed",
                "same_program": str(program), "sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
                "checks": ["numeric", "mutable_loop", "conditional_loop", "multiple_loops", "unicode_paths"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=Path("results/initial_model/program.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.program)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
