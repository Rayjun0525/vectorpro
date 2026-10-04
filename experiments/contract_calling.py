"""Same saved tensor file, contract IDs and native roots; no LLM or encoder."""
import hashlib
import argparse
import json
from pathlib import Path
import sys

from vectorpro.contracts import tool_schema
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/contracts_model"))
    root = parser.parse_args().output
    if root.exists():
        raise RuntimeError("preserve results; choose a new --output root before repeating")
    root.mkdir()
    runtime = VectorRuntime.load("results/catalog_retrieval/final/program.pt")
    program = root / "program.pt"
    runtime.save(program)
    ids = {name: runtime.contract(name)["id"] for name in ("xor", "map", "fill", "file.write")}
    cases = []
    for index in range(2):
        host_root = root / f"native-{index}"
        host_root.mkdir()
        target = host_root / "새 파일.bin"
        target.write_bytes(bytes.fromhex("0017ff62"))
        (host_root / "keep.bin").write_bytes(b"keep")
        loaded = VectorRuntime.load(program, host=HostContext(host_root))
        assert ids == {name:loaded.contract(name)["id"] for name in ids}
        numeric = loaded.call_contract(ids["xor"], {"x0":73, "x1":9}, 16)
        mapped = loaded.call_contract(ids["map"], {"x0":"새 파일.bin", "x1":13}, 24)
        written = loaded.call_contract(ids["file.write"], {"x0":"메시지.bin", "x1":b"abc".hex()}, 16)
        rejected = []
        before = {p.name:p.read_bytes() for p in host_root.iterdir()}
        for args in ({"x0":"../escape.bin", "x1":7}, {"x0":"새 파일.bin", "x1":True}):
            try:
                loaded.call_contract(ids["map"], args, 16)
            except ValueError as error:
                rejected.append(str(error))
            else:
                raise AssertionError("invalid call executed")
        assert before == {p.name:p.read_bytes() for p in host_root.iterdir()}
        assert numeric["outputs"] == [64]
        assert target.read_bytes().hex() == "0d1af26f"
        assert (host_root/"메시지.bin").read_bytes() == b"abc"
        assert (host_root/"keep.bin").read_bytes() == b"keep"
        cases.append({"root":str(host_root), "numeric":numeric,"map":mapped,"write":written,
                      "rejected":rejected, "files":{p.name:p.read_bytes().hex() for p in host_root.iterdir()}})
    assert "transformers" not in sys.modules and "llama_cpp" not in sys.modules
    report = {"program_sha256":hashlib.sha256(program.read_bytes()).hexdigest(),
              "contract_ids":ids, "contracts":runtime.contracts(),
              "tool_schema":tool_schema(runtime.contract("file.write")), "cases":cases,
              "llm_imported":False, "encoder_imported":False,
              "scope":"execution of previously learned functions; not new learning or natural-language accuracy"}
    (root/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"roots":len(cases),"passed":True,"llm_imported":False,"program_sha256":report["program_sha256"]}))


if __name__=="__main__":
    main()
