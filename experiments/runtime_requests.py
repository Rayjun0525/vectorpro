"""Request -> learn -> execute -> file effects -> save -> reload, in one runtime.

The addition rule is learned. The file workflow is assembled to demonstrate
execution machinery; learning I/O workflows is not yet implemented.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from vectorpro.host import HostContext
from vectorpro.learning import Capability, LearningPlan, OutputWidth
from vectorpro.learning.registry import program_provenance
from vectorpro.machine import HALT, Instr, assemble
from vectorpro.runtime import VectorRuntime


def main():
    folder = Path(__file__).resolve().parents[1] / "results" / "runtime_requests"
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261004)
    host = HostContext(folder)
    runtime = VectorRuntime(host=host, seed=20261004)
    runtime.provide_host_operations()
    unknown = runtime.request("add", [(7, 3)], 8)
    learned = runtime.request("add", [(7, 3)], 8,
                              plan=LearningPlan("add", "sum with carry", 2, OutputWidth.PLUS_ONE),
                              source=lambda o, w: o[0] + o[1])
    if learned.status != "learned_and_executed" or learned.outputs != [10]:
        raise AssertionError(asdict(learned))
    known = runtime.request("add", [(200, 100)], 8)
    assert known.outputs == [300] and known.status == "executed"
    program = assemble([
        Instr("file.read", ("src",), "buffer"),
        Instr("buffer.get", ("buffer", "zero"), "a"),
        Instr("buffer.get", ("buffer", "one"), "b"),
        Instr("add", ("a", "b"), "sum"),
        Instr("buffer.set", ("buffer", "zero", "sum")),
        Instr("file.write", ("dst", "buffer"), then=HALT),
    ], ["src", "dst"], dict(src="zero", dst="zero", buffer="zero", a="zero", b="zero",
                             sum="zero", zero="zero", one="one"), "sum", runtime.registry.key_of)
    plan = LearningPlan("sum_file", "sum first two bytes and replace the first byte", 2, OutputWidth.SAME)
    provenance = program_provenance(program, "provided I/O demonstration")
    runtime.registry.add(Capability(plan, runtime.registry.build(plan, provenance), provenance))
    (folder / "input.bin").write_bytes(bytes([7, 3]))
    src, dst = host.put(b"input.bin"), host.put(b"output.bin")
    file_result = runtime.request("sum_file", [(src, dst)], 16)
    assert (folder / "output.bin").read_bytes() == bytes([10, 3])
    runtime.save(folder / "program.json")
    fresh_host = HostContext(folder)
    loaded = VectorRuntime.load(folder / "program.json", host=fresh_host)
    assert loaded.request("add", [(200, 100)], 8).outputs == [300]
    new_src, new_dst = fresh_host.put(b"input.bin"), fresh_host.put(b"reloaded.bin")
    assert loaded.request("sum_file", [(new_src, new_dst)], 16).outputs == [10]
    assert (folder / "reloaded.bin").read_bytes() == bytes([10, 3])
    summary = {"unknown": asdict(unknown), "learned": asdict(learned), "known": asdict(known),
               "file_request": asdict(file_result), "effect_log": host.events,
               "reload_identical": True, "io_workflow_learned": False}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
