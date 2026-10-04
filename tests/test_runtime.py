"""The unified request route and observable effects through tensor programs."""
import pytest
import torch

from vectorpro.cells import TableCell
from vectorpro.host import HostContext
from vectorpro.learning import Capability, LearnerConfig, LearningPlan, OutputWidth
from vectorpro.learning.registry import program_provenance, unit_provenance
from vectorpro.machine import HALT, Instr, VectorProgram, assemble, disassemble
from vectorpro.runtime import VectorRuntime
from vectorpro.schemas import ScanSchema
from vectorpro.tasks import FULL_ADDER
from vectorpro.training import TrainConfig
from vectorpro.units import FunctionUnit


def add_known(runtime, name="add"):
    plan = LearningPlan(name, "sum with carry", 2, OutputWidth.PLUS_ONE)
    unit = FunctionUnit(name, ScanSchema(2, (0,)), TableCell.from_rule(FULL_ADDER))
    prov = unit_provenance(unit)
    runtime.registry.add(Capability(plan, runtime.registry.build(plan, prov), prov))


def store_program(runtime, name, instructions, inputs, registers, output):
    program = assemble(instructions, inputs, registers, output, runtime.registry.key_of)
    plan = LearningPlan(name, name, len(inputs), OutputWidth.SAME)
    prov = program_provenance(program, "runtime test")
    runtime.registry.add(Capability(plan, runtime.registry.build(plan, prov), prov))
    return program


def test_known_request_does_not_query_teacher_or_train(monkeypatch):
    runtime = VectorRuntime()
    add_known(runtime)
    def forbidden(*args):
        raise AssertionError("known requests must execute directly")
    monkeypatch.setattr(runtime.learner, "learn", forbidden)
    result = runtime.request("add", [(255, 1)], 8, source=forbidden)
    assert result.status == "executed" and result.outputs == [256]


def test_unknown_request_needs_evidence_and_never_guesses():
    runtime = VectorRuntime()
    result = runtime.request("new", [(2, 3)], 4)
    assert result.status == "needs_learning_examples" and result.outputs is None
    assert len(runtime.registry) == 0


def test_learn_then_execute_then_reload_and_grow(tmp_path):
    torch.manual_seed(0)
    config = LearnerConfig(train=TrainConfig(steps=800), restarts=2)
    runtime = VectorRuntime(config=config)
    plan = LearningPlan("xor", "exclusive or", 2, OutputWidth.SAME)
    result = runtime.request("xor", [(5, 3)], 8, plan=plan, source=lambda o, w: o[0] ^ o[1])
    assert result.status == "learned_and_executed" and result.outputs == [6]
    assert runtime.registry.get("xor").provenance["table"] == [[0], [1], [1], [0]]
    path = tmp_path / "program.json"
    runtime.save(path)
    loaded = VectorRuntime.load(path, config=config)
    assert loaded.request("xor", [(5, 3)], 8).outputs == [6]
    old_key = loaded.registry.key_of("xor").clone()
    add_known(loaded)
    assert not torch.equal(old_key, loaded.registry.key_of("add"))
    # The newly allocated key follows the original registry's generator state.
    add_known(runtime)
    assert torch.equal(loaded.registry.key_of("add"), runtime.registry.key_of("add"))
    loaded.save(path)
    again = VectorRuntime.load(path)
    assert again.request("add", [(7, 3)], 8).outputs == [10]


def test_unknown_request_can_learn_a_composition():
    runtime = VectorRuntime()
    add_known(runtime)
    plan = LearningPlan("double", "double", 1, OutputWidth.SAME)
    result = runtime.request("double", [(200,)], 8, plan=plan,
                             source=lambda o, w: (o[0] * 2) % (1 << w))
    assert result.status == "learned_and_executed" and result.outputs == [144]
    assert result.history[-1]["strategy"] == "reuse"
    assert runtime.registry.get("double").provenance["kind"] == "program"


def test_failed_learning_does_not_register_or_execute():
    runtime = VectorRuntime()
    plan = LearningPlan("product", "full product", 2, OutputWidth.DOUBLE, rounds=(8,))
    result = runtime.request("product", [(3, 5)], 8, plan=plan, source=lambda o, w: o[0] * o[1])
    assert result.status == "learning_failed" and result.outputs is None
    assert "product" not in runtime.registry


def test_tensor_program_executes_discarded_writes_and_reloads(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    add_known(runtime)
    (tmp_path / "input.bin").write_bytes(bytes([7, 3, 0]))
    src = host.put(b"input.bin")
    dst = host.put(b"output.bin")
    program = store_program(runtime, "sum_file", [
        Instr("file.read", ("src",), "buffer"),
        Instr("buffer.get", ("buffer", "zero"), "a"),
        Instr("buffer.get", ("buffer", "one"), "b"),
        Instr("add", ("a", "b"), "sum"),
        Instr("buffer.set", ("buffer", "zero", "sum")),
        Instr("file.write", ("dst", "buffer"), then=HALT),
    ], ["src", "dst"], dict(src="zero", dst="zero", buffer="zero", a="zero",
                             b="zero", sum="zero", zero="zero", one="one"), "sum")
    result = runtime.request("sum_file", [(src, dst)], 16)
    assert result.outputs == [10]
    assert (tmp_path / "output.bin").read_bytes() == bytes([10, 3, 0])
    assert [e["operation"] for e in host.events] == [
        "file.read", "buffer.get", "buffer.get", "buffer.set", "file.write"]
    assert "discard file.write" in "\n".join(disassemble(program, runtime.registry))
    assert runtime.registry.get("sum_file").executable.effects
    assert [op.name for op in runtime.registry.operators()] == ["add"]
    path = tmp_path / "program.json"
    runtime.save(path)
    new_host = HostContext(tmp_path)
    loaded = VectorRuntime.load(path, host=new_host)
    new_src, new_dst = new_host.put(b"input.bin"), new_host.put(b"reloaded.bin")
    assert loaded.request("sum_file", [(new_src, new_dst)], 16).outputs == [10]
    assert (tmp_path / "reloaded.bin").read_bytes() == bytes([10, 3, 0])


def test_old_program_data_loads_without_call_flags():
    runtime = VectorRuntime()
    add_known(runtime)
    program = store_program(runtime, "p", [Instr("add", ("a", "b"), "c"), Instr(then=HALT)],
                            ["a", "b"], dict(a="zero", b="zero", c="zero"), "c")
    data = program.to_data()
    del data["calls"]
    restored = VectorProgram.from_data(data)
    assert restored.calls.tolist() == [1, 0]


def test_branch_can_skip_effects(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    path, content = host.put(b"should-not-exist"), host.put(b"x")
    store_program(runtime, "conditional_write", [
        Instr(test="flag", then="write", otherwise=HALT),
        Instr("file.write", ("path", "content"), then=HALT, label="write"),
    ], ["flag", "path", "content"], dict(flag="zero", path="zero", content="zero"), "flag")
    assert runtime.request("conditional_write", [(0, path, content)], 8).outputs == [0]
    assert not (tmp_path / "should-not-exist").exists() and host.events == []
    assert runtime.request("conditional_write", [(1, path, content)], 8).outputs == [1]
    assert (tmp_path / "should-not-exist").read_bytes() == b"x"


def test_stateful_byte_memory_and_context_restoration(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    handle = runtime.request("buffer.new", [(3,)], 8).outputs[0]
    assert runtime.request("buffer.set", [(handle, 2, 99)], 8).outputs == [99]
    assert runtime.request("buffer.get", [(handle, 2)], 8).outputs == [99]
    assert runtime.request("buffer.length", [(handle,)], 8).outputs == [3]
    with pytest.raises(RuntimeError, match="active HostContext"):
        runtime.registry.run("buffer.length", [(handle,)], 8)


def test_host_errors_are_explicit_and_learning_search_excludes_effects(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    assert runtime.registry.operators() == []
    bad_path = host.put(b"../outside")
    with pytest.raises(ValueError, match="escapes"):
        runtime.request("file.read", [(bad_path,)], 8)
    missing_path = host.put(b"missing")
    with pytest.raises(FileNotFoundError):
        runtime.request("file.read", [(missing_path,)], 8)
    with pytest.raises(ValueError, match="single execution lane"):
        runtime.request("buffer.new", [(1,), (1,)], 8)
    hostless = VectorRuntime(runtime.registry)
    with pytest.raises(RuntimeError, match="HostContext"):
        hostless.request("buffer.new", [(1,)], 8)


def test_nested_effects_are_rejected_before_batched_program_runs(tmp_path):
    host = HostContext(tmp_path)
    runtime = VectorRuntime(host=host)
    runtime.provide_host_operations()
    store_program(runtime, "allocate", [Instr("buffer.new", ("length",), "handle", then=HALT)],
                  ["length"], dict(length="zero", handle="zero"), "handle")
    store_program(runtime, "outer", [Instr("allocate", ("length",), "handle", then=HALT)],
                  ["length"], dict(length="zero", handle="zero"), "handle")
    assert runtime.registry.has_effects
    assert runtime.registry.get("outer").executable.effects
    with host.activate(), pytest.raises(ValueError, match="single execution lane"):
        runtime.registry.run("outer", [(1,), (2,)], 8)
    assert host.events == [] and host.buffers == {}


def test_rejects_incompatible_requests_before_execution():
    runtime = VectorRuntime()
    add_known(runtime)
    for operands, width in [([(1,)], 8), ([(256, 0)], 8), ([], 8), ([(1, 2)], 0)]:
        with pytest.raises(ValueError):
            runtime.request("add", operands, width)
