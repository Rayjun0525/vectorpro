import copy
import pytest
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.runtime import VectorRuntime
from vectorpro.learning.stateful import StateLesson
from experiments.record_results import packed, requests, verify


@pytest.mark.parametrize("data,value", [(b"", 0), (b"\0\xff\n", 17), ("한글".encode(), (1 << 64) - 1)])
def test_record_portable_bytes_and_width(data, value, tmp_path):
    records = []
    for host in (MemoryHostContext({}), HostContext(tmp_path)):
        result = host.invoke("record.pack", (host.put(data), value), 64)
        records.append(bytes(host.buffers[result]))
        assert records[-1].hex() == packed(data, value)
        fresh = MemoryHostContext({})
        handle = fresh.put(records[-1])
        output = fresh.invoke("record.buffer", (handle,), 64)
        assert bytes(fresh.buffers[output]) == data
        assert fresh.invoke("record.value", (handle,), 64) == value
        if value > 65535:
            with pytest.raises(ValueError, match="width"):
                fresh.invoke("record.value", (handle,), 16)
    assert records[0] == records[1]


@pytest.mark.parametrize("data", [b"", b"VPR2" + bytes(16), b"VPR1" + bytes(16) + b"extra", b"VPR1" + b"\x01" + bytes(15)])
def test_invalid_records_reject_without_file_mutation(data):
    host = MemoryHostContext({"keep": b"keep"})
    handle = host.put(data)
    for operation in ("record.buffer", "record.value"):
        with pytest.raises(ValueError, match="record"):
            host.invoke(operation, (handle,), 16)
    assert host.files == {"keep": b"keep"} and not host.events


def test_record_limits_and_state_output_validation():
    host = MemoryHostContext({}, max_buffer_bytes=20)
    with pytest.raises(ValueError, match="byte limit"):
        host.invoke("record.pack", (host.put(b"x"), 0), 16)
    for invalid in (-1, 1 << 64):
        with pytest.raises(ValueError, match="uint64"):
            host.invoke("record.pack", (host.put(b""), invalid), 16)
    lesson = requests()[0]["state_lesson"]
    for invalid in (3, "x", "f"):
        changed = copy.deepcopy(lesson)
        changed["training"][0]["output"] = invalid
        with pytest.raises(ValueError, match="hexadecimal"):
            StateLesson.from_dict(changed).validate()


def test_learn_results_reload_and_connect_without_native_learning(tmp_path):
    runtime = VectorRuntime.load("results/list_selection_verified/program.pt", host=HostContext(tmp_path))
    outcomes = [runtime.teach_contract(**r) for r in requests()]
    assert all(o["status"] == "registered" for o in outcomes), outcomes
    assert not list(tmp_path.iterdir()) and not runtime.host.events
    runtime.provide_host_operations(["record.value"])
    runtime.save(tmp_path / "program.pt")
    loaded = VectorRuntime.load(tmp_path / "program.pt", host=HostContext(tmp_path))
    assert len(verify(loaded, outcomes, tmp_path)) == 4
    assert runtime.contract("selected_copy")["id"] == loaded.contract("selected_copy")["id"]


def test_incorrect_buffer_validation_fails_atomically(tmp_path):
    runtime = VectorRuntime.load("results/list_selection_verified/program.pt", host=HostContext(tmp_path))
    initial = runtime.contracts()
    request = requests()[0]
    request["state_lesson"]["validation"][0]["output"] = packed(b"wrong", 0)
    outcome = runtime.teach_contract(**request)
    assert outcome["status"] == "learning_failed"
    assert runtime.contracts() == initial
    assert not list(tmp_path.iterdir()) and not runtime.host.events
