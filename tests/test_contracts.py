import json
from pathlib import Path

import pytest
import torch

from vectorpro.contracts import tool_schema
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime

SOURCE = Path(__file__).resolve().parents[1] / "results/initial_model/program.json"


def runtime(root=None):
    return VectorRuntime.load(SOURCE, host=HostContext(root) if root else None)


@pytest.mark.parametrize("suffix", [".json", ".pt"])
def test_contract_roundtrip_direct_call_and_registry_growth(tmp_path, suffix, monkeypatch):
    r = runtime()
    before = r.contract("xor")
    path = tmp_path / ("program" + suffix)
    r.save(path)
    loaded = VectorRuntime.load(path)
    assert loaded.contract("xor") == before
    monkeypatch.setattr(loaded.learner, "learn", lambda *a: pytest.fail("must not teach"))
    assert loaded.call_contract(before["id"], {"x1": 9, "x0": 73}, 16)["outputs"] == [64]
    loaded.provide_host_operations()
    assert loaded.contract("xor")["id"] == before["id"]
    before["parameters"][0]["role"] = "corrupted external object"
    assert loaded.contract("xor")["parameters"][0]["role"] != before["parameters"][0]["role"]
    if suffix == ".pt":
        payload = torch.load(path, weights_only=True)
        assert all(isinstance(v, torch.Tensor) for v in payload.values())


def test_content_id_tracks_dependency_changes_and_rejects_stale_call(tmp_path):
    r = runtime(tmp_path)
    old = r.contract("map")["id"]
    r.registry.get("xor").provenance["table"][0][0] = 1
    assert r.contract("map")["id"] != old
    with pytest.raises(KeyError, match="stale"):
        r.call_contract(old, {"x0":"target.bin", "x1":7}, 16)
    assert r.host.events == [] and r.host.buffers == {}


def test_native_file_call_reloads_on_new_root_without_model(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    r = runtime(first)
    contract = r.contract("map")
    r.save(tmp_path / "program.pt")
    (second / "새 파일.bin").write_bytes(bytes.fromhex("0017ff62"))
    (second / "keep.bin").write_bytes(b"keep")
    loaded = VectorRuntime.load(tmp_path / "program.pt", host=HostContext(second))
    result = loaded.call_contract(contract["id"], {"x0":"새 파일.bin", "x1":13}, 24)
    assert result["outputs"] == [4]
    assert (second / "새 파일.bin").read_bytes() == bytes.fromhex("0d1af26f")
    assert (second / "keep.bin").read_bytes() == b"keep"
    assert list(first.iterdir()) == []


@pytest.mark.parametrize("arguments,width,version", [
    ({"x0":"target.bin", "x1":True},16,1),
    ({"x0":"target.bin", "x1":65536},16,1),
    ({"x0":42, "x1":7},16,1),
    ({"x0":"../escape.bin", "x1":7},16,1),
    ({"x0":"target.bin", "x1":7, "extra":0},16,1),
    ({"x0":"target.bin"},16,1),
    ({"x0":"target.bin", "x1":7},True,1),
    ({"x0":"target.bin", "x1":7},16,2),
])
def test_invalid_calls_have_no_native_effects_or_allocations(tmp_path, arguments, width, version):
    r = runtime(tmp_path)
    path = tmp_path / "target.bin"; path.write_bytes(b"keep")
    with pytest.raises((ValueError, KeyError)):
        r.call_contract(r.contract("map")["id"], arguments, width, version=version)
    assert path.read_bytes() == b"keep"
    assert r.host.events == [] and r.host.buffers == {}


def test_buffer_result_is_portable_and_tool_schema_uses_common_contract(tmp_path):
    r = runtime(tmp_path)
    contract = r.contract("buffer.new")
    result = r.call_contract(contract["id"], {"x0":3}, 16)
    assert result["outputs"] == [{"hex":"000000"}]
    write = r.contract("file.write")
    result = r.call_contract(write["id"], {"x0":"new.bin", "x1":"00ff17"}, 16)
    assert result["outputs"] == [3] and (tmp_path / "new.bin").read_bytes().hex() == "00ff17"
    schema = tool_schema(write)["function"]
    assert len(schema["name"]) <= 64
    assert schema["parameters"]["properties"]["arguments"]["required"] == ["x0","x1"]


def test_tampered_saved_contract_is_rejected(tmp_path):
    r = runtime()
    path = tmp_path / "program.json"; r.save(path)
    data = json.loads(path.read_text()); data["contracts"][0]["parameters"][0]["type"] = "path"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contracts"):
        VectorRuntime.load(path)


def test_symlink_escape_and_bad_buffer_are_rejected_before_any_allocation(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    r = runtime(root)
    with pytest.raises(ValueError, match="escapes"):
        r.call_contract(r.contract("map")["id"], {"x0":"outside/file.bin", "x1":7}, 16)
    with pytest.raises(ValueError, match="hexadecimal"):
        r.call_contract(r.contract("file.write")["id"], {"x0":"valid.bin", "x1":"00 ff"}, 16)
    assert r.host.events == [] and r.host.buffers == {}


def test_catalog_uses_common_id_and_verified_execution_calls_shared_api(tmp_path, monkeypatch):
    from vectorpro.agent import AgentSession
    from vectorpro.semantic_catalog import TensorCatalog
    r = runtime(tmp_path)
    catalog = TensorCatalog(r, lambda texts: torch.ones((len(texts),4))/2, {"test":"constant"})
    contract = r.contract("xor")
    assert next(c for c in catalog.contracts if c["name"]=="xor")["id"] == contract["id"]
    session = AgentSession(r, None, catalog=catalog)
    evidence = {"query":"xor", "width":16, "operands":[[73,9]],
                "numeric_validation":{"width":4, "operands":[[1,3],[5,3]], "targets":[2,6]}}
    assert session.call("resolve_request", evidence)["status"] == "matched"
    original = r.call_contract
    seen=[]
    def spy(contract_id, arguments, width, **kwargs):
        seen.append((contract_id,arguments,width))
        return original(contract_id,arguments,width,**kwargs)
    monkeypatch.setattr(r,"call_contract",spy)
    result=session.call("execute_resolved",{})
    assert result["outputs"] == [64]
    assert seen == [(contract["id"],{"x0":73,"x1":9},16)]
    with pytest.raises(ValueError, match="matched"):
        session.call("execute_resolved",{})


def test_contract_cli_read_and_call(tmp_path, capsys):
    from vectorpro.__main__ import main
    program = tmp_path / "program.pt"; runtime().save(program)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"action":"contracts", "name":"xor"}))
    assert main(["--program",str(program),"--request",str(request)]) == 0
    contract = json.loads(capsys.readouterr().out)["contracts"][0]
    request.write_text(json.dumps({"action":"call_contract", "contract_id":contract["id"],
                                   "arguments":{"x0":73,"x1":9}, "width":16}))
    before = program.read_bytes()
    assert main(["--program",str(program),"--request",str(request)]) == 0
    assert json.loads(capsys.readouterr().out)["outputs"] == [64]
    assert program.read_bytes() == before  # direct execution does not rewrite the program
