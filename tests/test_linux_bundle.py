import copy
import json
import subprocess
import sys
import pytest
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.runtime import VectorRuntime
from vectorpro import process_host, network_host, system_host
from vectorpro.learning.observations import validate
from experiments.linux_bundle import SOURCE, curriculum, verify, created_case, FixtureServer
from experiments.process_results import request_bytes


@pytest.fixture(scope="module")
def acquired(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    runtime = VectorRuntime.load(SOURCE, host=HostContext(root))
    with pytest.MonkeyPatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("acquisition performed native external operation")
        patch.setattr(subprocess, "Popen", forbidden)
        patch.setattr(network_host, "urlopen", forbidden)
        patch.setattr(system_host, "run", forbidden)
        outcomes = [runtime.teach_contract(**r) for r in curriculum()]
    assert all(o["status"] == "registered" for o in outcomes), outcomes
    assert not list(root.iterdir())
    runtime.save(root / "program.pt")
    loaded = VectorRuntime.load(root / "program.pt")
    assert loaded.contract("capture_to_file")["id"] == VectorRuntime.load(SOURCE).contract("capture_to_file")["id"]
    return loaded, outcomes


def test_acquired_reload_native_integration(acquired, tmp_path):
    assert len(verify(*acquired, tmp_path)) == 13


def test_pipeline_streams_concurrently(tmp_path):
    producer = "import sys,time,pathlib;sys.stdout.buffer.write(b'ready');sys.stdout.flush();p=pathlib.Path('ack');deadline=time.monotonic()+2\nwhile not p.exists():\n if time.monotonic()>deadline: raise RuntimeError('consumer was not concurrent')\n time.sleep(.01)\nsys.stdout.buffer.write(b'finished')"
    consumer = "import sys,pathlib;assert sys.stdin.buffer.read(5)==b'ready';pathlib.Path('ack').write_text('yes');sys.stdout.buffer.write(sys.stdin.buffer.read())"
    result = process_host.decode(process_host.pipeline(HostContext(tmp_path), request_bytes({
        "stages": [{"argv": [sys.executable, "-c", producer]}, {"argv": [sys.executable, "-c", consumer]}], "timeout": 3}), b""))
    assert result["codes"] == [0, 0]
    assert bytes.fromhex(result["stdout"]) == b"finished"


@pytest.mark.parametrize("mode", ["timeout", "missing", "limit"])
def test_pipeline_failure_reaps_all_children(tmp_path, monkeypatch, mode):
    original, launched = subprocess.Popen, []
    def observe(*args, **kwargs):
        proc = original(*args, **kwargs)
        launched.append(proc)
        return proc
    monkeypatch.setattr(subprocess, "Popen", observe)
    stages = [{"argv": [sys.executable, "-c", "import time;time.sleep(5)"]}, {"argv": ["/bin/cat"]}]
    error = TimeoutError
    if mode == "missing":
        stages[1] = {"argv": ["/missing/vectorpro"]}
        error = FileNotFoundError
    if mode == "limit":
        stages[0] = {"argv": [sys.executable, "-c", "import sys;sys.stdout.write('x'*10000)"]}
        error = ValueError
    with pytest.raises(error):
        process_host.pipeline(HostContext(tmp_path, max_buffer_bytes=1024), request_bytes({"stages": stages, "timeout": .2}), b"")
    assert launched and all(p.poll() is not None for p in launched)
    assert all(p.stdout is None or p.stdout.closed for p in launched)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("pipefail,code", [(True, 7), (False, 0)])
def test_pipeline_failure_policy(tmp_path, pipefail, code):
    data = request_bytes({"stages": [{"argv": [sys.executable, "-c", "import sys;sys.stderr.write('failure');sys.exit(7)"]}, {"argv": ["/bin/cat"]}], "pipefail": pipefail})
    result = process_host.decode(process_host.pipeline(HostContext(tmp_path), data, b""))
    assert result["code"] == code and result["codes"] == [7, 0]
    assert bytes.fromhex(result["stderr"]) == b"failure"


def test_http_binary_post_error_and_limit(tmp_path):
    with FixtureServer({"/big": (200, b"x" * 1000)}) as peer:
        host = HostContext(tmp_path)
        result = network_host.decode(network_host.run(host, request_bytes({"url": peer.url, "method": "POST", "body": "ff00"})))
        assert result == {"status": 201, "body": "ff00"}
        assert network_host.decode(network_host.run(host, request_bytes({"url": peer.url + "/missing"})))["status"] == 404
        with pytest.raises(ValueError, match="byte limit"):
            network_host.run(HostContext(tmp_path, max_buffer_bytes=128), request_bytes({"url": peer.url + "/big"}))


def test_system_types_and_unknown(tmp_path):
    for query in ("cpu_count", "pid"):
        assert type(json.loads(system_host.run(HostContext(tmp_path), query.encode()))) is int
    with pytest.raises(ValueError):
        system_host.run(HostContext(tmp_path), b"environment")


def test_process_observation_cannot_erase_prior_unwanted_mutation():
    fixture = created_case("folder/a", b"data")["processes"][0]
    validate([fixture], [])
    host = MemoryHostContext({"keep": b"\xaa", "unwanted": b"wrong"}, directories=[])
    with pytest.raises(ValueError, match="before state"):
        host._apply_observed_state(fixture)
    assert host.files["unwanted"] == b"wrong"
    bad = copy.deepcopy(fixture)
    del bad["before"]
    with pytest.raises(ValueError):
        validate([bad], [])
    bad = copy.deepcopy(fixture)
    bad["after_directories"] = [{}]
    with pytest.raises(ValueError):
        validate([bad], [])


def test_observation_requires_exact_query_and_no_native_fallback():
    host = MemoryHostContext({})
    host.observations = [{"operation": "system.info", "request": b"pid".hex(), "response": b"123".hex()}]
    with pytest.raises(ValueError, match="match"):
        host._system_info(b"cpu_count")
    assert getattr(host, "observation_index", 0) == 0
    assert host._system_info(b"pid") == b"123"
    with pytest.raises(ValueError, match="recorded"):
        host._system_info(b"pid")
