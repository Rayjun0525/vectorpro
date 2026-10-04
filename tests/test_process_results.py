import copy
import subprocess
import sys
import pytest
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.runtime import VectorRuntime
from vectorpro.process_host import decode
from experiments.process_results import request_bytes, lesson_request, verify


def invoke(host, spec, data=b""):
    handle = host.invoke("process.run", (host.put(request_bytes(spec)), host.put(data)), 16)
    return decode(bytes(host.buffers[handle])), handle


def test_native_stdin_stdout_stderr_exit_cwd_environment(tmp_path):
    (tmp_path / "work").mkdir()
    host = HostContext(tmp_path)
    script = "import os,sys;sys.stdout.buffer.write(sys.stdin.buffer.read());sys.stderr.write(os.environ['DEMO']+':'+os.path.basename(os.getcwd()));sys.exit(7)"
    result, handle = invoke(host, {"argv": [sys.executable, "-c", script], "cwd": "work", "env": {"DEMO": "ok"}}, b"\xff\0data")
    assert bytes.fromhex(result["stdout"]) == b"\xff\0data"
    assert bytes.fromhex(result["stderr"]) == b"ok:work"
    assert host.invoke("process.code", (handle,), 16) == 7
    for operation, expected in (("process.stdout", b"\xff\0data"), ("process.stderr", b"ok:work")):
        extracted = host.invoke(operation, (handle,), 16)
        assert bytes(host.buffers[extracted]) == expected
    actual, _ = invoke(host, {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}, b"abc\0more\xff")
    assert bytes.fromhex(actual["stdout"]) == b"ABC\0MORE\xff"


def test_timeout_output_limits_missing_executable_and_signal(tmp_path):
    host = HostContext(tmp_path)
    with pytest.raises(TimeoutError):
        invoke(host, {"argv": [sys.executable, "-c", "import time;time.sleep(2)"], "timeout": 0.05})
    with pytest.raises(FileNotFoundError):
        invoke(host, {"argv": ["/missing/vectorpro-executable"]})
    limited = HostContext(tmp_path, max_buffer_bytes=512)
    with pytest.raises(ValueError, match="byte limit"):
        invoke(limited, {"argv": [sys.executable, "-c", "import sys;sys.stdout.write('x'*10000)"]})
    result, handle = invoke(host, {"argv": [sys.executable, "-c", "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"]})
    assert result["code"] == -15 and host.invoke("process.code", (handle,), 16) == 143


@pytest.mark.parametrize("spec", [{"argv": []}, {"argv": ["cat"], "shell": True}, {"argv": ["cat"], "timeout": 0},
    {"argv": ["cat"], "cwd": "../escape"}, {"argv": ["cat"], "env": {"BAD=NAME": "x"}}])
def test_bad_specs_do_not_launch(tmp_path, monkeypatch, spec):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid specification launched a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        invoke(HostContext(tmp_path), spec)


def test_memory_never_launches_and_requires_exact_recorded_call(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("learning launched a process")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    host = MemoryHostContext({})
    with pytest.raises(ValueError, match="recorded evidence"):
        invoke(host, {"argv": ["/bin/cat"]})
    case = lesson_request()["state_lesson"]["training"][0]
    host.processes = case["processes"]
    with pytest.raises(ValueError, match="match"):
        invoke(host, {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}, b"different")
    result, _ = invoke(host, {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}, b"abc")
    assert result["stdout"] == b"ABC".hex()
    with pytest.raises(ValueError, match="recorded evidence"):
        invoke(host, {"argv": ["/usr/bin/tr", "a-z", "A-Z"]}, b"abc")


def test_acquisition_without_launch_reload_native(tmp_path, monkeypatch):
    runtime = VectorRuntime.load("results/record_results_verified/program.pt", host=HostContext(tmp_path))
    with monkeypatch.context() as m:
        def forbidden(*args, **kwargs):
            pytest.fail("acquisition launched native process")
        m.setattr(subprocess, "Popen", forbidden)
        outcome = runtime.teach_contract(**lesson_request())
    assert outcome["status"] == "registered", outcome
    assert not list(tmp_path.iterdir()) and not runtime.host.events
    runtime.save(tmp_path / "program.pt")
    loaded = VectorRuntime.load(tmp_path / "program.pt", host=HostContext(tmp_path))
    assert len(verify(loaded, outcome["contract"], tmp_path)) == 4


def test_wrong_recorded_validation_does_not_register(tmp_path):
    runtime = VectorRuntime.load("results/record_results_verified/program.pt", host=HostContext(tmp_path))
    initial = runtime.contracts()
    request = copy.deepcopy(lesson_request())
    request["state_lesson"]["validation"][0]["processes"][0]["stdout"] = b"wrong".hex()
    result = runtime.teach_contract(**request)
    assert result["status"] == "learning_failed" and runtime.contracts() == initial
    assert not list(tmp_path.iterdir()) and not runtime.host.events
