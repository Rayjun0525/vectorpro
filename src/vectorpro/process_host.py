"""Provided process semantics; state learning uses exact recorded responses only."""
import json
import os
import signal
import subprocess
import tempfile
import time


def specification(data, root):
    from vectorpro.host import MemoryHostContext
    spec = json.loads(data)
    if not isinstance(spec, dict) or set(spec) - {"argv", "cwd", "env", "timeout"}:
        raise ValueError("process request accepts argv/cwd/env/timeout")
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or "\0" in a for a in argv) or not argv[0]:
        raise ValueError("process argv requires nonempty executable and string arguments")
    timeout = spec.get("timeout", 5)
    if type(timeout) not in (int, float) or not 0 < timeout <= 30:
        raise ValueError("process timeout must be in (0,30]")
    env = spec.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not k or "=" in k or "\0" in k or not isinstance(v, str) or "\0" in v for k, v in env.items()):
        raise ValueError("process env requires string names and values")
    cwd = spec.get("cwd")
    if cwd is not None:
        cwd = MemoryHostContext.normalize(cwd)
    if root is not None:
        path = (root / cwd).resolve() if cwd else root
        if not path.is_relative_to(root):
            raise ValueError("process cwd escapes host root")
    else:
        path = cwd
    return argv, path, {**os.environ, **env}, timeout


def encode(stdout, stderr, code):
    return json.dumps({"stdout": stdout.hex(), "stderr": stderr.hex(), "code": code},
                      separators=(",", ":"), sort_keys=True).encode()


def decode(data):
    result = json.loads(data)
    if not isinstance(result, dict) or set(result) - {"stdout", "stderr", "code", "codes"} or not {"stdout", "stderr", "code"} <= set(result) or type(result["code"]) is not int:
        raise ValueError("invalid process result")
    if "codes" in result and (not isinstance(result["codes"], list) or not result["codes"] or any(type(c) is not int for c in result["codes"])):
        raise ValueError("invalid pipeline exit codes")
    for key in ("stdout", "stderr"):
        value = result[key]
        if not isinstance(value, str) or len(value) % 2 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError("process streams require byte hex")
    return result


def run(host, request, stdin):
    argv, cwd, env, timeout = specification(request, host.root)
    # Capture in temporary files to avoid retaining unbounded PIPE output in RAM.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                stdout=stdout, stderr=stderr, start_new_session=os.name == "posix")
        deadline = time.monotonic() + timeout
        pending = stdin
        try:
            while True:
                if stdout.tell() + stderr.tell() > host.max_buffer_bytes // 2:
                    raise ValueError("process output exceeds byte limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("process timeout")
                try:
                    proc.communicate(pending, timeout=min(0.02, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pending = None
            if stdout.tell() + stderr.tell() > host.max_buffer_bytes // 2:
                raise ValueError("process output exceeds byte limit")
            stdout.seek(0)
            stderr.seek(0)
            return encode(stdout.read(), stderr.read(), proc.returncode)
        finally:
            # Kill the POSIX process group even if its parent exited first.
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.kill()
            proc.wait()
            if proc.stdin:
                proc.stdin.close()


def pipeline_specification(request, root):
    spec = json.loads(request)
    if not isinstance(spec, dict) or set(spec) - {"stages", "timeout", "pipefail"}:
        raise ValueError("pipeline accepts stages/timeout/pipefail")
    stages = spec.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 8:
        raise ValueError("pipeline needs 1..8 process stages")
    timeout = spec.get("timeout", 5)
    if type(timeout) not in (int, float) or not 0 < timeout <= 30 or type(spec.get("pipefail", True)) is not bool:
        raise ValueError("invalid pipeline timeout or pipefail")
    configs = [specification(json.dumps(stage).encode(), root) for stage in stages]
    return configs, min(timeout, *(config[3] for config in configs)), spec.get("pipefail", True)


def pipeline(host, request, stdin):
    from contextlib import ExitStack
    configs, timeout, pipefail = pipeline_specification(request, host.root)
    procs = []
    with ExitStack() as resources:
        source = resources.enter_context(tempfile.TemporaryFile())
        source.write(stdin)
        source.seek(0)
        output = resources.enter_context(tempfile.TemporaryFile())
        errors = [resources.enter_context(tempfile.TemporaryFile()) for _ in configs]
        deadline = time.monotonic() + timeout
        previous = source
        try:
            for i, (argv, cwd, env, _) in enumerate(configs):
                proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=previous,
                    stdout=output if i == len(configs) - 1 else subprocess.PIPE,
                    stderr=errors[i], start_new_session=os.name == "posix")
                procs.append(proc)
                if previous is not source:
                    previous.close()
                previous = proc.stdout
            while True:
                if output.tell() + sum(e.tell() for e in errors) > host.max_buffer_bytes // 2:
                    raise ValueError("pipeline output exceeds byte limit")
                if all(proc.poll() is not None for proc in procs):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("pipeline timeout")
                time.sleep(0.01)
            codes = [proc.returncode for proc in procs]
            code = next((c for c in codes if c != 0), codes[-1]) if pipefail else codes[-1]
            output.seek(0)
            diagnostic = b""
            for error in errors:
                error.seek(0)
                diagnostic += error.read()
            result = decode(encode(output.read(), diagnostic, code))
            result["codes"] = codes
            return json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        finally:
            for proc in procs:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif proc.poll() is None:
                    proc.kill()
            for proc in procs:
                proc.wait()
                if proc.stdout:
                    proc.stdout.close()
