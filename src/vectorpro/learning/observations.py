"""Validate data-only external observations and full process filesystem states."""
import json
from vectorpro.host import MemoryHostContext


def hexadecimal(value):
    if not isinstance(value, str) or len(value) % 2 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("observations require hexadecimal byte pairs")
    return bytes.fromhex(value)


def snapshot(fixture):
    if not {"before", "before_directories", "after", "after_directories"} & set(fixture):
        return
    if not {"before", "before_directories", "after", "after_directories"} <= set(fixture):
        raise ValueError("process state requires full files AND directories")
    for phase in ("before", "after"):
        files, directories = fixture[phase], fixture[phase + "_directories"]
        if not isinstance(files, dict) or not isinstance(directories, list) or any(not isinstance(p, str) for p in [*files, *directories]):
            raise ValueError("observed state requires string paths")
        decoded = {p: hexadecimal(v) for p, v in files.items()}
        if len(files) > 32 or len(directories) > 32 or sum(map(len, decoded.values())) > 65536:
            raise ValueError("observed state exceeds snapshot bounds")
        if len(set(directories)) != len(directories) or any(MemoryHostContext.normalize(p) != p for p in [*files, *directories]):
            raise ValueError("invalid observed state paths")
        MemoryHostContext.directory_state(decoded, directories)
    files, directories = fixture["after"], fixture["after_directories"]
    if not isinstance(files, dict) or len(files) > 32 or not isinstance(directories, list) or len(directories) > 32 or len(set(directories)) != len(directories):
        raise ValueError("observed state exceeds snapshot bounds")
    decoded = {p: hexadecimal(v) for p, v in files.items()}
    if sum(len(v) for v in decoded.values()) > 65536 or any(MemoryHostContext.normalize(p) != p for p in [*files, *directories]):
        raise ValueError("invalid observed state paths or byte limit")
    MemoryHostContext.directory_state(decoded, directories)


def validate(processes, observations):
    from vectorpro.process_host import specification, pipeline_specification, decode, encode
    from vectorpro.network_host import specification as http_specification, decode as http_decode
    from vectorpro.system_host import validate as system_query
    if not isinstance(processes, (list, tuple)) or not isinstance(observations, (list, tuple)) or len(processes) + len(observations) > 16:
        raise ValueError("at most 16 external observations per case")
    byte_count = 0
    for fixture in processes:
        required = {"request", "stdin", "stdout", "stderr", "code"}
        if not isinstance(fixture, dict) or not required <= set(fixture) or set(fixture) - required - {"operation", "before", "before_directories", "after", "after_directories"}:
            raise ValueError("process evidence requires request/stdin/stdout/stderr/code")
        for key in ("request", "stdin", "stdout", "stderr"):
            byte_count += len(hexadecimal(fixture[key]))
        operation = fixture.get("operation", "process.run")
        if operation not in ("process.run", "process.pipeline"):
            raise ValueError("unknown recorded process operation")
        (specification if operation == "process.run" else pipeline_specification)(hexadecimal(fixture["request"]), None)
        decode(encode(hexadecimal(fixture["stdout"]), hexadecimal(fixture["stderr"]), fixture["code"]))
        snapshot(fixture)
    for fixture in observations:
        if not isinstance(fixture, dict) or set(fixture) != {"operation", "request", "response"}:
            raise ValueError("observation requires operation/request/response")
        request, response = hexadecimal(fixture["request"]), hexadecimal(fixture["response"])
        byte_count += len(request) + len(response)
        if fixture["operation"] == "network.http":
            http_specification(request)
            http_decode(response)
        elif fixture["operation"] == "system.info":
            name = system_query(request)
            value = json.loads(response)
            if (name in ("platform", "machine", "cwd") and not isinstance(value, str)) or (name in ("cpu_count", "pid") and (type(value) is not int or value < 1)):
                raise ValueError("system observation has wrong type")
        else:
            raise ValueError("unknown external observation")
    if byte_count > 65536:
        raise ValueError("external observations exceed evidence byte limit")
