"""Small portable system-query semantics, returned as JSON bytes."""
import json
import os
import platform


def validate(query):
    name = query.decode("utf-8")
    if name not in ("platform", "machine", "cpu_count", "cwd", "pid"):
        raise ValueError("unknown system query")
    return name


def run(host, query):
    name = validate(query)
    values = {"platform": platform.system, "machine": platform.machine,
              "cpu_count": os.cpu_count, "cwd": lambda: str(host.root), "pid": os.getpid}
    return json.dumps(values[name](), ensure_ascii=False).encode()
