"""Execute/teach a saved vector program using a JSON request, on any supported OS."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateLesson
from vectorpro.runtime import VectorRuntime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True, help="one saved vector program file")
    parser.add_argument("--request", type=Path, required=True, help="JSON inputs and optional lesson")
    parser.add_argument("--host-root", type=Path, help="root for byte-buffer and file operations")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.request.read_text(encoding="utf-8"))
        plan = LearningPlan.from_dict(data["plan"]) if "plan" in data else None
        lesson = ExampleLesson.from_dict(data["lesson"]) if "lesson" in data else None
        state_lesson = StateLesson.from_dict(data["state_lesson"]) if "state_lesson" in data else None
        host = HostContext(args.host_root) if args.host_root is not None else None
        torch.manual_seed(args.seed)
        runtime = (VectorRuntime.load(args.program, host=host) if args.program.exists()
                   else VectorRuntime(host=host, seed=args.seed))
        if host is not None:
            runtime.provide_host_operations()
        action = data.get("action")
        if action in ("contracts", "call_contract"):
            if set(data) - ({"action", "name"} if action == "contracts" else
                            {"action", "contract_id", "arguments", "width", "version"}):
                raise ValueError("contract requests cannot include legacy execution or teaching fields")
            if action == "contracts":
                response = {"status": "contracts", "contracts": [runtime.contract(data["name"])]
                            if "name" in data else runtime.contracts()}
            else:
                response = runtime.call_contract(data["contract_id"], data["arguments"], data["width"],
                                                 version=data.get("version", 1))
            print(json.dumps(response, ensure_ascii=False))
            return 0
        # Byte inputs allocate fresh transient handles; raw handles are not portable.
        operands = []
        for row in data["operands"]:
            converted = []
            for value in row:
                if isinstance(value, dict):
                    if host is None:
                        raise ValueError("buffer inputs require --host-root")
                    if set(value) == {"utf8"}:
                        value = host.put(value["utf8"].encode("utf-8"))
                    elif set(value) == {"hex"}:
                        value = host.put(bytes.fromhex(value["hex"]))
                    else:
                        raise ValueError("buffer inputs must specify utf8 or hex")
                converted.append(value)
            operands.append(tuple(converted))
        result = runtime.request(data["name"], operands, data["width"], plan=plan, lesson=lesson,
                                 state_lesson=state_lesson)
        response = asdict(result)
        if host is not None:
            response["effects"] = host.events
            # Report bytes for explicitly requested buffer results only.
            if data.get("output_format") == "hex" and result.outputs is not None:
                response["output_buffers_hex"] = [bytes(host.buffers[h]).hex() for h in result.outputs]
        if result.status in ("executed", "learned_and_executed"):
            runtime.save(args.program)
        print(json.dumps(response, ensure_ascii=False))
        return 0 if result.outputs is not None else 2
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
