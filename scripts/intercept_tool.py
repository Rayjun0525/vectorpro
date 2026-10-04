"""Let a human/assistant intercept model tool calls without an external model API.

The assistant reads the real guide, decides the next tool/arguments, then runs
this bridge. It never fabricates HTTP/model replies. Each call loads and saves
the same program so subsequent calls reuse the actual acquired capabilities.
"""
import argparse
import json
from pathlib import Path

from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--arguments", type=Path, required=True)
    args = parser.parse_args()
    host = HostContext(args.host_root) if args.host_root else None
    runtime = VectorRuntime.load(args.program, host=host) if args.program.exists() else VectorRuntime(host=host)
    bridge = AgentSession(runtime, None, args.program)
    result = bridge.call(args.tool, json.loads(args.arguments.read_text(encoding="utf-8")))
    print(json.dumps({"mode": "assistant_tool_interception", "external_model_api": False,
                      "tool": args.tool, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
