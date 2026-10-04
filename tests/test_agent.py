import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest
import torch

from vectorpro.agent import AgentSession, HTTPChatModel
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def call(name, args, id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(json.loads(json.dumps(messages)))
        assert {tool["function"]["name"] for tool in tools} >= {"teach", "execute", "ask_user"}
        return next(self.replies)


def test_model_adapter_teaches_executes_saves_then_runs_without_model(tmp_path):
    torch.manual_seed(0)
    data = json.loads((Path(__file__).resolve().parents[1] / "experiments/requests/learn_xor.json").read_text())
    teaching = {key: data[key] for key in ("name", "plan", "lesson")}
    model = ScriptedModel([call("list_capabilities", {}), call("teach", {"request": teaching}),
                           call("execute", {"name": "xor", "width": 16, "operands": [[123, 45]]}),
                           {"role": "assistant", "content": "Executed the acquired function."}])
    runtime = VectorRuntime()
    session = AgentSession(runtime, model, tmp_path / "program.json")
    result = session.run("Learn exclusive-or and compute it for 123 and 45.")
    assert result["status"] == "answered"
    assert result["tools"][1]["result"]["status"] == "learned"
    assert result["tools"][2]["result"]["outputs"] == [86]
    assert model.requests[-1][-1]["role"] == "tool"
    restored = VectorRuntime.load(tmp_path / "program.json")
    assert restored.request("xor", [(12345, 4567)], 16).outputs == [12345 ^ 4567]
    assert restored.registry.get("xor").provenance["evidence_source"] == "LLM adapter supplied examples"


def test_state_teaching_isolated_from_native_execution_and_clarification(tmp_path):
    (tmp_path / "native").write_bytes(b"actual")
    runtime = VectorRuntime(host=HostContext(tmp_path))
    data = json.loads((Path(__file__).resolve().parents[1] / "experiments/requests/learn_transfer.json").read_text())
    session = AgentSession(runtime, ScriptedModel([]))
    taught = session.call("teach", {"request": {"name": data["name"], "state_lesson": data["state_lesson"]}})
    assert taught["status"] == "learned"
    assert not (tmp_path / "out").exists()
    result = session.call("execute", {"name": data["name"], "width": 16,
                                      "operands": [[{"utf8": "native"}, {"utf8": "out"}]]})
    assert result["outputs"] == [6] and (tmp_path / "out").read_bytes() == b"actual"
    session.model = ScriptedModel([call("ask_user", {"question": "Which output should be correct?"})])
    assert session.run("Do something with the file")["status"] == "needs_input"


def test_failed_or_invalid_teaching_cannot_supply_code_or_modify_saved_program(tmp_path):
    runtime = VectorRuntime()
    path = tmp_path / "program.json"
    runtime.save(path)
    previous = path.read_bytes()
    session = AgentSession(runtime, ScriptedModel([]), path)
    with pytest.raises(ValueError, match="only plans and examples"):
        session.call("teach", {"request": {"name": "bad", "program": "arbitrary code"}})
    assert path.read_bytes() == previous and "bad" not in runtime.registry
    session.model = ScriptedModel([call("unknown", {})] * 2)
    session.max_calls = 2
    result = session.run("Unknown tool")
    assert result["status"] == "call_budget" and all(e["result"]["status"] == "error" for e in result["tools"])


def test_actual_http_chat_transport_returns_tool_result_in_next_request():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            assert self.headers["Authorization"] == "Bearer test-key"
            message = (call("list_capabilities", {}) if len(seen) == 1
                       else {"role": "assistant", "content": "No learned functions yet."})
            payload = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = HTTPChatModel(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "test-model", "test-key")
        result = AgentSession(VectorRuntime(), model).run("What can you do?")
        assert result["status"] == "answered"
        assert seen[1]["messages"][-1]["tool_call_id"] == "call_1"
        assert json.loads(seen[1]["messages"][-1]["content"]) == []
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
