"""Explicit width constraints must reject swapped values before native execution."""
import json
import pytest
from vectorpro.agent import AgentSession, binding_tools, tool
from vectorpro.runtime import VectorRuntime


def schema(intent):
    offered = [tool("prepare_0", "Bind a number", {
        "width": {"type": "integer"}, "x0": {"type": "integer"}}, ("width", "x0"))]
    return binding_tools(offered, intent)[0]["function"]["parameters"]["properties"]


@pytest.mark.parametrize("intent", [
    "Fill target.bin with 29. Register width is 16 bits.",
    "Use a 16-bit register and value 29.",
    "값 29를 사용해. 레지스터 폭은 16비트야.",
    "폭 16비트, 값 29."])
def test_declared_width_does_not_use_operand(intent):
    fields = schema(intent)
    assert fields["width"]["enum"] == [16]
    assert 29 in fields["x0"]["enum"]


def test_conflicting_and_absent_widths_are_not_resolved_by_guessing():
    assert schema("width 8 or width 16, value 29")["width"]["enum"] == [8, 16, 29]
    assert schema("Use numbers 8 and 16")["width"]["enum"] == [8, 16]
    assert schema("Read data16-bit.bin, value 29")["width"]["enum"] == [29]


def test_width_restriction_is_enforced_when_model_ignores_tool_schema():
    class Model:
        def complete(self, messages, tools):
            if tools[0]["function"]["name"] == "search_goal":
                name, args = "search_goal", {"query": "test"}
            elif any(t["function"]["name"] == "prepare_0" for t in tools):
                name, args = "prepare_0", {"width": 29, "x0": 16}
            else:
                name, args = "ask_user", {"question": "Please clarify"}
            return {"role":"assistant", "tool_calls":[{"id":"call", "function":{"name":name,"arguments":json.dumps(args)}}]}
    # A minimal fake catalog exposes a proposal but no runtime execution path.
    class Catalog:
        def search_goal(self, *args, **kwargs):
            return [{"name":"opaque", "input_types":["value"], "argument_roles":["numeric parameter"], "description":"test", "score":1.0}]
    runtime = VectorRuntime()
    catalog = Catalog()
    catalog.runtime = runtime
    session = AgentSession(runtime, Model(), catalog=catalog, max_calls=3)
    response = session.run("Use value 29, width 16")
    assert any(e["result"].get("status") == "error" for e in response["tools"])
    assert session._proposed is None
