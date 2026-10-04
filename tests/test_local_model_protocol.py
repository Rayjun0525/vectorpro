import json

from experiments.local_small_llm import gemma_messages, native_message
from vectorpro.agent import binding_tools


def test_gemma_framing_preserves_bound_inputs_and_separate_evidence():
    evidence = '{"width":4,"operands":[[1,3]],"targets":[2]}'
    messages = [
        {"role": "system", "content": "Keep requested inputs fixed."},
        {"role": "user", "content": "Compute XOR of 12345 and 4567."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {
            "name": "prepare_0", "arguments": '{"width":16,"x0":12345,"x1":4567}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"status":"proposed"}'},
        {"role": "user", "content": "Validation only: " + evidence}]
    original = json.dumps(messages)
    adapted = gemma_messages(messages, [{"type": "function", "function": {"name": "verify_numeric"}}])
    assert [m["role"] for m in adapted] == ["user", "assistant", "user"]
    assert "Keep requested inputs fixed." in adapted[0]["content"]
    assert "verify_numeric" in adapted[0]["content"]
    assert native_message(adapted[1]["content"])["tool_calls"][0]["function"]["arguments"] == json.dumps({"width": 16, "x0": 12345, "x1": 4567})
    assert 'TOOL_RESULT c1' in adapted[2]["content"] and evidence in adapted[2]["content"]
    assert json.dumps(messages) == original


def test_gemma_plain_reply_and_tool_result_do_not_invent_calls():
    assert native_message("I need a target output.") == {"role": "assistant", "content": "I need a target output."}
    output = '<tool_call>{"name":"ask_user","arguments":{"question":"목표는?"}}</tool_call>'
    call = native_message(output)["tool_calls"][0]["function"]
    assert call["name"] == "ask_user" and json.loads(call["arguments"]) == {"question": "목표는?"}


def test_binding_schema_uses_request_literals_without_computing_answers():
    tools = [{"function":{"name":"prepare_0","parameters":{"properties":{
        "width":{"type":"integer"},"x0":{"type":"integer"},"x1":{"type":"integer"}}}}}]
    revised = binding_tools(tools, "Take 37 away from 313 at width 16")
    fields = revised[0]["function"]["parameters"]["properties"]
    assert fields["x0"]["enum"] == [37,313,16]
    assert 276 not in fields["x0"]["enum"] and 0 not in fields["x1"]["enum"]
    assert fields["width"]["enum"] == [16]
    assert "enum" not in tools[0]["function"]["parameters"]["properties"]["x0"]
