"""Ollama transport envelope and sandbox route selection boundaries."""
from experiments.strong_goal_model import OllamaModel, execute_goal, fresh_cases
from experiments.assistant_intercept import intercept_goal
from pathlib import Path


def test_transport_preserves_prompt_and_does_not_send_labels():
    model = OllamaModel('http://unused', 'gemma4:12b')
    requests = []
    def request(path, payload):
        requests.append((path, payload))
        return {'message': {'content': '{"name":"interpret_goal","arguments":{"decision":"unsupported"}}'}}
    model.request = request
    messages = [{'role': 'system', 'content': 'Draft only'}, {'role': 'user', 'content': 'Encrypt source'}]
    tools = [{'function': {'name': 'interpret_goal', 'parameters': {'type': 'object'}}}]
    reply = model.complete(messages, tools)
    assert reply['tool_calls'][0]['function']['arguments'] == {'decision': 'unsupported'}
    assert messages[0]['content'] == 'Draft only'
    assert requests[0][1]['messages'][1] == messages[1]
    assert requests[0][1]['options']['num_predict'] == 600
    assert 'expected_rules' not in str(requests)


def test_invalid_transport_reply_is_not_a_tool_call():
    model = OllamaModel('http://unused', 'model')
    model.request = lambda *args: {'message': {'content': 'incomplete JSON'}}
    reply = model.complete([{'role': 'system', 'content': 'Draft'}], [])
    assert 'tool_calls' not in reply


def test_missing_contract_does_not_create_native_root(tmp_path):
    program = Path('results/goal_router_execution_replay/program.pt')
    root = tmp_path / 'native'
    goal = intercept_goal('keep everything', {'decision': 'supported',
        'states': {'source': 'unchanged', 'destination': 'unchanged'}})
    result = execute_goal(program, goal, root)
    assert result['status'] == 'no_unique_stored_contract'
    assert result['observations'] == []
    assert not root.exists()
    assert len(fresh_cases()) == 16
