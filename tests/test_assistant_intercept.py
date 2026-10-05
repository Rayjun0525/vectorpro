from experiments.assistant_intercept import intercept_goal, observation, compact_model_goal
from vectorpro.goal_evidence import check_goal
from vectorpro.host import MemoryHostContext
import pytest


def test_interceptor_keeps_reverse_roles_and_frame_explicit():
    goal = intercept_goal('reverse', {'decision': 'supported', 'states': {'source': 'initial:destination', 'destination': 'absent'}})
    host = MemoryHostContext({'source.bin': b'old destination', 'keep': b'keep'}, directories=['empty'])
    files = {'source.bin': b'old source', 'target.bin': b'old destination', 'keep': b'keep'}
    record = observation('reverse', ['source', 'destination'], ['source.bin', 'target.bin'], files, ['empty'], host)
    assert check_goal(goal, record) == 1
    host.files['keep'] = b'changed'
    with pytest.raises(ValueError, match='contradict'):
        check_goal(goal, observation('reverse', ['source', 'destination'], ['source.bin', 'target.bin'], files, ['empty'], host))


@pytest.mark.parametrize('decision', ['unsupported', 'contradictory', 'unclear'])
def test_declined_intercept_has_no_executable_goal(decision):
    assert intercept_goal('request', {'decision': decision}) is None


def test_missing_role_rejected():
    with pytest.raises(ValueError):
        intercept_goal('request', {'decision': 'supported', 'states': {'source': 'unchanged'}})


@pytest.mark.parametrize('args,status', [
    ({'decision': 'supported', 'states': {'source': 'unchanged', 'destination': 'initial:source'}}, 'needs_goal_review'),
    ({'decision': 'unsupported'}, 'needs_input'),
    ({'decision': 'supported'}, 'goal_draft_failed'),
    ({'decision': 'invalid'}, 'goal_draft_failed'),
])
def test_compact_adapter_requires_complete_supported_interpretation(args, status):
    class Model:
        def complete(self, messages, tools):
            assert [t['function']['name'] for t in tools] == ['interpret_goal']
            return {'role': 'assistant', 'tool_calls': [{'function': {'name': 'interpret_goal', 'arguments': args}}]}
    assert compact_model_goal(Model(), 'request')['status'] == status
