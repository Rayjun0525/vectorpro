import copy
import json
import pytest
from experiments.goal_questions import question, answer, teacher_answer, draft

COPY = [{'kind': 'unchanged', 'parameter': 'source'}, {'kind': 'equals_initial', 'parameter': 'destination', 'source': 'source'}, {'kind': 'unchanged_except', 'parameters': ['destination']}]
MOVE = [{'kind': 'absent', 'parameter': 'source'}, {'kind': 'equals_initial', 'parameter': 'destination', 'source': 'source'}, {'kind': 'unchanged_except', 'parameters': ['source', 'destination']}]


def test_one_difference_selects_goal_without_authorizing():
    candidates = [COPY, MOVE]
    q = question(candidates, 'request')
    for expected in candidates:
        selected = answer(candidates, q, teacher_answer(expected, q), q['question_sha256'])
        result = draft('request', selected)
        assert result['status'] == 'needs_goal_review'
        assert result['goal']['rules'] == expected
        assert result['intent_independently_verified'] is False
        wire = json.loads(json.dumps(q))
        assert answer(candidates, wire, teacher_answer(expected, wire), wire['question_sha256']) == selected


@pytest.mark.parametrize('choice', ['outside', 'unclear'])
def test_missing_or_unsupported_answer_cannot_create_goal(choice):
    q = question([COPY, MOVE], 'request')
    assert draft('request', answer([COPY, MOVE], q, choice, q['question_sha256']))['status'] == 'needs_input'


@pytest.mark.parametrize('change', ['hash', 'option', 'candidate', 'intent'])
def test_stale_and_tampered_questions_rejected(change):
    candidates = copy.deepcopy([COPY, MOVE])
    q = question(candidates, 'request')
    token = q['question_sha256']
    if change == 'hash':
        token = 'invalid'
    elif change == 'option':
        q['options'][0]['value'] = 'tampered'
    elif change == 'candidate':
        candidates.reverse()
    else:
        q['intent'] = 'other request'
    with pytest.raises(ValueError):
        answer(candidates, q, 'option_0', token)
