import copy
import torch
from vectorpro.goal_memory import GoalMemory
from experiments.goal_condition_search import ConditionSearch


class Encoder:
    def __call__(self, texts):
        values = {'keep': [1., 0, 0], 'remove': [0., 1, 0], 'unsupported': [0., 0, 1],
                  'new-keep': [0.99, 0.1, 0], 'ambiguous': [1., 1, 0]}
        return torch.tensor([values[t] for t in texts])


def test_condition_search_uses_persisted_examples_not_llm_or_execution():
    examples = [{'intent': 'keep', 'rules': [{'kind': 'unchanged', 'parameter': 'source'}]},
                {'intent': 'remove', 'rules': [{'kind': 'absent', 'parameter': 'source'}]},
                {'intent': 'unsupported', 'rules': None}]
    memory = GoalMemory(examples, Encoder(), {'encoder': 'fixed'})
    before = copy.deepcopy(memory.to_data())
    search = ConditionSearch(memory)
    result = search.propose('new-keep')
    assert result['status'] == 'needs_goal_review'
    assert result['goal']['rules'] == examples[0]['rules']
    assert result['intent_independently_verified'] is False
    assert result['score_is_probability'] is False
    assert search.propose('unsupported')['status'] == 'needs_input'
    assert search.propose('ambiguous')['status'] == 'needs_input'
    assert memory.to_data() == before
