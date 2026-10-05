import copy
import json
import pytest

from experiments.reference_acquisition import manifests
from vectorpro.goal_draft import accept_goal, propose_goal
from vectorpro.goal_evidence import check_goal
from vectorpro.goal_interpretation import goal_from_states, normalize_goal
from vectorpro.reference_evidence import ReferenceProviders


class Model:
    def __init__(self, states=None, **issues):
        self.args = {'states': states or {'source': 'unchanged', 'destination': 'initial:source'},
                     'unrepresented_effects': [], 'unresolved': [], **issues}

    def complete(self, messages, tools):
        assert [t['function']['name'] for t in tools] == ['describe_goal']
        assert 'supported' not in tools[0]['function']['parameters']['properties']
        return {'role': 'assistant', 'tool_calls': [{'function': {'name': 'describe_goal', 'arguments': self.args}}]}


def record(before, after, before_dirs=None, after_dirs=None):
    return {'intent': 'request', 'interface': {'parameters': [{'name': 'source'}, {'name': 'destination'}]},
            'state_lesson': {'training': [{'inputs': ['s', 'd'], 'before': before, 'after': after,
                'before_directories': before_dirs or [], 'after_directories': after_dirs or []}], 'validation': []},
            'heldout': []}


def test_compact_frame_preserves_extra_files_and_empty_directories():
    draft = propose_goal(Model(), ReferenceProviders(manifests()), 'request', encoding='compact')
    goal = accept_goal(draft, draft['proposal_sha256'])
    evidence = record({'s': 'aa', 'd': 'bb', 'keep': 'cc'}, {'s': 'aa', 'd': 'aa', 'keep': 'cc'}, ['empty'], ['empty'])
    assert check_goal(goal, evidence) == 1
    for mutation in ('file', 'directory'):
        changed = copy.deepcopy(evidence)
        case = changed['state_lesson']['training'][0]
        if mutation == 'file':
            case['after']['keep'] = 'ff'
        else:
            case['after_directories'] = []
        with pytest.raises(ValueError, match='contradict'):
            check_goal(goal, changed)


def test_noop_and_reverse_do_not_require_model_support_decision():
    providers = ReferenceProviders(manifests())
    for states in ({'source': 'unchanged', 'destination': 'unchanged'},
                   {'source': 'initial:destination', 'destination': 'absent'}):
        draft = propose_goal(Model(states), providers, 'request', encoding='compact')
        assert draft['status'] == 'needs_goal_review'
        assert accept_goal(draft, draft['proposal_sha256']) == draft['goal']


@pytest.mark.parametrize('issues,reason', [
    ({'unrepresented_effects': ['compression']}, 'goal_outside_representation'),
    ({'unresolved': ['source must both remain and disappear']}, 'goal_needs_clarification')])
def test_unrepresented_or_conflicting_effects_never_become_executable_goal(issues, reason):
    result = propose_goal(Model(**issues), ReferenceProviders(manifests()), 'request', encoding='compact')
    assert result['status'] == 'needs_input'
    assert result['reason'] == reason
    assert 'goal' not in result


@pytest.mark.parametrize('args', [
    {'states': {'source': 'unchanged'}},
    {'states': {'source': 'unchanged', 'destination': 'initial:unknown'}},
    {'states': {'source': 'initial:source', 'destination': 'unchanged'}},
    {'unresolved': 'not an array'},
    {'unrepresented_effects': ['x' * 301]},
    {'unrepresented_effects': ['copying']},
    {'extra': 'must not be accepted'}])
def test_malformed_semantics_fail_before_goal_creation(args):
    result = propose_goal(Model(**args), ReferenceProviders(manifests()), 'request', encoding='compact')
    assert result['status'] == 'goal_draft_failed'


def test_unspecified_effect_requires_input_without_inventing_a_plan():
    result = propose_goal(Model({'source': 'unspecified', 'destination': 'unspecified'}),
                          ReferenceProviders(manifests()), 'request', encoding='compact')
    assert result['status'] == 'needs_input'
    assert 'goal' not in result


def test_self_equality_is_not_silently_weakened_for_directories():
    goal = goal_from_states('request', ['source', 'destination'],
                            {'source': 'initial:source', 'destination': 'unchanged'})
    assert any(r['kind'] == 'equals_initial' for r in goal['rules'])
    evidence = record({'d': 'aa'}, {'d': 'aa'}, ['s'], ['s'])
    with pytest.raises(ValueError, match='contradict'):
        check_goal(goal, evidence)
    # Only a caller-provided initial file domain justifies this comparison rewrite.
    normalized = normalize_goal(goal, file_parameters=['source'])
    assert all(r['kind'] != 'equals_initial' for r in normalized['rules'])
    assert check_goal(normalized, evidence) == 1
    assert normalize_goal(goal) == goal


def test_redundant_frame_exceptions_normalize_without_changing_preservation():
    goal = {'intent': 'request', 'rules': [{'kind': 'unchanged', 'parameter': 'source'},
        {'kind': 'unchanged', 'parameter': 'destination'},
        {'kind': 'unchanged_except', 'parameters': ['destination', 'source']}]}
    normalized = normalize_goal(goal)
    assert next(r for r in normalized['rules'] if r['kind'] == 'unchanged_except')['parameters'] == []
    for after in ({'s': 'aa', 'd': 'bb'}, {'s': 'ff', 'd': 'bb'}):
        evidence = record({'s': 'aa', 'd': 'bb'}, after)
        outcomes = []
        for candidate in (goal, normalized):
            try:
                check_goal(candidate, evidence)
                outcomes.append(True)
            except ValueError:
                outcomes.append(False)
        assert outcomes[0] == outcomes[1]


def test_cli_semantics_option_does_not_change_program_or_files(tmp_path, monkeypatch, capsys):
    from vectorpro.agent import main
    from vectorpro.runtime import VectorRuntime
    providers = tmp_path / 'providers.json'
    providers.write_text(json.dumps(manifests()))
    program = tmp_path / 'program.pt'
    VectorRuntime().save(program)
    original = program.read_bytes()
    host = tmp_path / 'host'
    host.mkdir()
    (host / 'source.bin').write_bytes(b'caller file')
    monkeypatch.setattr('vectorpro.agent.HTTPChatModel', lambda *args: Model())
    monkeypatch.setattr(ReferenceProviders, 'collect', lambda *args: pytest.fail('draft collected evidence'))
    assert main(['--program', str(program), '--host-root', str(host),
        '--endpoint', 'http://unused', '--model', 'fixture', '--intent', 'request',
        '--draft-goal', '--goal-encoding', 'compact', '--reference-providers', str(providers)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'needs_goal_review'
    assert accept_goal(result, result['proposal_sha256']) == result['goal']
    assert program.read_bytes() == original
    assert {p.name: p.read_bytes() for p in host.iterdir()} == {'source.bin': b'caller file'}


@pytest.mark.parametrize('fenced', [False, True])
def test_exact_json_content_is_validated_as_draft_data_only(fenced):
    class ContentModel:
        def complete(self, messages, tools):
            content = json.dumps(Model().args)
            if fenced:
                content = '```json\n' + content + '\n```'
            return {'role': 'assistant', 'content': content}
    result = propose_goal(ContentModel(), ReferenceProviders(manifests()), 'request', encoding='compact')
    assert result['status'] == 'needs_goal_review'
    assert accept_goal(result, result['proposal_sha256']) == result['goal']


@pytest.mark.parametrize('reply', [
    {'role': 'assistant', 'content': 'Some prose before ' + json.dumps(Model().args)},
    {'role': 'assistant', 'content': '{"execute": "delete files"}'},
    {'role': 'assistant', 'content': json.dumps(Model().args), 'tool_calls': [
        {'function': {'name': 'accept_goal', 'arguments': Model().args}}]},
    {'role': 'user', 'content': json.dumps(Model().args)}])
def test_json_fallback_does_not_accept_prose_commands_or_unknown_calls(reply):
    class ContentModel:
        def complete(self, messages, tools):
            return reply
    assert propose_goal(ContentModel(), ReferenceProviders(manifests()), 'request', encoding='compact')['status'] == 'goal_draft_failed'
