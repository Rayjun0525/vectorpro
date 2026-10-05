"""Experimental candidate contrast: questions clarify, never authorize execution."""
import copy
import math
from vectorpro.acquisition import digest
from vectorpro.goal_evidence import validate_goal


def profile(rules):
    result = {}
    for rule in rules:
        if rule['kind'] == 'unchanged_except':
            result['allowed_changes'] = tuple(sorted(rule['parameters']))
        else:
            key = 'state:' + rule['parameter']
            value = rule['kind'] + (':' + rule['source'] if rule['kind'] == 'equals_initial' else '')
            if key in result:
                raise ValueError('Contrast experiment requires one final predicate per parameter')
            result[key] = value
    return result


def portable(value):
    return list(value) if isinstance(value, tuple) else value


def question(candidates, intent):
    if not 2 <= len(candidates) <= 16:
        raise ValueError('Contrast needs 2..16 candidates')
    profiles = [profile(rules) for rules in candidates]
    keys = set.intersection(*(set(p) for p in profiles))
    partitions = []
    for key in sorted(keys):
        values = [p[key] for p in profiles]
        counts = [values.count(v) for v in set(values)]
        entropy = -sum((n / len(values)) * math.log2(n / len(values)) for n in counts)
        if entropy:
            partitions.append((entropy, key))
    if not partitions:
        raise ValueError('No distinguishing condition')
    _, key = max(partitions)
    values = sorted(set(p[key] for p in profiles), key=repr)
    options = [{'id': 'option_' + str(i), 'value': portable(value)} for i, value in enumerate(values)]
    options += [{'id': 'outside', 'value': None}, {'id': 'unclear', 'value': None}]
    data = {'intent': intent, 'condition': key, 'options': options,
            'candidate_hashes': [digest(rules) for rules in candidates]}
    data['question_sha256'] = digest(data)
    return data


def answer(candidates, q, choice, question_sha256):
    if q != question(candidates, q['intent']) or question_sha256 != q['question_sha256']:
        raise ValueError('Question or candidates changed')
    selected = next((o for o in q['options'] if o['id'] == choice), None)
    if selected is None:
        raise ValueError('Unknown answer option')
    if choice in ('outside', 'unclear'):
        return []
    return [copy.deepcopy(rules) for rules in candidates if portable(profile(rules)[q['condition']]) == selected['value']]


def teacher_answer(expected, q):
    """Evaluation oracle; must never be used as an autonomous model answer."""
    if expected is None:
        return 'outside'
    value = portable(profile(expected).get(q['condition']))
    return next((o['id'] for o in q['options'] if o['value'] == value and o['id'] not in ('outside', 'unclear')), 'outside')


def draft(intent, candidates):
    if len(candidates) != 1:
        return {'status': 'needs_input'}
    goal = validate_goal({'intent': intent, 'rules': candidates[0]})
    return {'status': 'needs_goal_review', 'goal': goal, 'intent_independently_verified': False}
