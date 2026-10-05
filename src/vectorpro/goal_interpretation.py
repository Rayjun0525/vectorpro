"""Bounded semantic drafts: callers/models describe states, backend builds frames."""
import json

from vectorpro.acquisition import digest
from vectorpro.goal_evidence import validate_goal


def normalize_goal(goal, *, file_parameters=()):
    """Normalize equivalent predicates; self-byte equality needs a file assumption.

    The optional assumption is for comparison under known file inputs only. It
    must not silently strengthen a live draft or change its approval digest.
    """
    goal = validate_goal(goal)
    known_files = set(file_parameters)
    rules = []
    for rule in goal['rules']:
        if (rule['kind'] == 'equals_initial' and rule['parameter'] == rule['source']
                and rule['parameter'] in known_files):
            rule = {'kind': 'unchanged', 'parameter': rule['parameter']}
        rules.append(rule)
    preserved = {r['parameter'] for r in rules if r['kind'] == 'unchanged'}
    for rule in rules:
        if rule['kind'] == 'unchanged_except':
            rule['parameters'] = sorted(set(rule['parameters']) - preserved)
    unique = {json.dumps(r, sort_keys=True): r for r in rules}
    return {'intent': goal['intent'], 'rules': [unique[k] for k in sorted(unique)]}


def goal_from_states(intent, names, states):
    """Common structured-input path; no language model or execution."""
    if (not isinstance(names, list) or not 1 <= len(names) <= 16
            or any(not isinstance(n, str) or not 1 <= len(n) <= 100 for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('bounded unique parameter names required')
    if not isinstance(states, dict) or set(states) != set(names):
        raise ValueError('states must cover every interface parameter exactly')
    allowed = {'unchanged', 'absent', 'present'} | {'initial:' + n for n in names}
    rules, changed = [], []
    for name in names:
        value = states[name]
        if not isinstance(value, str) or value not in allowed:
            raise ValueError('unknown final state')
        rules.append({'kind': 'equals_initial', 'parameter': name, 'source': value[8:]}
                     if value.startswith('initial:') else {'kind': value, 'parameter': name})
        if value != 'unchanged':
            changed.append(name)
    rules.append({'kind': 'unchanged_except', 'parameters': changed})
    return normalize_goal({'intent': intent, 'rules': rules})


def propose_interpretation(model, parameters, intent):
    """Extract semantics without asking the model to select supported contracts."""
    from vectorpro.agent import tool
    from vectorpro.goal_draft import review_lines
    names = [p['name'] for p in parameters]
    allowed_by_name = {n: ['unchanged', 'absent', 'present', 'unspecified']
                       + ['initial:' + other for other in names if other != n] for n in names}
    effect_names = ['byte_transformation', 'filtering', 'compression', 'encryption',
                    'conditional_effect', 'process', 'network', 'other_effect']
    effects = {'type': 'array', 'items': {'type': 'string', 'enum': effect_names},
               'maxItems': 8, 'uniqueItems': True}
    issues = {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 300},
              'maxItems': 8}
    offered = tool('describe_goal', 'Describe requested final states and any effects or ambiguities not captured by them',
        {'states': {'type': 'object', 'properties': {n: {'type': 'string', 'enum': allowed_by_name[n]} for n in names},
                    'required': names, 'additionalProperties': False},
         'unrepresented_effects': effects, 'unresolved': issues},
        ('states', 'unrepresented_effects', 'unresolved'))
    messages = [{'role': 'system', 'content':
        'Describe the whole requested final file state in one describe_goal call. '
        'Roles do not fix direction: source may receive the original destination bytes. '
        'unchanged = preserve the original entry and bytes. absent = remove the entry. '
        'initial:NAME = receive the ORIGINAL bytes of the OTHER named role. '
        'present = existence ONLY; never use it instead of copying bytes. '
        'For a move the original role must be absent, not unchanged. '
        'Preserving everything means all roles unchanged. Other entries are preserved by the backend. '
        'For copying, moving, deleting or preserving, unrepresented_effects is []. '
        'Only list additional effects using codes: byte_transformation, filtering, compression, encryption, '
        'conditional_effect, process, network, other_effect. Do not discard any requested effect. '
        'unresolved is [] for clear consistent goals. For conflicting requirements or unspecified goals, '
        'explain the issue there and use unspecified for affected roles. '
        'This is a draft for caller review, never approval or execution.\nInterface: ' + json.dumps(parameters)},
        {'role': 'user', 'content': intent}]
    message = model.complete(messages, [offered])
    try:
        if message.get('role') != 'assistant':
            raise ValueError('assistant semantic description required')
        calls = message.get('tool_calls', [])
        if not calls:
            # Some Chat Completions servers return data JSON instead of a tool
            # envelope. Accept only the complete, strictly validated draft data.
            content = message.get('content', '')
            if not isinstance(content, str) or not 1 <= len(content) <= 16000:
                raise ValueError('bounded semantic JSON required')
            content = content.strip()
            if content.startswith('```json\n') and content.endswith('\n```'):
                content = content[8:-4]
            args = json.loads(content)
        elif len(calls) != 1 or calls[0]['function']['name'] != 'describe_goal':
            raise ValueError('one semantic description required')
        else:
            args = calls[0]['function']['arguments']
            args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(args, dict) or set(args) != {'states', 'unrepresented_effects', 'unresolved'}:
            raise ValueError('complete semantic description required')
        for field in ('unrepresented_effects', 'unresolved'):
            values = args[field]
            if (not isinstance(values, list) or len(values) > 8
                    or any(not isinstance(v, str) or not 1 <= len(v) <= 300 for v in values)):
                raise ValueError('bounded semantic issues required')
        if any(effect not in effect_names for effect in args['unrepresented_effects']):
            raise ValueError('unknown semantic effect')
        values = args['states']
        if (not isinstance(values, dict) or set(values) != set(names)
                or any(not isinstance(v, str) or v not in allowed_by_name[n] for n, v in values.items())):
            raise ValueError('complete valid states required')
        if args['unrepresented_effects'] or args['unresolved'] or 'unspecified' in values.values():
            reason = 'goal_outside_representation' if args['unrepresented_effects'] else 'goal_needs_clarification'
            labels = {'byte_transformation': '바이트 변환', 'filtering': '필터링',
                      'compression': '압축', 'encryption': '암호화', 'conditional_effect': '조건부 효과',
                      'process': '프로세스', 'network': '네트워크', 'other_effect': '그 밖의 효과'}
            if args['unrepresented_effects']:
                question = '현재 파일 상태 목표에 담기지 않는 효과: ' + ', '.join(labels[e] for e in args['unrepresented_effects']) + '. 이 효과를 표현할 목표·근거가 필요합니다.'
            elif args['unresolved']:
                question = '확인이 필요한 요구: ' + '; '.join(args['unresolved'])
            else:
                missing = [name for name, value in values.items() if value == 'unspecified']
                question = ', '.join(missing) + '의 원하는 최종 상태를 알려 주세요.'
            return {'status': 'needs_input', 'reason': reason, 'interpretation': args,
                    'question': question,
                    'intent_independently_verified': False}
        goal = goal_from_states(intent, names, values)
        return {'status': 'needs_goal_review', 'goal': goal, 'interpretation': args,
                'proposal_sha256': digest(goal), 'review': review_lines(goal),
                'intent_independently_verified': False}
    except (ValueError, KeyError, TypeError) as error:
        return {'status': 'goal_draft_failed', 'reason': str(error)}
