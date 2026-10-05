"""Hand-authored assistant replay versus live Gemma and actual tensor execution."""
import argparse
import json
from pathlib import Path
from vectorpro.goal_evidence import validate_goal, check_goal
from vectorpro.goal_draft import propose_goal
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime
from vectorpro.host import HostContext, MemoryHostContext
from experiments.reference_acquisition import manifests
from experiments.goal_question_benchmark import canonical


def intercept_goal(intent, response):
    if response['decision'] != 'supported':
        return None
    states = response['states']
    if set(states) != {'source', 'destination'}:
        raise ValueError('Complete role states required')
    rules = []
    for name, value in states.items():
        rules.append({'kind': 'equals_initial', 'parameter': name, 'source': value[8:]} if value.startswith('initial:') else {'kind': value, 'parameter': name})
    rules.append({'kind': 'unchanged_except', 'parameters': [n for n, v in states.items() if v != 'unchanged']})
    return validate_goal({'intent': intent, 'rules': rules})


def compact_model_goal(model, intent):
    from vectorpro.agent import tool
    state = {'type': 'string', 'enum': ['unchanged', 'absent', 'initial:source', 'initial:destination']}
    offered = tool('interpret_goal', 'Describe final states or decline this whole request',
        {'decision': {'type': 'string', 'enum': ['supported', 'unsupported', 'contradictory', 'unclear']},
         'states': {'type': 'object', 'properties': {'source': state, 'destination': state},
                    'required': ['source', 'destination'], 'additionalProperties': False}}, ('decision',))
    messages = [{'role': 'system', 'content':
        'Interpret the whole requested file effect. The roles source and destination are known; concrete paths are not needed to describe final states. '
        'supported: copying exact bytes, moving, removing destination, or preserving files. '
        'unsupported: byte filtering/transformation/compression/encryption or conditional operations. '
        'contradictory: incompatible final states. unclear: no specified effect. '
        'For supported provide both role states. unchanged preserves initial contents; absent removes the entry; '
        'initial:source copies original source bytes; initial:destination copies original destination bytes. '
        'Keep direction and removal separate. A source may receive destination bytes. '
        'Other files must remain unchanged. This only drafts an interpretation, never executes or approves.'},
        {'role': 'user', 'content': intent}]
    try:
        response = model.complete(messages, [offered])
        calls = response.get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'interpret_goal':
            raise ValueError('One interpretation required')
        args = calls[0]['function']['arguments']
        args = json.loads(args) if isinstance(args, str) else args
        if args.get('decision') not in ('supported', 'unsupported', 'contradictory', 'unclear'):
            raise ValueError('Invalid decision')
        goal = intercept_goal(intent, args)
        return {'status': 'needs_goal_review', 'goal': goal} if goal else {'status': 'needs_input'}
    except (ValueError, KeyError, TypeError):
        return {'status': 'goal_draft_failed'}


def observation(intent, names, paths, files, directories, host):
    after_files = host.files if isinstance(host, MemoryHostContext) else {str(p.relative_to(host.root)).replace('\\', '/'): p.read_bytes() for p in host.root.rglob('*') if p.is_file()}
    after_dirs = sorted(host.directories) if isinstance(host, MemoryHostContext) else sorted(str(p.relative_to(host.root)).replace('\\', '/') for p in host.root.rglob('*') if p.is_dir())
    case = {'inputs': paths, 'before': {k: v.hex() for k, v in files.items()}, 'after': {k: v.hex() for k, v in after_files.items()}, 'before_directories': directories, 'after_directories': after_dirs}
    return {'intent': intent, 'interface': {'parameters': [{'name': n} for n in names]}, 'state_lesson': {'training': [case], 'validation': []}, 'heldout': []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/assistant_intercept'))
    parser.add_argument('--live-gemma', action='store_true')
    parser.add_argument('--compact-gemma', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve evidence')
    args.output.mkdir()
    cases = json.loads(Path('results/contrastive_encoder/validation.json').read_text())
    answers = json.loads(Path('experiments/assistant_intercept_answers.json').read_text(encoding='utf-8'))
    assert [a['index'] for a in answers] == list(range(len(cases)))
    (args.output / 'answers.json').write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    program = Path('results/goal_router_execution_replay/program.pt')
    program_bytes = program.read_bytes()
    runtime = VectorRuntime.load(program)
    model = None
    if args.live_gemma:
        from experiments.verified_acquisition_gemma import GemmaModel
        model = GemmaModel('/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf')
    providers = ReferenceProviders(manifests())
    rows = []
    for index, (case, response) in enumerate(zip(cases, answers)):
        goal = intercept_goal(case['intent'], response)
        correct = goal is None if case['expected_rules'] is None else goal is not None and canonical(goal['rules']) == canonical(case['expected_rules'])
        row = {'index': index, 'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'assistant_correct': correct, 'assistant_goal': goal, 'native': []}
        if model:
            proposal = compact_model_goal(model, case['intent']) if args.compact_gemma else propose_goal(model, providers, case['intent'], encoding='rules')
            proposed = proposal['status'] == 'needs_goal_review'
            passed = proposal['status'] == 'needs_input' if case['expected_rules'] is None else proposed and canonical(proposal['goal']['rules']) == canonical(case['expected_rules'])
            row['gemma'] = {'correct': passed, 'wrong_proposal': proposed and not passed, 'result': proposal}
        matches = []
        role_paths = response.get('literal_paths', {'source': 'source.bin', 'destination': 'target.bin'})
        if goal is not None:
            for contract in runtime.contracts():
                names = [p['name'] for p in contract['parameters']]
                if names != ['source', 'destination']:
                    continue
                for argument_order in (names, list(reversed(names))):
                    arguments = dict(zip(names, [role_paths[n] for n in argument_order]))
                    matched = True
                    for payload in (b'preview\0\xff', b''):
                        files = {role_paths['source']: payload, role_paths['destination']: b'old destination', 'keep': b'keep'}
                        host = MemoryHostContext(files, directories=['empty'])
                        virtual = VectorRuntime.load(program, host=host)
                        virtual.call_contract(contract['id'], arguments, 16)
                        record = observation(case['intent'], names, [role_paths[n] for n in names], files, ['empty'], host)
                        try:
                            check_goal(goal, record)
                        except ValueError:
                            matched = False
                            break
                    if matched:
                        matches.append({'id': contract['id'], 'arguments': arguments})
            if len(matches) == 1:
                selected = matches[0]
                for number, payload in enumerate((b'native\0\xff', b'', '새 입력 값'.encode())):
                    root = args.output / f'case{index}_{number}'
                    root.mkdir()
                    files = {role_paths['source']: payload, role_paths['destination']: b'old destination', 'keep': b'keep'}
                    for path, value in files.items():
                        (root / path).write_bytes(value)
                    (root / 'empty').mkdir()
                    native = VectorRuntime.load(program, host=HostContext(root))
                    result = native.call_contract(selected['id'], selected['arguments'], 16)
                    record = observation(case['intent'], ['source', 'destination'], [role_paths['source'], role_paths['destination']], files, ['empty'], native.host)
                    # Expected goal is used AFTER selecting and executing the assistant route, never as a selector.
                    passed = False
                    if case['expected_rules'] is not None:
                        try:
                            check_goal({'intent': case['intent'], 'rules': case['expected_rules']}, record)
                            passed = True
                        except ValueError:
                            pass
                    row['native'].append({'passed': passed, 'result': result, 'observation': record})
            else:
                row['execution_status'] = 'no_unique_stored_contract'
        else:
            row['execution_status'] = 'declined_no_execution'
        row['matching_routes'] = matches
        rows.append(row)
        metrics = {'assistant_correct': sum(r['assistant_correct'] for r in rows), 'cases': len(rows),
            'native_passed': sum(n['passed'] for r in rows for n in r['native']), 'native_total': sum(len(r['native']) for r in rows)}
        if model:
            metrics.update({'gemma_correct': sum(r['gemma']['correct'] for r in rows), 'gemma_wrong_proposals': sum(r['gemma']['wrong_proposal'] for r in rows)})
            (args.output / 'raw_calls.json').write_text(json.dumps(model.raw, ensure_ascii=False, indent=2), encoding='utf-8')
        (args.output / 'summary.json').write_text(json.dumps({'protocol': 'assistant hand-authored replay after seeing labels; not blind or model-size causal proof; caller supplies concrete role paths; expected labels only score interpretation and final native states', 'compact_gemma': args.compact_gemma, 'metrics': metrics, 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': index, 'assistant_correct': correct, 'native_passed': sum(n['passed'] for n in row['native']), 'gemma_correct': row.get('gemma', {}).get('correct')}), flush=True)
    assert program.read_bytes() == program_bytes


if __name__ == '__main__':
    main()
