"""Fixed minimal pairs, condition routing, and exact virtual tensor effect checks."""
import argparse
import json
from pathlib import Path
from vectorpro.semantic_catalog import Encoder
from vectorpro.goal_memory import GoalMemory
from vectorpro.runtime import VectorRuntime
from vectorpro.host import MemoryHostContext
from vectorpro.goal_evidence import check_goal
from experiments.goal_condition_search import ConditionSearch
from experiments.goal_question_benchmark import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/goal_condition_search'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve evidence')
    args.output.mkdir()
    program = Path('results/goal_router_execution_replay/program.pt')
    original_bytes = program.read_bytes()
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    encoder, identity = Encoder(directory), json.loads((directory / 'download.json').read_text())
    runtime = VectorRuntime.load(program)
    memory = GoalMemory.from_runtime(runtime, encoder, identity)
    groups = list({json.dumps(e['rules'], sort_keys=True): e['rules'] for e in memory.examples if e['rules'] is not None}.values())
    def label(kind, reverse=False):
        src, dest = ('destination', 'source') if reverse else ('source', 'destination')
        if kind == 'keep':
            return next(g for g in groups if any(r['kind'] == 'unchanged_except' and not r['parameters'] for r in g))
        if kind == 'delete':
            return next(g for g in groups if any(r['kind'] == 'absent' and r['parameter'] == 'destination' for r in g) and not any(r['kind'] == 'equals_initial' for r in g))
        return next(g for g in groups if any(r['kind'] == 'equals_initial' and r['parameter'] == dest for r in g) and any(r['kind'] == ('absent' if kind == 'move' else 'unchanged') and r['parameter'] == src for r in g))
    pairs = [
        ('copy', False, 'Transfer the source bytes into destination, retaining the source file.'),
        ('move', False, 'Transfer the source bytes into destination, deleting the source file.'),
        ('copy', False, 'source를 destination에 전달하고 source는 남겨둬.'),
        ('move', False, 'source를 destination에 전달하고 source는 없애줘.'),
        ('copy', True, 'Transfer the destination bytes into source, retaining the destination file.'),
        ('move', True, 'Transfer the destination bytes into source, deleting the destination file.'),
        ('copy', True, 'destination을 source에 전달하고 destination은 남겨둬.'),
        ('move', True, 'destination을 source에 전달하고 destination은 없애줘.'),
        ('keep', False, 'Keep source and destination exactly as they were; alter no files.'),
        ('delete', False, 'Keep source exactly as it was; remove destination only.'),
        ('keep', False, 'source와 destination을 둘 다 그대로 두고 다른 파일도 그대로 둬.'),
        ('delete', False, 'source는 그대로 두고 destination만 제거하고 다른 파일도 그대로 둬.'),
    ]
    cases = [{'intent': text, 'expected_rules': label(kind, reverse)} for kind, reverse, text in pairs]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'Keep source intact and also delete source after copying it.',
        'Reverse source bytes into destination while retaining source.',
        'source를 보존하고 destination에 암호화해서 전달해.',
        'Do the appropriate thing with source and destination.',
    ]]
    (args.output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    search = ConditionSearch(memory)
    rows = []
    for index, case in enumerate(cases):
        baseline, condition = memory.propose(case['intent']), search.propose(case['intent'])
        agreement = condition if baseline['status'] == condition['status'] == 'needs_goal_review' and canonical(baseline['goal']['rules']) == canonical(condition['goal']['rules']) else {'status': 'needs_input'}
        results = {}
        for name, proposal in [('baseline', baseline), ('conditions', condition), ('agreement', agreement)]:
            proposed = proposal['status'] == 'needs_goal_review'
            correct = not proposed if case['expected_rules'] is None else proposed and canonical(proposal['goal']['rules']) == canonical(case['expected_rules'])
            results[name] = {'correct': correct, 'wrong_proposal': proposed and not correct, 'proposal': proposal}
        # Explicitly test proposed goals against actual stored contract effects on isolated fixtures.
        previews = []
        if condition['status'] == 'needs_goal_review':
            for contract in runtime.contracts():
                names = [p['name'] for p in contract['parameters']]
                if names != ['source', 'destination']:
                    continue
                for paths in (['source.bin', 'target.bin'], ['target.bin', 'source.bin']):
                    observations = []
                    for payload in (b'new\0\xff', b'', b'longer unseen bytes'):
                        files = {'source.bin': payload, 'target.bin': b'old destination', 'keep': b'keep'}
                        host = MemoryHostContext(files, directories=['empty'])
                        virtual = VectorRuntime.load(program, host=host)
                        virtual.call_contract(contract['id'], dict(zip(names, paths)), 16)
                        observations.append({'inputs': ['source.bin', 'target.bin'], 'before': {k: v.hex() for k, v in files.items()},
                            'after': {k: v.hex() for k, v in host.files.items()}, 'before_directories': ['empty'], 'after_directories': sorted(host.directories)})
                    record = {'intent': case['intent'], 'interface': {'parameters': [{'name': n} for n in names]},
                        'state_lesson': {'training': observations[:1], 'validation': observations[1:2]}, 'heldout': observations[2:]}
                    try:
                        check_goal(condition['goal'], record)
                        previews.append({'contract_id': contract['id'], 'arguments': dict(zip(names, paths)), 'checked_cases': 3})
                    except ValueError:
                        pass
        rows.append({'index': index, 'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'results': results, 'matching_tensor_routes': previews})
        print(json.dumps({'case': index, 'results': {m: {'correct': r['correct'], 'wrong': r['wrong_proposal']} for m, r in results.items()}, 'matching_routes': len(previews)}), flush=True)
    metrics = {name: {'correct': sum(r['results'][name]['correct'] for r in rows), 'supported_correct': sum(r['supported'] and r['results'][name]['correct'] for r in rows), 'supported_total': 12, 'total': 16, 'wrong_proposals': sum(r['results'][name]['wrong_proposal'] for r in rows)} for name in ('baseline', 'conditions', 'agreement')}
    assert program.read_bytes() == original_bytes
    (args.output / 'summary.json').write_text(json.dumps({'protocol': '16 new fixed minimal-pair requests; no LLM/caller answers; condition weights fit existing persisted examples only; preview never proves intent', 'metrics': metrics, 'cases': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
