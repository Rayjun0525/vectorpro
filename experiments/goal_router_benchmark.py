"""Frozen contrast/role stress cases; compare teacher expansion and learned routing."""
import hashlib
import json
import argparse
from pathlib import Path
from vectorpro.goal_memory import GoalMemory
from vectorpro.semantic_catalog import Encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/goal_router_stress'))
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise RuntimeError('Preserve previous measurements')
    output.mkdir()
    teachers = json.loads(Path('experiments/goal_router_training.json').read_text(encoding='utf-8'))
    seed = json.loads(Path('experiments/goal_memory_seed.json').read_text(encoding='utf-8'))
    # Rules are the teacher's explicit labels, not inferred from the request being tested.
    groups = []
    for row in teachers:
        if row['rules'] is not None and row['rules'] not in groups:
            groups.append(row['rules'])
    # Identify the six goal classes by their exact postconditions.
    def rules(kind, reverse=False):
        dest, src = ('source', 'destination') if reverse else ('destination', 'source')
        if kind == 'keep':
            return next(g for g in groups if any(r['kind'] == 'unchanged_except' and r['parameters'] == [] for r in g))
        if kind == 'delete':
            return next(g for g in groups if any(r['kind'] == 'absent' and r['parameter'] == 'destination' for r in g) and not any(r['kind'] == 'equals_initial' for r in g))
        return next(g for g in groups if any(r['kind'] == 'equals_initial' and r['parameter'] == dest and r['source'] == src for r in g) and any(r['kind'] == ('absent' if kind == 'move' else 'unchanged') and r['parameter'] == src for r in g))
    texts = [
        ('copy', False, 'Put identical bytes from source into destination; keep source available.'),
        ('copy', False, '원본 source를 지우지 않고 destination에 같은 내용을 남겨줘.'),
        ('copy', False, 'After copying source to destination, both files must contain the original source bytes.'),
        ('move', False, 'Destination must receive source contents, and source must disappear afterward.'),
        ('move', False, 'source 내용은 destination에 남기되 source 파일은 없애줘.'),
        ('move', False, 'Keep only destination after relocating source there.'),
        ('keep', False, 'Leave everything on disk exactly as it is, including both files.'),
        ('keep', False, 'source와 destination 모두 현재 상태 그대로 유지해줘.'),
        ('keep', False, 'Do not copy, remove, or change either file.'),
        ('delete', False, 'Erase destination only; source and all other files must stay untouched.'),
        ('delete', False, 'destination만 없애고 source는 그대로 놔둬.'),
        ('delete', False, 'Make destination absent without editing source.'),
        ('copy', True, 'Replace source contents with a copy of destination; preserve destination.'),
        ('copy', True, 'destination을 source로 복사하고 destination은 보존해줘.'),
        ('copy', True, 'Both files should contain the original destination bytes without removing destination.'),
        ('move', True, 'Move destination into source, removing the old destination file.'),
        ('move', True, 'destination의 내용을 source에 옮긴 뒤 destination은 삭제해줘.'),
        ('move', True, 'Only source should remain, containing the previous destination contents.'),
    ]
    cases = [{'intent': text, 'expected_rules': rules(kind, reverse)} for kind, reverse, text in texts]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'Copy source to destination and also keep destination unchanged when their contents differ.',
        'source를 삭제하면서 동시에 source를 원래 상태로 보존해줘.',
        'Append source bytes to the existing destination bytes.',
        'destination에 source를 이어 붙여줘.',
        'If source is empty delete destination; otherwise copy source.',
        '파일을 적당히 처리해줘.',
    ]]
    if args.validation:
        texts = [
            ('copy', False, 'Create destination from source without removing or altering source.'),
            ('copy', False, 'source의 복제본을 destination에 만들고 source는 손대지 마.'),
            ('move', False, 'Remove source only after its original contents have been placed in destination.'),
            ('move', False, 'source를 destination으로 이전해서 이전 source 경로에는 파일이 없게 해줘.'),
            ('keep', False, 'The final disk state must be identical to the initial disk state.'),
            ('keep', False, '두 파일 모두 수정도 삭제도 하지 말고 전체 디스크 상태를 유지해.'),
            ('delete', False, 'Leave source alone and ensure destination no longer exists.'),
            ('delete', False, 'source는 수정하지 말고 destination 파일만 지워.'),
            ('copy', True, 'Duplicate destination at source while leaving destination intact.'),
            ('copy', True, 'source에 destination의 사본을 만들고 destination은 변경하지 마.'),
            ('move', True, 'Relocate destination to source so that destination becomes absent.'),
            ('move', True, 'destination을 source로 이동해서 destination 경로는 비워줘.'),
        ]
        cases = [{'intent': text, 'expected_rules': rules(kind, reverse)} for kind, reverse, text in texts]
        cases += [{'intent': text, 'expected_rules': None} for text in [
            'Remove source but keep source present with its original contents.',
            'source를 복사하면서 destination의 기존 내용을 합쳐줘.',
            'Convert source text to uppercase and save it in destination.',
            'source와 destination을 어떻게든 정리해줘.',
            'Copy only if destination does not already exist.',
            'Run a web server serving the source file.',
        ]]
    frozen = json.dumps(cases, ensure_ascii=False, indent=2).encode('utf-8')
    (output / 'cases.json').write_bytes(frozen)
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    encoder = Encoder(directory)
    identity = json.loads((directory / 'download.json').read_text())
    def canonical(value):
        return sorted(json.dumps({**r, **({'parameters': sorted(r['parameters'])} if 'parameters' in r else {})}, sort_keys=True) for r in value)
    summaries = {}
    for name, examples, routing in [('nearest13', seed, 'nearest'), ('nearest43', teachers, 'nearest'), ('ridge43', teachers, 'ridge'), ('consensus43', teachers, 'ridge_consensus')]:
        memory = GoalMemory(examples, encoder, identity, routing=routing)
        rows = []
        for case in cases:
            result = memory.propose(case['intent'])
            proposed = result['status'] == 'needs_goal_review'
            correct = (not proposed) if case['expected_rules'] is None else (proposed and canonical(result['goal']['rules']) == canonical(case['expected_rules']))
            rows.append({'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'correct': correct, 'wrong_proposal': proposed and not correct, 'result': result})
        summaries[name] = {'correct': sum(r['correct'] for r in rows), 'total': len(rows), 'wrong_proposals': sum(r['wrong_proposal'] for r in rows), 'supported_correct': sum(r['correct'] and r['supported'] for r in rows), 'supported_total': sum(r['supported'] for r in rows), 'cases': rows}
        print(json.dumps({name: {k: v for k, v in summaries[name].items() if k != 'cases'}}), flush=True)
    (output / 'summary.json').write_text(json.dumps({'cases_sha256': hashlib.sha256(frozen).hexdigest(), 'protocol': 'fixed requests; authored labels; draft only; validation suite unseen by routers', 'suite': 'validation' if args.validation else 'stress', 'results': summaries}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
