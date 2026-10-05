"""Frozen train pairs; old diagnostics and new heldout requests measured separately."""
import argparse
import json
from pathlib import Path
from time import monotonic
import torch
from vectorpro.semantic_catalog import Encoder
from vectorpro.goal_memory import GoalMemory
from vectorpro.runtime import VectorRuntime
from vectorpro.acquisition import digest
from experiments.contrastive_goal_embedding import fit, ProjectedEncoder
from experiments.goal_question_benchmark import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/contrastive_goal_embedding'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve evidence')
    args.output.mkdir()
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    base = Encoder(directory)
    identity = json.loads((directory / 'download.json').read_text())
    original = VectorRuntime.load('results/goal_router_execution_replay/program.pt')
    saved = GoalMemory.from_runtime(original, base, identity)
    examples = saved.examples
    groups = list({json.dumps(e['rules'], sort_keys=True): e['rules'] for e in examples if e['rules'] is not None}.values())
    def label(kind, reverse=False):
        src, dest = ('destination', 'source') if reverse else ('source', 'destination')
        if kind == 'keep':
            return next(g for g in groups if any(r['kind'] == 'unchanged_except' and not r['parameters'] for r in g))
        if kind == 'delete':
            return next(g for g in groups if any(r['kind'] == 'absent' and r['parameter'] == 'destination' for r in g) and not any(r['kind'] == 'equals_initial' for r in g))
        return next(g for g in groups if any(r['kind'] == 'equals_initial' and r['parameter'] == dest for r in g) and any(r['kind'] == ('absent' if kind == 'move' else 'unchanged') and r['parameter'] == src for r in g))
    definitions = [
        ('copy', False, 'Put source data at destination while leaving the original source unaltered.'),
        ('move', False, 'Put source data at destination and remove the original source.'),
        ('copy', True, 'Put destination data at source while leaving the original destination unaltered.'),
        ('move', True, 'Put destination data at source and remove the original destination.'),
        ('copy', False, 'source의 내용은 destination에 똑같이 두고 source는 지우지 마.'),
        ('move', False, 'source의 내용은 destination에 똑같이 두고 source는 지워.'),
        ('copy', True, 'destination의 내용은 source에 똑같이 두고 destination은 지우지 마.'),
        ('move', True, 'destination의 내용은 source에 똑같이 두고 destination은 지워.'),
        ('keep', False, 'Leave both named files and every other entry completely untouched.'),
        ('delete', False, 'Only destination should disappear; every other entry stays unchanged.'),
        ('keep', False, '두 경로에 있는 파일 내용과 나머지 파일들을 모두 보존해.'),
        ('delete', False, '대상 destination의 항목만 없애고 source와 나머지는 전부 보존해.'),
        ('copy', False, 'Copy source (alpha.dat) to destination (omega.dat), keeping alpha.dat.'),
        ('move', False, 'Move source (alpha.dat) to destination (omega.dat), removing alpha.dat.'),
        ('copy', True, 'Copy destination (omega.dat) to source (alpha.dat), keeping omega.dat.'),
        ('move', True, 'Move destination (omega.dat) to source (alpha.dat), removing omega.dat.'),
    ]
    cases = [{'intent': text, 'expected_rules': label(kind, reverse)} for kind, reverse, text in definitions]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'Make destination contain a compressed version of source.',
        'source를 destination에 복사하되 각 바이트를 반전해.',
        'Keep source present and delete source in the same final state.',
        '원본 파일을 보존하는 동시에 원본 파일이 없어야 해.',
        'Handle the two paths in whatever manner is appropriate.',
        '두 경로를 알아서 처리해줘.',
        'Copy source to destination only when destination is empty.',
        'Swap the initial source and destination contents.',
    ]]
    assert not {c['intent'] for c in cases} & {e['intent'] for e in examples}
    (args.output / 'heldout.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'protocol.json').write_text(json.dumps({'train_sha256': digest(examples), 'heldout_sha256': digest(cases), 'steps': 400,
        'learning_rate': 0.01, 'regularization': 0.001, 'minimum': 0.55, 'margin': 0.03,
        'protocol': 'heldout fixed before fitting; no threshold/epoch selection from heldout; frozen base MiniLM; learn square metric projection only; no LLM'}, indent=2), encoding='utf-8')
    start = monotonic()
    weight, history = fit(saved.vectors, examples)
    training_seconds = monotonic() - start
    torch.save({'weight': weight, 'identity': identity, 'train_sha256': digest(examples)}, args.output / 'projection.pt')
    checkpoint = torch.load(args.output / 'projection.pt', weights_only=True)
    projected = ProjectedEncoder(base, checkpoint['weight'])
    learned_identity = {'base': identity, 'projection_sha256': digest(weight.tolist())}
    learned = GoalMemory(examples, projected, learned_identity)
    # Save projected corpus using the existing program format, then reload without refitting.
    copied = VectorRuntime.load('results/goal_router_execution_replay/program.pt')
    learned.attach(copied)
    copied.save(args.output / 'program.pt')
    learned = GoalMemory.from_runtime(VectorRuntime.load(args.output / 'program.pt'), projected, learned_identity)
    nearest = GoalMemory(examples, base, identity)
    summaries = {}
    for suite, inputs in [('old_diagnostic', json.loads(Path('results/goal_condition_search/cases.json').read_text())), ('new_heldout', cases)]:
        rows = []
        for case in inputs:
            predictions = {'nearest43': nearest.propose(case['intent']), 'old_consensus43': saved.propose(case['intent']), 'contrastive43': learned.propose(case['intent'])}
            outcomes = {}
            for name, result in predictions.items():
                proposed = result['status'] == 'needs_goal_review'
                correct = not proposed if case['expected_rules'] is None else proposed and canonical(result['goal']['rules']) == canonical(case['expected_rules'])
                outcomes[name] = {'correct': correct, 'wrong_proposal': proposed and not correct, 'result': result}
            rows.append({'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'outcomes': outcomes})
        metrics = {name: {'correct': sum(r['outcomes'][name]['correct'] for r in rows), 'total': len(rows),
            'supported_correct': sum(r['supported'] and r['outcomes'][name]['correct'] for r in rows), 'supported_total': sum(r['supported'] for r in rows),
            'wrong_proposals': sum(r['outcomes'][name]['wrong_proposal'] for r in rows)} for name in predictions}
        summaries[suite] = {'metrics': metrics, 'cases': rows}
        print(json.dumps({suite: metrics}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'training_seconds': training_seconds, 'history': history, 'suites': summaries}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
