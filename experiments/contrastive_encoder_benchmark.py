"""Train only installed MiniLM's final block; fixed new validation, no LLM."""
import argparse
import json
from pathlib import Path
from time import monotonic
import torch
import torch.nn.functional as F
from vectorpro.acquisition import digest
from vectorpro.runtime import VectorRuntime
from vectorpro.goal_memory import GoalMemory
from vectorpro.semantic_catalog import Encoder
from experiments.contrastive_goal_embedding import ProjectedEncoder
from experiments.goal_question_benchmark import canonical


def tune(encoder, examples, steps=100):
    model = encoder.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block = model.encoder.layer[-1]
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    inputs = encoder.tokenizer([e['intent'] for e in examples], padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
        hidden = output.hidden_states[-2].detach()
        mask = inputs['attention_mask'].unsqueeze(-1)
        anchor = F.normalize((output.last_hidden_state * mask).sum(1) / mask.sum(1), dim=1).detach()
    attention = model.get_extended_attention_mask(inputs['attention_mask'], inputs['attention_mask'].shape)
    labels = [digest(e['rules']) for e in examples]
    same = torch.tensor([[a == b for b in labels] for a in labels])
    diagonal = torch.eye(len(examples), dtype=torch.bool)
    positive, negative = same & ~diagonal, ~same & ~diagonal
    optimizer = torch.optim.Adam(block.parameters(), lr=0.0001)
    history = []
    for step in range(steps):
        result = block(hidden, attention_mask=attention)[0]
        z = F.normalize((result * mask).sum(1) / mask.sum(1), dim=1)
        similarities = z @ z.T
        loss = (1 - similarities[positive]).square().mean() + similarities[negative].square().mean() + 0.05 * (z - anchor).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite encoder loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            history.append({'step': step + 1, 'loss': float(loss.detach())})
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/contrastive_encoder'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve evidence')
    args.output.mkdir()
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    base = Encoder(directory)
    identity = json.loads((directory / 'download.json').read_text())
    runtime = VectorRuntime.load('results/goal_router_execution_replay/program.pt')
    saved = GoalMemory.from_runtime(runtime, base, identity)
    examples = saved.examples
    groups = list({json.dumps(e['rules'], sort_keys=True): e['rules'] for e in examples if e['rules'] is not None}.values())
    def label(kind, reverse=False):
        src, dest = ('destination', 'source') if reverse else ('source', 'destination')
        if kind == 'keep':
            return next(g for g in groups if any(r['kind'] == 'unchanged_except' and not r['parameters'] for r in g))
        if kind == 'delete':
            return next(g for g in groups if any(r['kind'] == 'absent' and r['parameter'] == 'destination' for r in g) and not any(r['kind'] == 'equals_initial' for r in g))
        return next(g for g in groups if any(r['kind'] == 'equals_initial' and r['parameter'] == dest for r in g) and any(r['kind'] == ('absent' if kind == 'move' else 'unchanged') and r['parameter'] == src for r in g))
    items = [
        ('copy', False, 'Populate destination with source bytes without losing source.'),
        ('move', False, 'Populate destination with source bytes, leaving source absent.'),
        ('copy', True, 'Populate source with destination bytes without losing destination.'),
        ('move', True, 'Populate source with destination bytes, leaving destination absent.'),
        ('copy', False, 'source에서 destination으로 사본을 전달하고 source는 남겨.'),
        ('move', False, 'source에서 destination으로 내용을 이전하고 source는 제거해.'),
        ('copy', True, 'destination에서 source로 사본을 전달하고 destination은 남겨.'),
        ('move', True, 'destination에서 source로 내용을 이전하고 destination은 제거해.'),
        ('keep', False, 'Both paths must retain their initial contents and nothing else may change.'),
        ('delete', False, 'The destination path must vanish; source and every other path must retain their initial contents.'),
        ('copy', False, 'source input-x.bin should remain after duplicating it to destination archive-y.bin.'),
        ('move', True, 'destination archive-y.bin should vanish after relocation to source input-x.bin.'),
    ]
    cases = [{'intent': text, 'expected_rules': label(kind, reverse)} for kind, reverse, text in items]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'Filter out zero bytes while copying source to destination.',
        'source를 destination에 보내되 압축된 내용이어야 해.',
        'Leave source exactly unchanged and make source disappear.',
        'Please do something with the two paths.',
    ]]
    prior = json.loads(Path('results/contrastive_goal_embedding/heldout.json').read_text())
    assert not {c['intent'] for c in cases} & ({e['intent'] for e in examples} | {c['intent'] for c in prior})
    (args.output / 'validation.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'protocol.json').write_text(json.dumps({'train_sha256': digest(examples), 'validation_sha256': digest(cases),
        'steps': 100, 'learning_rate': 0.0001, 'anchor_weight': 0.05, 'minimum': 0.55, 'margin': 0.03,
        'protocol': 'fixed validation before fitting; only final encoder block trained; no early stopping or heldout tuning'}, indent=2), encoding='utf-8')
    # Collect baselines before mutation of the model; no extra installed checkpoint.
    projected_weight = torch.load('results/contrastive_goal_embedding/projection.pt', weights_only=True)['weight']
    projection = ProjectedEncoder(base, projected_weight)
    plain = GoalMemory(examples, base, identity)
    projected_memory = GoalMemory(examples, projection, {'base': identity, 'projection': 'experimental'})
    before = {}
    for suite, inputs in [('prior24_diagnostic', prior), ('new16_validation', cases)]:
        before[suite] = [{'nearest43': plain.propose(c['intent']), 'old_consensus43': saved.propose(c['intent']), 'projection43': projected_memory.propose(c['intent'])} for c in inputs]
    start = monotonic()
    history = tune(base, examples)
    training_seconds = monotonic() - start
    state = base.model.encoder.layer[-1].state_dict()
    torch.save({'state': state, 'identity': identity, 'train_sha256': digest(examples)}, args.output / 'last_block.pt')
    restored = torch.load(args.output / 'last_block.pt', weights_only=True)
    base.model.encoder.layer[-1].load_state_dict(restored['state'])
    trained_identity = {'base': identity, 'last_block_training': digest(examples), 'steps': 100}
    trained = GoalMemory(examples, base, trained_identity)
    copied = VectorRuntime.load('results/goal_router_execution_replay/program.pt')
    trained.attach(copied)
    copied.save(args.output / 'program.pt')
    trained = GoalMemory.from_runtime(VectorRuntime.load(args.output / 'program.pt'), base, trained_identity)
    summaries = {}
    for suite, inputs in [('prior24_diagnostic', prior), ('new16_validation', cases)]:
        rows = []
        for case, predictions in zip(inputs, before[suite]):
            predictions['last_block43'] = trained.propose(case['intent'])
            results = {}
            for name, result in predictions.items():
                proposed = result['status'] == 'needs_goal_review'
                correct = not proposed if case['expected_rules'] is None else proposed and canonical(result['goal']['rules']) == canonical(case['expected_rules'])
                results[name] = {'correct': correct, 'wrong_proposal': proposed and not correct, 'result': result}
            rows.append({'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'results': results})
        metrics = {name: {'correct': sum(r['results'][name]['correct'] for r in rows), 'total': len(rows),
            'supported_correct': sum(r['supported'] and r['results'][name]['correct'] for r in rows), 'supported_total': sum(r['supported'] for r in rows),
            'wrong_proposals': sum(r['results'][name]['wrong_proposal'] for r in rows)} for name in predictions}
        summaries[suite] = {'metrics': metrics, 'cases': rows}
        print(json.dumps({suite: metrics}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'training_seconds': training_seconds, 'history': history, 'suites': summaries}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
