"""Separate caller clarification potential from autonomous Gemma condition answers."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from vectorpro.goal_memory import GoalMemory
from vectorpro.semantic_catalog import Encoder
from vectorpro.acquisition import digest
from experiments.goal_questions import question, answer, teacher_answer, draft


def canonical(rules):
    return sorted(json.dumps({**r, **({'parameters': sorted(r['parameters'])} if 'parameters' in r else {})}, sort_keys=True) for r in rules)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/goal_questions'))
    parser.add_argument('--live-gemma', action='store_true')
    parser.add_argument('--candidate-count', type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous evidence')
    args.output.mkdir()
    teachers = json.loads(Path('experiments/goal_router_training.json').read_text(encoding='utf-8'))
    groups = list({digest(r['rules']): r['rules'] for r in teachers if r['rules'] is not None}.values())
    def label(kind, reverse=False):
        src, dest = ('destination', 'source') if reverse else ('source', 'destination')
        if kind == 'keep':
            return next(g for g in groups if any(r['kind'] == 'unchanged_except' and not r['parameters'] for r in g))
        if kind == 'delete':
            return next(g for g in groups if any(r['kind'] == 'absent' and r['parameter'] == 'destination' for r in g) and not any(r['kind'] == 'equals_initial' for r in g))
        return next(g for g in groups if any(r['kind'] == 'equals_initial' and r['parameter'] == dest for r in g) and any(r['kind'] == ('absent' if kind == 'move' else 'unchanged') and r['parameter'] == src for r in g))
    known = json.loads(Path('results/goal_router_validation/cases.json').read_text(encoding='utf-8'))
    old_outcomes = json.loads(Path('results/goal_router_validation/summary.json').read_text())['results']['consensus43']['cases']
    cases = [{**c, 'suite': 'prior_abstentions'} for c, r in zip(known, old_outcomes) if r['supported'] and not r['correct']]
    fresh = [('copy', False, 'Keep source intact while making destination hold the same bytes.'),
             ('move', False, 'At the end destination has the old source contents and source is gone.'),
             ('keep', False, '변경되는 파일이 하나도 없도록 source와 destination을 보존해.'),
             ('delete', False, 'Remove the destination entry and preserve the source entry exactly.'),
             ('copy', True, 'Use destination as the original and make its duplicate at source.'),
             ('move', True, 'source만 남기되 내용은 작업 전 destination의 내용이어야 해.')]
    cases += [{'suite': 'new', 'intent': text, 'expected_rules': label(kind, reverse)} for kind, reverse, text in fresh]
    cases += [{'suite': 'new', 'intent': text, 'expected_rules': None} for text in [
        'Encrypt source and keep the encrypted result at destination.',
        'source와 destination을 잘 처리해.',
        'Delete source and also leave source untouched and present.',
        'Sort the bytes of source and write them to destination.']]
    (args.output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    encoder = Encoder(directory)
    identity = json.loads((directory / 'download.json').read_text())
    memory = GoalMemory(teachers, encoder, identity, routing='ridge_consensus')
    model = None
    if args.live_gemma:
        from experiments.verified_acquisition_gemma import GemmaModel
        model = GemmaModel('/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf')
    outcomes = []
    for index, case in enumerate(cases):
        vector = F.normalize(encoder([case['intent']]).detach().cpu().float(), dim=1)[0]
        scores = (memory.vectors @ vector).tolist()
        ranked = sorted(((max(s for s, e in zip(scores, teachers) if e['rules'] is not None and canonical(e['rules']) == canonical(g)), g) for g in groups), key=lambda pair: pair[0], reverse=True)
        candidates = [g for _, g in ranked[:args.candidate_count]]
        # No expected label is used to rank candidates or generate questions.
        row = {'index': index, 'suite': case['suite'], 'intent': case['intent'], 'supported': case['expected_rules'] is not None,
               'candidate_coverage': case['expected_rules'] is not None and any(canonical(g) == canonical(case['expected_rules']) for g in candidates),
               'baseline': memory.propose(case['intent']), 'questions': {}}
        for mode in ('scripted_caller_1', 'scripted_caller_2', 'gemma_2'):
            if mode == 'gemma_2' and model is None:
                continue
            selected, trace = candidates, []
            for _ in range(1 if mode.endswith('_1') else 2):
                if len(selected) < 2:
                    break
                q = question(selected, case['intent'])
                if mode.startswith('scripted'):
                    choice = teacher_answer(case['expected_rules'], q)
                else:
                    from vectorpro.agent import tool
                    offered = tool('answer_condition', 'Choose one condition value or reject unsupported/unclear whole request',
                        {'choice': {'type': 'string', 'enum': [o['id'] for o in q['options']]}}, ('choice',))
                    messages = [{'role': 'system', 'content':
                        'Read the entire request. Answer ONE final-state condition. unchanged means preserve initial state; '
                        'absent means missing; equals_initial:NAME means original bytes from NAME. '
                        'allowed_changes lists the ONLY parameters permitted to change. '
                        'Choose outside if any requested effect needs encryption, byte transformation, conditional logic, '
                        'or contradictory states, or none of the values matches. Choose unclear if the request lacks a goal. '
                        'Your answer is a draft, not authorization. Question: ' + json.dumps(q, ensure_ascii=False)},
                        {'role': 'user', 'content': case['intent']}]
                    try:
                        response = model.complete(messages, [offered])
                        calls = response.get('tool_calls', [])
                        if len(calls) != 1 or calls[0]['function']['name'] != 'answer_condition':
                            raise ValueError('One answer required')
                        values = calls[0]['function']['arguments']
                        values = json.loads(values) if isinstance(values, str) else values
                        if set(values) != {'choice'}:
                            raise ValueError('Only a choice accepted')
                        choice = values['choice']
                    except (ValueError, KeyError, TypeError):
                        choice = 'unclear'
                try:
                    selected = answer(selected, q, choice, q['question_sha256'])
                except ValueError:
                    selected = []
                trace.append({'question': q, 'choice': choice})
            result = draft(case['intent'], selected)
            proposed = result['status'] == 'needs_goal_review'
            correct = not proposed if case['expected_rules'] is None else proposed and canonical(result['goal']['rules']) == canonical(case['expected_rules'])
            row['questions'][mode] = {'correct': correct, 'wrong_proposal': proposed and not correct, 'trace': trace, 'result': result}
        outcomes.append(row)
        (args.output / 'summary.json').write_text(json.dumps({'protocol': 'max2 questions; teacher oracle is extra information, not autonomous accuracy; draft only', 'candidate_count': args.candidate_count, 'cases': outcomes}, ensure_ascii=False, indent=2), encoding='utf-8')
        if model:
            (args.output / 'raw_calls.json').write_text(json.dumps(model.raw, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': index, 'suite': case['suite'], 'coverage': row['candidate_coverage'], 'results': {m: {'correct': v['correct'], 'wrong': v['wrong_proposal']} for m, v in row['questions'].items()}}), flush=True)


if __name__ == '__main__':
    main()
