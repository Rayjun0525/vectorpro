"""Freeze a small local-model diagnostic before inference; never execute drafts."""
import argparse
import json
from pathlib import Path

from experiments.assistant_intercept import compact_model_goal
from experiments.reference_acquisition import manifests
from experiments.verified_acquisition_gemma import GemmaModel
from vectorpro.goal_draft import propose_goal
from vectorpro.goal_interpretation import goal_from_states, normalize_goal
from vectorpro.reference_evidence import ReferenceProviders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        {'name': 'reverse_copy', 'intent': 'Use destination as the original. Replace source with its exact bytes, keeping destination intact.',
         'states': {'source': 'initial:destination', 'destination': 'unchanged'}},
        {'name': 'reverse_move', 'intent': 'destination 원본을 source에 옮기고 destination 항목은 제거해. 다른 파일은 보존해.',
         'states': {'source': 'initial:destination', 'destination': 'absent'}},
        {'name': 'contradiction_en', 'intent': 'At the end source must not exist, and the same source entry must remain intact at the end.', 'states': None},
        {'name': 'contradiction_ko', 'intent': '작업 종료 시 destination 파일은 없어야 하고, 동일한 destination 파일은 원래 내용 그대로 존재해야 해.', 'states': None},
        {'name': 'unsupported', 'intent': 'Keep source intact and write its encrypted bytes into destination.', 'states': None},
        {'name': 'unclear', 'intent': '두 파일로 적절한 작업을 해줘.', 'states': None},
    ]
    # These are a small authored diagnostic, not a blind independent evaluation.
    (args.output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    model = GemmaModel(args.model)
    providers = ReferenceProviders(manifests())
    rows = []
    for method in ('baseline', 'semantic'):
        for case in cases:
            result = (compact_model_goal(model, case['intent']) if method == 'baseline'
                      else propose_goal(model, providers, case['intent'], encoding='compact'))
            proposed = result['status'] == 'needs_goal_review'
            expected = (goal_from_states(case['intent'], ['source', 'destination'], case['states'])
                        if case['states'] is not None else None)
            correct = result['status'] == 'needs_input' if expected is None else (
                proposed and normalize_goal(result['goal'], file_parameters=['source', 'destination'])
                == normalize_goal(expected, file_parameters=['source', 'destination']))
            rows.append({'method': method, 'name': case['name'], 'supported': expected is not None,
                         'correct': correct, 'wrong_proposal': proposed and not correct, 'proposal': result})
            metrics = {}
            for name in ('baseline', 'semantic'):
                subset = [r for r in rows if r['method'] == name]
                metrics[name] = {'completed': len(subset), 'correct': sum(r['correct'] for r in subset),
                                 'supported_correct': sum(r['supported'] and r['correct'] for r in subset),
                                 'wrong_proposals': sum(r['wrong_proposal'] for r in subset)}
            (args.output / 'summary.json').write_text(json.dumps({
                'model': args.model, 'protocol': 'small authored diagnostic; drafts only; no execution or approval',
                'metrics': metrics, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
            (args.output / 'raw_calls.json').write_text(json.dumps(model.raw, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'method': method, 'name': case['name'], 'correct': correct,
                              'status': result['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
