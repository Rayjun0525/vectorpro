"""Aggregate frozen clarification artifacts; no model calls or label feedback."""
import json
from pathlib import Path
from experiments.goal_question_benchmark import canonical


def main():
    reports = {}
    for folder in ('goal_questions', 'goal_questions_all_candidates'):
        root = Path('results') / folder
        data = json.loads((root / 'summary.json').read_text())
        labels = json.loads((root / 'cases.json').read_text())
        rows = data['cases']
        report = {}
        for suite in ('prior_abstentions', 'new'):
            metrics = {}
            for mode in ['baseline'] + list(rows[0]['questions']):
                correct = wrong = supported_correct = supported = calls = 0
                for row, label in zip(rows, labels):
                    if row['suite'] != suite:
                        continue
                    result = row['baseline'] if mode == 'baseline' else row['questions'][mode]['result']
                    proposed = result['status'] == 'needs_goal_review'
                    passed = not proposed if label['expected_rules'] is None else proposed and canonical(result['goal']['rules']) == canonical(label['expected_rules'])
                    correct += passed
                    wrong += proposed and not passed
                    supported += row['supported']
                    supported_correct += passed and row['supported']
                    if mode != 'baseline':
                        calls += len(row['questions'][mode]['trace'])
                metrics[mode] = {'correct': correct, 'wrong_proposals': wrong, 'supported_correct': supported_correct,
                                 'supported_total': supported, 'questions': calls}
            report[suite] = metrics
        if (root / 'raw_calls.json').exists():
            raw = json.loads((root / 'raw_calls.json').read_text())
            report['gemma_cost'] = {'calls': len(raw), 'seconds': sum(r['seconds'] for r in raw),
                'total_tokens': sum(r.get('usage', {}).get('total_tokens', 0) for r in raw)}
        reports[folder] = report
    Path('results/goal_question_report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    print(json.dumps(reports))


if __name__ == '__main__':
    main()
