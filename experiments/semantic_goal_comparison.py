"""Freeze unseen cases before comparing classification-first and semantic drafts."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.assistant_intercept import compact_model_goal, intercept_goal
from experiments.goal_question_benchmark import canonical
from experiments.reference_acquisition import manifests
from experiments.strong_goal_model import OllamaModel, execute_goal, fresh_cases
from vectorpro.goal_draft import propose_goal
from vectorpro.goal_evidence import check_goal
from vectorpro.goal_interpretation import normalize_goal
from vectorpro.reference_evidence import ReferenceProviders


def unseen_cases():
    descriptions = [
        ('source는 보관하고 destination 내용만 source의 작업 전 내용으로 덮어써.', 'unchanged', 'initial:source'),
        ('destination 내용으로 source를 덮어써. destination은 원래대로 남겨.', 'initial:destination', 'unchanged'),
        ('source의 원본 바이트를 destination에 넘긴 뒤 source 경로는 없어져야 해.', 'absent', 'initial:source'),
        ('destination의 원본 바이트를 source에 남기고 destination 경로는 없애.', 'initial:destination', 'absent'),
        ('Keep the original at source. Only destination should receive its exact bytes.', 'unchanged', 'initial:source'),
        ('The original is destination. Duplicate it into source without altering destination.', 'initial:destination', 'unchanged'),
        ('After relocating source into destination, the old source name must not exist.', 'absent', 'initial:source'),
        ('After relocating destination into source, the old destination name must not exist.', 'initial:destination', 'absent'),
        ('실제 파일을 수정하거나 삭제하거나 새로 만들지 말고 현재 상태를 그대로 유지해.', 'unchanged', 'unchanged'),
        ('Do no work on these files: both entries and all other entries must remain untouched.', 'unchanged', 'unchanged'),
        ('source는 그대로 두고 destination 항목만 없애 줘.', 'unchanged', 'absent'),
        ('Keep source and all other entries intact; destination alone must cease to exist.', 'unchanged', 'absent'),
    ]
    cases = [{'intent': text, 'expected_rules': intercept_goal(text, {'decision': 'supported',
        'states': {'source': source, 'destination': destination}})['rules']}
        for text, source, destination in descriptions]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'destination에는 source 내용을 역순으로 저장하되 source는 보존해.',
        'Copy source to destination only if destination is missing.',
        'destination을 없애면서 같은 destination이 원래 내용 그대로 존재하도록 해.',
        'Choose some helpful action for these files.']]
    return cases


def final_unseen_cases():
    descriptions = [
        ('source 원본을 유지하며 destination의 내용만 그 원본과 똑같게 해.', 'unchanged', 'initial:source'),
        ('destination이 원본이야. 그것은 유지하고 source에는 동일한 사본을 둬.', 'initial:destination', 'unchanged'),
        ('source의 내용을 destination에 남기되 source 파일 항목은 제거해.', 'absent', 'initial:source'),
        ('destination의 내용을 source에 남기되 destination 파일 항목은 제거해.', 'initial:destination', 'absent'),
        ('Put a byte-for-byte replica of source at destination while leaving source intact.', 'unchanged', 'initial:source'),
        ('Put a byte-for-byte replica of destination at source while leaving destination intact.', 'initial:destination', 'unchanged'),
        ('The resulting destination must contain the old source bytes, and source must be removed.', 'absent', 'initial:source'),
        ('The resulting source must contain the old destination bytes, and destination must be removed.', 'initial:destination', 'absent'),
        ('source와 destination 모두 현재 상태를 유지하고 다른 항목도 변경하지 마.', 'unchanged', 'unchanged'),
        ('Retain every existing entry, including source and destination, with identical contents.', 'unchanged', 'unchanged'),
        ('destination 파일 항목을 삭제하고 source 및 나머지는 원래대로 유지해.', 'unchanged', 'absent'),
        ('Delete the destination entry without touching source or any other entry.', 'unchanged', 'absent'),
    ]
    cases = [{'intent': text, 'expected_rules': intercept_goal(text, {'decision': 'supported',
        'states': {'source': source, 'destination': destination}})['rules']}
        for text, source, destination in descriptions]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'source를 암호화한 바이트를 destination에 저장하고 source는 그대로 둬.',
        'Duplicate source into destination only when source is nonempty.',
        'source 파일은 사라져야 하고 동일한 source 파일은 삭제 없이 그대로 유지해.',
        '두 파일로 뭔가 적당한 작업을 해줘.']]
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/semantic_goal_comparison'))
    parser.add_argument('--model', default='gemma4:12b')
    parser.add_argument('--final-unseen', action='store_true', help='only run the separately frozen final 16 cases')
    args = parser.parse_args()
    args.output.mkdir()
    suites = {'replay': fresh_cases(), 'unseen': unseen_cases()}
    if args.final_unseen:
        suites = {'final_unseen': final_unseen_cases()}
    (args.output / 'cases.json').write_text(json.dumps(suites, ensure_ascii=False, indent=2))
    model = OllamaModel('http://host.docker.internal:11434', args.model)
    (args.output / 'model.json').write_text(json.dumps(model.request('/api/show', {'model': args.model}), ensure_ascii=False, indent=2))
    program = Path('results/goal_router_execution_replay/program.pt')
    original = program.read_bytes()
    providers = ReferenceProviders(manifests())
    rows = []
    for suite, cases in suites.items():
        for method in ('baseline', 'semantic'):
            for index, case in enumerate(cases):
                proposal = (compact_model_goal(model, case['intent']) if method == 'baseline' else
                            propose_goal(model, providers, case['intent'], encoding='compact'))
                proposed = proposal['status'] == 'needs_goal_review'
                expected = case['expected_rules']
                strict = proposal['status'] == 'needs_input' if expected is None else proposed and canonical(proposal['goal']['rules']) == canonical(expected)
                equivalent = strict
                if expected is not None and proposed:
                    equivalent = (normalize_goal(proposal['goal'], file_parameters=['source', 'destination']) ==
                                  normalize_goal({'intent': case['intent'], 'rules': expected}, file_parameters=['source', 'destination']))
                execution = execute_goal(program, proposal['goal'], args.output / f'{suite}_{method}_{index}') if proposed else {'status': 'declined', 'observations': []}
                passed = []
                for record in execution['observations']:
                    try:
                        if expected is None:
                            raise ValueError('unsupported')
                        check_goal({'intent': case['intent'], 'rules': expected}, record)
                        passed.append(True)
                    except ValueError:
                        passed.append(False)
                rows.append({'suite': suite, 'method': method, 'index': index, 'intent': case['intent'],
                    'supported': expected is not None, 'strict_correct': strict,
                    'file_domain_equivalent': equivalent, 'wrong_proposal': proposed and not equivalent,
                    'proposal': proposal, 'execution': execution, 'native_passed': passed})
                metrics = {}
                for s in suites:
                    for m in ('baseline', 'semantic'):
                        selected = [r for r in rows if r['suite'] == s and r['method'] == m]
                        metrics[s + '_' + m] = {'cases': len(selected),
                            'strict_correct': sum(r['strict_correct'] for r in selected),
                            'file_domain_equivalent': sum(r['file_domain_equivalent'] for r in selected),
                            'supported_equivalent': sum(r['supported'] and r['file_domain_equivalent'] for r in selected),
                            'wrong_proposals': sum(r['wrong_proposal'] for r in selected),
                            'native_passed': sum(sum(r['native_passed']) for r in selected),
                            'native_total': sum(len(r['native_passed']) for r in selected)}
                (args.output / 'raw_calls.json').write_text(json.dumps(model.raw, ensure_ascii=False, indent=2))
                (args.output / 'summary.json').write_text(json.dumps({'model': args.model, 'metrics': metrics,
                    'equivalence_domain': 'source/destination initially regular files; normalization is not universal equivalence',
                    'program_sha256': hashlib.sha256(original).hexdigest(), 'cases': rows}, ensure_ascii=False, indent=2))
                print(json.dumps({k: rows[-1][k] for k in ('suite', 'method', 'index', 'file_domain_equivalent', 'wrong_proposal')}), flush=True)
    assert original == program.read_bytes()


if __name__ == '__main__':
    main()
