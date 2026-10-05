"""Label-hidden Ollama goal comparison; native effects stay in output sandboxes."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen

from experiments.assistant_intercept import compact_model_goal, intercept_goal, observation
from experiments.goal_question_benchmark import canonical
from experiments.reference_acquisition import manifests
from vectorpro.goal_draft import propose_goal
from vectorpro.goal_evidence import check_goal
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.reference_evidence import ReferenceProviders
from vectorpro.runtime import VectorRuntime


class OllamaModel:
    def __init__(self, endpoint, model):
        self.endpoint, self.model, self.raw = endpoint.rstrip('/'), model, []

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with urlopen(Request(self.endpoint + path, data=data,
                             headers={'Content-Type': 'application/json'}), timeout=600) as response:
            return json.load(response)

    def complete(self, messages, tools):
        schema = {'anyOf': [{'type': 'object', 'properties': {
            'name': {'const': t['function']['name']},
            'arguments': t['function']['parameters']},
            'required': ['name', 'arguments'], 'additionalProperties': False} for t in tools]}
        # Match the 1B grammar-constrained JSON tool envelope, using Ollama format.
        wire = [dict(m) for m in messages]
        wire[0] = {**wire[0], 'content': wire[0]['content'] + '\nReturn one JSON tool call {"name": TOOL_NAME, "arguments": ARGUMENTS}. Available tools: ' + json.dumps(tools)}
        start = monotonic()
        result = self.request('/api/chat', {'model': self.model, 'messages': wire,
            'format': schema, 'stream': False, 'think': False,
            'options': {'temperature': 0, 'seed': 0, 'num_predict': 600, 'num_ctx': 8192}})
        self.raw.append({'messages': messages, 'tools': tools, 'wire_messages': wire,
                         'response': result, 'seconds': monotonic() - start})
        try:
            call = json.loads(result['message']['content'])
            return {'role': 'assistant', 'tool_calls': [{'function': {
                'name': call['name'], 'arguments': call['arguments']}}]}
        except (ValueError, KeyError, TypeError):
            return {'role': 'assistant', 'content': result.get('message', {}).get('content', '')}


def fresh_cases():
    specifications = [
        ('새 destination에는 source의 원래 바이트를 담고, source는 그대로 남겨 줘.', 'unchanged', 'initial:source'),
        ('source가 사라지고 destination에 source의 원본 내용만 남도록 옮겨 줘.', 'absent', 'initial:source'),
        ('destination의 원래 내용을 source에 복제하되 destination은 건드리지 마.', 'initial:destination', 'unchanged'),
        ('destination 이름은 없애고 그 원본 내용을 source 쪽으로 옮겨 줘.', 'initial:destination', 'absent'),
        ('destination만 지워. source를 비롯한 나머지는 그대로 둬.', 'unchanged', 'absent'),
        ('두 파일과 다른 모든 항목에 아무 변경도 하지 마.', 'unchanged', 'unchanged'),
        ('Replace destination with an exact duplicate of source; retain the original source.', 'unchanged', 'initial:source'),
        ('Transfer the original source bytes to destination and eliminate the source entry.', 'absent', 'initial:source'),
        ('Overwrite source using the original destination bytes; retain destination.', 'initial:destination', 'unchanged'),
        ('Relocate destination into source; the former destination entry must disappear.', 'initial:destination', 'absent'),
        ('Erase destination alone and retain source without modification.', 'unchanged', 'absent'),
        ('Leave every file and directory exactly as found.', 'unchanged', 'unchanged'),
    ]
    cases = [{'intent': text, 'expected_rules': intercept_goal(text, {'decision': 'supported',
        'states': {'source': src, 'destination': dst}})['rules']} for text, src, dst in specifications]
    cases += [{'intent': text, 'expected_rules': None} for text in [
        'source를 gzip으로 압축해서 destination에 저장하고 원본은 보존해.',
        'Write only the first three bytes of source to destination.',
        'source는 삭제하면서 동시에 원래 파일이 같은 경로에 그대로 존재하게 해.',
        'Please deal with source and destination appropriately.']]
    return cases


def execute_goal(program, goal, root):
    names, paths = ['source', 'destination'], ['source.bin', 'target.bin']
    matches = []
    for contract in VectorRuntime.load(program).contracts():
        if [p['name'] for p in contract['parameters']] != names:
            continue
        for order in (paths, list(reversed(paths))):
            arguments = dict(zip(names, order))
            matched = True
            for payload in (b'preview\x00\xff', b''):
                files = {paths[0]: payload, paths[1]: b'old destination', 'keep': b'keep'}
                host = MemoryHostContext(files, directories=['empty'])
                try:
                    VectorRuntime.load(program, host=host).call_contract(contract['id'], arguments, 16)
                    check_goal(goal, observation(goal['intent'], names, paths, files, ['empty'], host))
                except (ValueError, RuntimeError):
                    matched = False
                    break
            if matched:
                matches.append({'id': contract['id'], 'arguments': arguments})
    if len(matches) != 1:
        return {'status': 'no_unique_stored_contract', 'matches': matches, 'observations': []}
    records = []
    for i, payload in enumerate((b'\x00\xffnew', b'', '독립 실행 입력'.encode())):
        directory = root / str(i)
        directory.mkdir(parents=True)
        files = {paths[0]: payload, paths[1]: b'old destination', 'keep': b'keep'}
        for path, data in files.items():
            (directory / path).write_bytes(data)
        (directory / 'empty').mkdir()
        host = HostContext(directory)
        VectorRuntime.load(program, host=host).call_contract(matches[0]['id'], matches[0]['arguments'], 16)
        records.append(observation(goal['intent'], names, paths, files, ['empty'], host))
    return {'status': 'executed', 'matches': matches, 'observations': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='http://host.docker.internal:11434')
    parser.add_argument('--model', default='gemma4:12b')
    parser.add_argument('--output', type=Path, default=Path('results/strong_goal_gemma4_12b'))
    args = parser.parse_args()
    args.output.mkdir()  # Refuse overwriting prior evidence.
    suites = {'replay': json.loads(Path('results/contrastive_encoder/validation.json').read_text()),
              'fresh': fresh_cases()}
    (args.output / 'cases.json').write_text(json.dumps(suites, ensure_ascii=False, indent=2))
    model = OllamaModel(args.endpoint, args.model)
    metadata = {'tags': model.request('/api/tags'), 'show': model.request('/api/show', {'model': args.model}),
                'version': model.request('/api/version')}
    (args.output / 'model.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    program = Path('results/goal_router_execution_replay/program.pt')
    original = program.read_bytes()
    providers = ReferenceProviders(manifests())
    rows = []
    for suite, cases in suites.items():
        for encoding in ('rules', 'compact'):
            for index, case in enumerate(cases):
                proposal = compact_model_goal(model, case['intent']) if encoding == 'compact' else propose_goal(model, providers, case['intent'])
                proposed = proposal['status'] == 'needs_goal_review'
                correct = (proposal['status'] == 'needs_input' if case['expected_rules'] is None else
                    proposed and canonical(proposal['goal']['rules']) == canonical(case['expected_rules']))
                execution = {'status': 'declined', 'observations': []}
                if proposed:
                    execution = execute_goal(program, proposal['goal'], args.output / f'{suite}_{encoding}_{index}')
                passed = []
                for record in execution['observations']:
                    try:
                        if case['expected_rules'] is None:
                            raise ValueError('unsupported request')
                        check_goal({'intent': case['intent'], 'rules': case['expected_rules']}, record)
                        passed.append(True)
                    except ValueError:
                        passed.append(False)
                row = {'suite': suite, 'encoding': encoding, 'index': index, 'intent': case['intent'],
                       'supported': case['expected_rules'] is not None, 'correct': correct,
                       'wrong_proposal': proposed and not correct, 'proposal': proposal,
                       'execution': execution, 'native_passed': passed}
                rows.append(row)
                (args.output / 'raw_calls.json').write_text(json.dumps(model.raw, ensure_ascii=False, indent=2))
                (args.output / 'summary.json').write_text(json.dumps({'model': args.model,
                    'program_sha256': hashlib.sha256(original).hexdigest(), 'cases': rows}, ensure_ascii=False, indent=2))
                print(json.dumps({k: row[k] for k in ('suite', 'encoding', 'index', 'correct', 'wrong_proposal', 'native_passed')}), flush=True)
    assert program.read_bytes() == original


if __name__ == '__main__':
    main()
