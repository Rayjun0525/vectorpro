"""Run existing tensor candidates in fresh memory roots, never native request files."""
import json
import argparse
from pathlib import Path
from vectorpro.runtime import VectorRuntime
from vectorpro.host import MemoryHostContext
from vectorpro.goal_evidence import check_goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/goal_candidate_previews'))
    output = parser.parse_args().output
    if output.exists():
        raise ValueError('Preserve previous evidence')
    output.mkdir()
    program = Path('results/goal_router_execution_replay/program.pt')
    before_bytes = program.read_bytes()
    initial = VectorRuntime.load(program)
    contracts = initial.contracts()
    teachers = json.loads(Path('experiments/goal_router_training.json').read_text(encoding='utf-8'))
    goals = []
    for row in teachers:
        if row['rules'] is not None and row['rules'] not in goals:
            goals.append(row['rules'])
    observations = []
    for contract in contracts:
        names = [p['name'] for p in contract['parameters']]
        if names != ['source', 'destination'] or any(p['type'] != 'path' for p in contract['parameters']):
            continue
        for paths in (['source.bin', 'target.bin'], ['target.bin', 'source.bin']):
            cases = []
            for payload in (b'a\0\xff', b'', b'other unseen bytes'):
                files = {'source.bin': payload, 'target.bin': b'different destination', 'keep': b'preserve'}
                host = MemoryHostContext(files, directories=['empty'])
                runtime = VectorRuntime.load(program, host=host)
                result = runtime.call_contract(contract['id'], dict(zip(names, paths)), 16)
                case = {'inputs': ['source.bin', 'target.bin'], 'before': {k: v.hex() for k, v in files.items()},
                        'after': {k: v.hex() for k, v in host.files.items()}, 'before_directories': ['empty'],
                        'after_directories': sorted(host.directories)}
                cases.append(case)
            matches = []
            record = {'intent': 'preview', 'interface': {'parameters': [{'name': n} for n in names]},
                      'state_lesson': {'training': cases[:1], 'validation': cases[1:2]}, 'heldout': cases[2:]}
            for rules in goals:
                try:
                    check_goal({'intent': 'preview', 'rules': rules}, record)
                    matches.append(rules)
                except ValueError:
                    pass
            observations.append({'contract_id': contract['id'], 'arguments': dict(zip(names, paths)),
                                 'cases': cases, 'matching_goals': matches})
    assert program.read_bytes() == before_bytes
    assert all(len(r['matching_goals']) == 1 for r in observations)
    (output / 'summary.json').write_text(json.dumps({'protocol': 'actual stored tensor calls; fresh MemoryHostContext per case; no learning or native files; preview is not intent proof',
        'candidate_routes': len(observations), 'executions': sum(len(r['cases']) for r in observations), 'observations': observations}, indent=2), encoding='utf-8')
    print(json.dumps({'routes': len(observations), 'executions': sum(len(r['cases']) for r in observations), 'unchanged_program': True}))


if __name__ == '__main__':
    main()
