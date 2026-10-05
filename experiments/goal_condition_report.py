"""Retrieval recall diagnostic on fixed cases; labels used only for scoring."""
import json
from pathlib import Path
import torch.nn.functional as F
from vectorpro.semantic_catalog import Encoder
from vectorpro.runtime import VectorRuntime
from vectorpro.goal_memory import GoalMemory
from experiments.goal_question_benchmark import canonical


def main():
    directory = Path('/opt/vectorpro-models/multilingual-minilm')
    encoder = Encoder(directory)
    identity = json.loads((directory / 'download.json').read_text())
    runtime = VectorRuntime.load('results/goal_router_execution_replay/program.pt')
    memory = GoalMemory.from_runtime(runtime, encoder, identity)
    cases = json.loads(Path('results/goal_condition_search/cases.json').read_text())
    groups = list({json.dumps(e['rules'], sort_keys=True): e['rules'] for e in memory.examples if e['rules'] is not None}.values())
    vectors = F.normalize(encoder([c['intent'] for c in cases]).detach().cpu().float(), dim=1)
    rows = []
    for case, vector in zip(cases, vectors):
        scores = (memory.vectors @ vector).tolist()
        ranked = sorted(((max(s for s, e in zip(scores, memory.examples) if e['rules'] is not None and canonical(e['rules']) == canonical(g)), g) for g in groups), key=lambda p: p[0], reverse=True)
        rank = next((i + 1 for i, (_, g) in enumerate(ranked) if case['expected_rules'] is not None and canonical(g) == canonical(case['expected_rules'])), None)
        rows.append({'intent': case['intent'], 'supported': case['expected_rules'] is not None, 'correct_goal_rank': rank,
                     'ranked_goals': [{'similarity': s, 'rules': g} for s, g in ranked]})
    report = {'protocol': 'same16 cases diagnostic; positive candidate ranking is not supportedness/intent proof; no LLM',
              'supported_total': sum(r['supported'] for r in rows),
              'recall': {str(k): sum(r['supported'] and r['correct_goal_rank'] <= k for r in rows) for k in (1, 2, 3, 6)}, 'cases': rows}
    Path('results/goal_condition_recall.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
