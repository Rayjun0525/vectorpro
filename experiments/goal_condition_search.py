"""Experimental condition-wise vector routing; no text parser or generative LLM."""
import json
import torch
import torch.nn.functional as F
from experiments.goal_questions import profile, portable


class ConditionSearch:
    def __init__(self, memory, minimum=0.6, margin=0.2):
        self.memory, self.minimum, self.margin = memory, minimum, margin
        self.goals = list({json.dumps(e['rules'], sort_keys=True): e['rules'] for e in memory.examples if e['rules'] is not None}.values())
        self.profiles = [profile(g) for g in self.goals]
        fields = sorted(set.intersection(*(set(p) for p in self.profiles)))
        x = memory.vectors.double()
        inverse_targets = x @ x.T + 0.05 * torch.eye(len(x), dtype=torch.float64)
        self.fields = {}
        for field in fields:
            labels = [json.dumps(portable(profile(e['rules'])[field])) if e['rules'] is not None else 'null' for e in memory.examples]
            values = list(dict.fromkeys(labels))
            target = F.one_hot(torch.tensor([values.index(v) for v in labels]), num_classes=len(values)).double()
            self.fields[field] = {'values': values, 'weights': (x.T @ torch.linalg.solve(inverse_targets, target)).float()}

    def propose(self, intent):
        from vectorpro.goal_draft import review_lines
        from vectorpro.acquisition import digest
        from vectorpro.goal_evidence import validate_goal
        vector = F.normalize(self.memory.encoder([intent]).detach().cpu().float(), dim=1)[0]
        if vector.shape != self.memory.vectors[0].shape or not torch.isfinite(vector).all():
            raise ValueError('Invalid query vector')
        predicted, evidence = {}, {}
        for field, data in self.fields.items():
            scores = vector @ data['weights']
            order = scores.argsort(descending=True).tolist()
            score = float(scores[order[0]])
            gap = score - float(scores[order[1]]) if len(order) > 1 else score
            value = json.loads(data['values'][order[0]])
            evidence[field] = {'value': value, 'score': score, 'margin': gap}
            if score < self.minimum or gap < self.margin or value is None:
                return {'status': 'needs_input', 'conditions': evidence}
            predicted[field] = value
        matching = [g for g, p in zip(self.goals, self.profiles) if all(portable(p[k]) == v for k, v in predicted.items())]
        if len(matching) != 1:
            return {'status': 'needs_input', 'conditions': evidence, 'reason': 'No unique existing goal satisfies all predicted conditions'}
        goal = validate_goal({'intent': intent, 'rules': matching[0]})
        return {'status': 'needs_goal_review', 'goal': goal, 'proposal_sha256': digest(goal), 'review': review_lines(goal),
                'intent_independently_verified': False, 'conditions': evidence, 'score_is_probability': False}
