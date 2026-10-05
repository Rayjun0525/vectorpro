"""Caller-labelled semantic goal examples: retrieval proposes, never authorizes."""
import copy
import json
import torch
import torch.nn.functional as F
from vectorpro.acquisition import digest
from vectorpro.goal_evidence import validate_goal
from vectorpro.goal_draft import review_lines


class GoalMemory:
    def __init__(self, examples, encoder, identity, minimum=0.55, margin=0.03):
        if not isinstance(examples, list) or not 1 <= len(examples) <= 256:
            raise ValueError("goal memory needs 1..256 caller-labelled examples")
        if not isinstance(identity, dict) or not identity or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in (minimum, margin)):
            raise ValueError("goal memory requires encoder identity and bounded thresholds")
        for example in examples:
            if not isinstance(example, dict) or set(example) != {"intent", "rules"} or not isinstance(example["intent"], str) or not 1 <= len(example["intent"]) <= 2000:
                raise ValueError("goal example requires intent and rules")
            if example["rules"] is not None:
                validate_goal(example)
        self.examples, self.encoder, self.identity = copy.deepcopy(examples), encoder, copy.deepcopy(identity)
        self.minimum, self.margin = minimum, margin
        vectors = encoder([e["intent"] for e in examples]).detach().cpu().float()
        if vectors.ndim != 2 or vectors.shape[0] != len(examples) or not 1 <= vectors.shape[1] <= 2048 or not torch.isfinite(vectors).all() or (vectors.norm(dim=1) < 1e-8).any():
            raise ValueError("invalid goal embedding matrix")
        self.vectors = F.normalize(vectors, dim=1)

    def to_data(self):
        return {"version": 1, "examples": copy.deepcopy(self.examples), "vectors": self.vectors.tolist(),
                "identity": copy.deepcopy(self.identity), "minimum": self.minimum, "margin": self.margin}

    @staticmethod
    def validate_data(data):
        if not isinstance(data, dict) or set(data) != {"version", "examples", "vectors", "identity", "minimum", "margin"} or data["version"] != 1:
            raise ValueError("invalid goal memory payload")
        class SavedEncoder:
            def __call__(self, texts):
                return torch.tensor(data["vectors"], dtype=torch.float32)
        memory = GoalMemory(data["examples"], SavedEncoder(), data["identity"], data["minimum"], data["margin"])
        if not 1 <= memory.vectors.shape[1] <= 2048:
            raise ValueError("goal vector width outside bounds")
        return copy.deepcopy(data)

    def attach(self, runtime):
        runtime._goal_memory = self.validate_data(self.to_data())

    @classmethod
    def from_runtime(cls, runtime, encoder, identity):
        data = cls.validate_data(runtime._goal_memory)
        if identity != data["identity"]:
            raise ValueError("goal memory encoder identity mismatch")
        memory = object.__new__(cls)
        memory.examples, memory.encoder, memory.identity = data["examples"], encoder, data["identity"]
        memory.minimum, memory.margin = data["minimum"], data["margin"]
        memory.vectors = F.normalize(torch.tensor(data["vectors"], dtype=torch.float32), dim=1)
        return memory

    def propose(self, intent):
        if not isinstance(intent, str) or not 1 <= len(intent) <= 2000:
            raise ValueError("goal memory requires bounded intent")
        vector = F.normalize(self.encoder([intent]).detach().cpu().float(), dim=1)
        if vector.shape != (1, self.vectors.shape[1]) or not torch.isfinite(vector).all():
            raise ValueError("invalid query embedding")
        scores = (self.vectors @ vector[0]).tolist()
        groups = {}
        for index, (example, score) in enumerate(zip(self.examples, scores)):
            key = digest(example["rules"])
            if key not in groups or score > groups[key][0]:
                groups[key] = (score, index)
        ranked = sorted(groups.values(), reverse=True)
        score, index = ranked[0]
        gap = score - ranked[1][0] if len(ranked) > 1 else score
        evidence = {"example_sha256": digest(self.examples[index]), "similarity": score, "margin": gap,
                    "encoder_identity": self.identity, "source": "caller-labelled goal examples; not independent intent proof"}
        rules = self.examples[index]["rules"]
        if score < self.minimum or gap < self.margin or rules is None:
            return {"status": "needs_input", "question": "확인된 예제와 충분히 구별되는 목표가 필요합니다.", "retrieval": evidence}
        goal = validate_goal({"intent": intent, "rules": copy.deepcopy(rules)})
        return {"status": "needs_goal_review", "goal": goal, "proposal_sha256": digest(goal),
                "review": review_lines(goal), "intent_independently_verified": False, "retrieval": evidence}
