"""Caller-controlled evidence with hidden, non-search acceptance cases.

Origin labels are declarations, not certificates. Models cannot add sources or
replace their answers. Evidence is bound to the caller's exact intent string.
"""
import copy
import hashlib
import json
from time import monotonic

from vectorpro.contract_learning import validate_draft, numeric_lesson, state_lesson
from vectorpro.learning import Learner, Registry
from vectorpro.learning.stateful import evaluate
from vectorpro.runtime import VectorRuntime


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class EvidenceBank:
    def __init__(self, records):
        if not isinstance(records, list) or len(records) > 32:
            raise ValueError("evidence bank needs at most 32 records")
        self._records = {}
        for original in records:
            record = copy.deepcopy(original)
            required = {"id", "intent", "origin", "interface", "heldout"}
            if not isinstance(record, dict) or not required <= set(record) or set(record) - required - {"lesson", "state_lesson"}:
                raise ValueError("evidence record accepts identity/intent/origin/interface/lesson/heldout")
            if any(not isinstance(record[k], str) or not 1 <= len(record[k]) <= 2000 for k in ("id", "intent", "origin")):
                raise ValueError("source identity, intent and origin must be bounded strings")
            if record["id"] in self._records:
                raise ValueError("duplicate evidence identity")
            if len(json.dumps(record, allow_nan=False).encode()) > 1024 * 1024:
                raise ValueError("evidence record exceeds byte limit")
            interface = record["interface"]
            if not isinstance(interface, dict) or set(interface) != {"parameters", "output", "allowed_operations"}:
                raise ValueError("evidence interface requires parameters/output/allowed_operations")
            validate_draft({"version": 1, "name": "evidence_interface", "description": record["intent"], **interface})
            if ("lesson" in record) == ("state_lesson" in record):
                raise ValueError("source requires numeric OR state lesson")
            kinds = [p["type"] for p in interface["parameters"]]
            if "state_lesson" in record:
                source = state_lesson(record["state_lesson"])
                hidden = state_lesson({**record["state_lesson"], "validation": record["heldout"]})
                if list(source.input_types) != kinds or source.output_type != interface["output"]["type"]:
                    raise ValueError("source evidence does not match interface")
                if any(c.output is None for c in source.training + source.validation + hidden.validation):
                    raise ValueError("verified evidence requires explicit return targets")
                if {c.signature() for c in source.validation} & {c.signature() for c in hidden.validation}:
                    raise ValueError("heldout conditions overlap validation")
            else:
                if any(k != "value" for k in kinds) or interface["output"]["type"] != "value" or interface["allowed_operations"]:
                    raise ValueError("numeric evidence requires pure value interface")
                source = numeric_lesson(record["lesson"])
                hidden = numeric_lesson({"training": record["lesson"]["training"], "validation": record["heldout"]})
                if source.validation.width == hidden.validation.width and set(source.validation.operands) & set(hidden.validation.operands):
                    raise ValueError("heldout inputs overlap validation")
                # Reuse existing target bounds/arity/disjointness validation.
                from vectorpro.learning import LearningPlan, OutputWidth
                for examples in (source, hidden):
                    plan = LearningPlan("evidence_interface", record["intent"], len(kinds), OutputWidth(interface["output"]["width"]),
                        rounds=(len(examples.training),), train_width=examples.training.width,
                        validation_width=examples.validation.width, validation_examples=len(examples.validation))
                    examples.validate(plan)
            self._records[record["id"]] = record

    def describe(self, intent):
        return [{"id": r["id"], "intent": r["intent"], "origin": r["origin"],
                 "interface": copy.deepcopy(r["interface"])}
                for r in self._records.values() if r["intent"] == intent]

    def permits(self, capability, intent):
        receipt = capability.provenance.get("acceptance", {})
        if not receipt or receipt.get("intent_sha256") != digest(intent):
            return False
        source = self._records.get(receipt.get("source_id"))
        return source is None or (source["intent"] == intent and receipt.get("source_sha256") == digest(source))

    def proposal(self, intent, source_id, name):
        source = self._records.get(source_id)
        if source is None or source["intent"] != intent:
            raise ValueError("no evidence source bound to this intent")
        return validate_draft({"version": 1, "name": name, "description": intent,
                               **copy.deepcopy(source["interface"])})

    def acquire(self, runtime, intent, source_id, draft, *, program_path=None, time_budget_seconds=60):
        draft = validate_draft(draft)
        source = self._records.get(source_id)
        if source is None or source["intent"] != intent:
            return {"status": "needs_evidence", "registered": False,
                    "reason": "no caller-controlled evidence bound to this intent"}
        if any(draft[k] != source["interface"][k] for k in source["interface"]):
            raise ValueError("proposed interface must match the caller evidence interface")
        # The candidate is disposable until hidden acceptance AND persistence pass.
        staged = VectorRuntime(Registry.from_data(runtime.registry.to_data()), config=runtime.learner.config)
        staged._rng.setstate(runtime._rng.getstate())
        key = "state_lesson" if "state_lesson" in source else "lesson"
        start = monotonic()
        result = staged.teach_contract(draft, **{key: copy.deepcopy(source[key])},
                                       time_budget_seconds=time_budget_seconds)
        if result["status"] != "registered":
            return result
        cap = staged.registry.get(draft["name"])
        if key == "state_lesson":
            check = state_lesson({**source[key], "validation": source["heldout"]})
            passed = []
            for case in check.validation:
                if monotonic() - start > time_budget_seconds:
                    return {"status": "learning_failed", "registered": False, "reason": "acceptance time budget exhausted"}
                passed.append(evaluate(cap.executable, case, check))
        else:
            hidden = numeric_lesson({"training": source[key]["training"], "validation": source["heldout"]}).validation
            try:
                outputs = cap.executable(hidden.operands, hidden.width)
                passed = [a == b for a, b in zip(outputs, hidden.targets)]
                if len(outputs) != len(hidden.targets):
                    passed = [False] * len(hidden.targets)
            except (ValueError, RuntimeError, IndexError, KeyError):
                passed = [False] * len(hidden.targets)
        if not all(passed) or monotonic() - start > time_budget_seconds:
            return {"status": "learning_failed", "registered": False,
                    "reason": "hidden acceptance failed; candidate discarded"}
        acceptance = {"source_id": source_id, "source_sha256": digest(source),
                      "intent_sha256": digest(intent), "origin": source["origin"],
                      "heldout_cases": len(passed), "passed": len(passed),
                      "scope": "caller-controlled finite evidence; not universal or independent intent proof"}
        cap.provenance["acceptance"] = acceptance
        contract = staged.contract(draft["name"])
        if program_path is not None:
            staged.save(program_path)
        runtime.registry = staged.registry
        runtime.learner = Learner(runtime.registry, runtime.learner.config)
        runtime._rng.setstate(staged._rng.getstate())
        runtime._tensor_extras = {}
        return {**result, "contract": contract, "acceptance": acceptance,
                "intent_independently_verified": False}
