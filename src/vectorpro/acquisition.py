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


def validate_bindings(bindings, contracts):
    from vectorpro.host import HOST_TYPES
    ids = {c["id"] for c in contracts if c["name"] not in HOST_TYPES}
    required = {"contract_id", "intent_sha256", "source_id", "source_sha256", "checked_cases"}
    if not isinstance(bindings, list) or len(bindings) > 256:
        raise ValueError("at most 256 verified request bindings")
    keys = set()
    for binding in bindings:
        if not isinstance(binding, dict) or not required <= set(binding) or set(binding) - required - {"goal_sha256"} or not isinstance(binding["contract_id"], str) or binding["contract_id"] not in ids:
            raise ValueError("request binding must reference a current contract")
        for field in ("intent_sha256", "source_sha256") + (("goal_sha256",) if "goal_sha256" in binding else ()):
            value = binding[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("request binding requires SHA-256 hex")
        if not isinstance(binding["source_id"], str) or not 1 <= len(binding["source_id"]) <= 2000 or type(binding["checked_cases"]) is not int or not 1 <= binding["checked_cases"] <= 768:
            raise ValueError("invalid request binding evidence")
        key = (binding["intent_sha256"], binding["contract_id"])
        if key in keys:
            raise ValueError("duplicate request binding")
        keys.add(key)
    return copy.deepcopy(bindings)


class EvidenceBank:
    def __init__(self, records, goal=None):
        from vectorpro.goal_evidence import validate_goal, check_goal
        self.goal = validate_goal(goal) if goal is not None else None
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
            if self.goal is not None:
                check_goal(self.goal, record)
            self._records[record["id"]] = record

    def describe(self, intent):
        return [{"id": r["id"], "intent": r["intent"], "origin": r["origin"],
                 "interface": copy.deepcopy(r["interface"])}
                for r in self._records.values() if r["intent"] == intent]

    def permits(self, capability, intent, runtime=None, contract_id=None):
        receipt = capability.provenance.get("acceptance", {})
        receipts = [receipt] if receipt and receipt.get("intent_sha256") == digest(intent) else []
        if runtime is not None:
            receipts += [b for b in runtime._intent_bindings
                         if b["intent_sha256"] == digest(intent) and b["contract_id"] == contract_id]
        for approved in receipts:
            if self.goal is not None and (self.goal["intent"] != intent or approved.get("goal_sha256") != digest(self.goal)):
                continue
            source = self._records.get(approved.get("source_id"))
            if source is None or (source["intent"] == intent and approved.get("source_sha256") == digest(source)):
                return True
        return False

    def reuse(self, runtime, intent, source_id, *, program_path=None, time_budget_seconds=10):
        """Check existing implementations against all source cases; never searches."""
        source = self._records.get(source_id)
        if source is None or source["intent"] != intent:
            return {"status": "needs_evidence", "reason": "no bound source"}
        if type(time_budget_seconds) not in (int, float) or not 0 < time_budget_seconds <= 60:
            raise ValueError("reuse time budget must be in (0,60]")
        interface = source["interface"]
        matches, checked = [], 0
        start = monotonic()
        for contract in runtime.contracts():
            cap = runtime.registry.get(contract["name"])
            params = [{k: p[k] for k in ("name", "type", "role")} for p in contract["parameters"]]
            if (cap.provenance["kind"] == "host" or params != interface["parameters"]
                    or contract["output"] != interface["output"]
                    or set(contract["execution"]["operations"]) - set(interface["allowed_operations"])):
                continue
            checked += 1
            if checked > 32 or monotonic() - start > time_budget_seconds:
                return {"status": "reuse_failed", "reason": "reuse verification budget exhausted"}
            if "state_lesson" in source:
                lessons = [state_lesson(source["state_lesson"]),
                           state_lesson({**source["state_lesson"], "validation": source["heldout"]})]
                cases = lessons[0].training + lessons[0].validation + lessons[1].validation
                passed = True
                for case in cases:
                    if monotonic() - start > time_budget_seconds:
                        return {"status": "reuse_failed", "reason": "reuse verification budget exhausted"}
                    if not evaluate(cap.executable, case, lessons[0]):
                        passed = False
                        break
                count = len(cases)
            else:
                numeric = numeric_lesson(source["lesson"])
                hidden = numeric_lesson({"training": source["lesson"]["training"], "validation": source["heldout"]})
                sets = (numeric.training, numeric.validation, hidden.validation)
                try:
                    passed = all(cap.executable(examples.operands, examples.width) == examples.targets for examples in sets)
                except (ValueError, RuntimeError, IndexError, KeyError):
                    passed = False
                count = sum(len(examples) for examples in sets)
            if passed:
                matches.append((contract, count))
        if monotonic() - start > time_budget_seconds:
            return {"status": "reuse_failed", "reason": "reuse verification budget exhausted"}
        if not matches:
            return {"status": "needs_learning_examples", "reason": "no existing contract passed current evidence"}
        if len(matches) != 1:
            return {"status": "ambiguous_reuse", "reason": "multiple contracts passed current evidence"}
        contract, count = matches[0]
        binding = {"contract_id": contract["id"], "intent_sha256": digest(intent),
                     "source_id": source_id, "source_sha256": digest(source), "checked_cases": count}
        if self.goal is not None:
            binding["goal_sha256"] = digest(self.goal)
        bindings = [b for b in runtime._intent_bindings if (b["contract_id"], b["intent_sha256"]) != (contract["id"], digest(intent))]
        bindings.append(binding)
        validate_bindings(bindings, runtime.contracts())
        staged = VectorRuntime(Registry.from_data(runtime.registry.to_data()), config=runtime.learner.config)
        staged._rng.setstate(runtime._rng.getstate())
        staged._intent_bindings = bindings
        staged._tensor_extras = runtime._tensor_extras.copy()
        if program_path is not None:
            staged.save(program_path)
        runtime._intent_bindings = copy.deepcopy(bindings)
        return {"status": "reused", "contract": contract, "binding": binding,
                "intent_independently_verified": False}

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
        staged._intent_bindings = copy.deepcopy(runtime._intent_bindings)
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
        if self.goal is not None:
            acceptance["goal_sha256"] = digest(self.goal)
        cap.provenance["acceptance"] = acceptance
        contract = staged.contract(draft["name"])
        if program_path is not None:
            staged.save(program_path)
        runtime.registry = staged.registry
        runtime.learner = Learner(runtime.registry, runtime.learner.config)
        runtime._rng.setstate(staged._rng.getstate())
        runtime._tensor_extras = {}
        runtime._intent_bindings = staged._intent_bindings
        return {**result, "contract": contract, "acceptance": acceptance,
                "intent_independently_verified": False}
