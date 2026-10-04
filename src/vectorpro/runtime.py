"""One entry point: execute known requests, otherwise learn from supplied targets.

The runtime accepts structured requests; optional natural-language/LLM interpretation
lives in the agent adapter. Learning never guesses missing target semantics.
"""
from __future__ import annotations

import json
import random
import tempfile
from time import monotonic
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from vectorpro.host import ARITIES, BASE_OPERATIONS, HostContext
from vectorpro.learning import Capability, Learner, LearnerConfig, LearningPlan, OutputWidth, Registry
from vectorpro.learning.examples import ExampleLesson, Operands, TargetSource


@dataclass
class RequestResult:
    status: str
    outputs: list[int] | None = None
    capability: str | None = None
    history: list[dict] = field(default_factory=list)


class VectorRuntime:
    def __init__(self, registry: Registry | None = None, config: LearnerConfig | None = None,
                 host: HostContext | None = None, seed: int = 0):
        self.registry = registry if registry is not None else Registry(seed=seed)
        self.learner = Learner(self.registry, config)
        self.host = host
        self._rng = random.Random(seed)
        self._tensor_extras = {}

    def provide_host_operations(self, operations: Sequence[str] | None = None) -> None:
        """Install primitive descriptions; these are execution machinery, not learned rules."""
        names = list(BASE_OPERATIONS if operations is None else operations)
        if any(name not in ARITIES for name in names):
            raise ValueError("unknown host operation")
        for name in names:
            arity = ARITIES[name]
            provenance = {"kind": "host", "operation": name}
            if name in self.registry:
                if self.registry.get(name).provenance != provenance:
                    raise ValueError(f"host primitive name conflict: {name}")
                continue
            plan = LearningPlan(name, f"host primitive {name}", arity, OutputWidth.SAME)
            self.registry.add(Capability(plan, self.registry.build(plan, provenance), provenance))

    def contract(self, name: str) -> dict:
        """Export a stored function's common contract without a model or encoder."""
        from vectorpro.contracts import build_contract
        return build_contract(self.registry, name)

    def contracts(self) -> list[dict]:
        return [self.contract(c.name) for c in self.registry]

    def teach_contract(self, draft: dict, *, lesson: dict | None = None, state_lesson: dict | None = None,
                       evidence_source: str = "caller_examples", time_budget_seconds: float = 60) -> dict:
        """Acquire in a detached registry; register only after examples AND draft bounds pass."""
        import math
        from vectorpro.contract_learning import validate_draft, numeric_lesson, state_lesson as parse_state
        from vectorpro.learning import OutputWidth
        draft = validate_draft(draft)
        if evidence_source not in ("caller_examples","llm_proposed_examples"):
            raise ValueError("evidence source is a label, not an independent correctness certificate")
        if (type(time_budget_seconds) not in (int,float) or not math.isfinite(time_budget_seconds)
                or not 0<time_budget_seconds<=60):
            raise ValueError("contract learning time budget must be finite and in (0,60]")
        if draft["name"] in self.registry:
            raise ValueError("draft name already exists; query and call its contract explicitly")
        if lesson is not None and state_lesson is not None:
            raise ValueError("provide numeric OR state evidence")
        if lesson is None and state_lesson is None:
            return {"status":"needs_learning_examples", "draft":draft, "registered":False}
        deadline = monotonic() + time_budget_seconds
        kinds = [p["type"] for p in draft["parameters"]]
        if lesson is not None:
            if any(k!="value" for k in kinds) or draft["output"]["type"]!="value" or draft["allowed_operations"]:
                raise ValueError("numeric lessons require pure value input/output contracts")
            examples=numeric_lesson(lesson)
            plan=LearningPlan(draft["name"],draft["description"],len(kinds),OutputWidth(draft["output"]["width"]),
                              rounds=(len(examples.training),),train_width=examples.training.width,
                              validation_width=examples.validation.width,validation_examples=len(examples.validation))
            examples.validate(plan)
            state=None
        else:
            state=parse_state(state_lesson)
            if state.output_type != draft["output"]["type"]:
                raise ValueError("state output type must exactly match the draft")
            if list(state.input_types)!=kinds:
                raise ValueError("state evidence input types must exactly match the draft")
            state=replace(state,time_budget_seconds=min(state.time_budget_seconds,time_budget_seconds),
                          allowed_operations=tuple(draft["allowed_operations"]))
            examples=plan=None
        config=self.learner.config
        bounded=replace(config, search_budget=min(config.search_budget,50000), search_size=min(config.search_size,3),
                        restarts=min(config.restarts,2),train=replace(config.train,steps=min(config.train.steps,1500)))
        staged=VectorRuntime(Registry.from_data(self.registry.to_data()),config=bounded)
        staged._rng.setstate(self._rng.getstate())
        if state is not None:
            staged.provide_host_operations(draft["allowed_operations"])
        outcome=staged.teach(draft["name"],plan=plan,lesson=examples,state_lesson=state)
        if outcome.status!="learned":
            return {"status":"learning_failed","registered":False,"reason":"no candidate passed supplied examples", "history":outcome.history}
        actual=staged.contract(draft["name"])
        reason=None
        if monotonic()>deadline:
            reason="contract learning time budget exhausted"
        elif actual["input_types"]!=kinds or actual["output"]!=draft["output"]:
            reason="candidate input/output type or width does not match draft"
        elif set(actual["execution"]["operations"])-set(draft["allowed_operations"]):
            reason="candidate uses operations outside the draft allowance"
        if reason:
            return {"status":"learning_failed","registered":False,"reason":reason,"history":outcome.history}
        cap=staged.registry.get(draft["name"])
        cap.provenance["contract_draft"]=draft
        cap.provenance["evidence_source"]=evidence_source
        contract=staged.contract(draft["name"])
        # Whole staged registry is accepted atomically; failed staging never modifies live state.
        self.registry=staged.registry
        self.learner=Learner(self.registry,config)
        self._rng.setstate(staged._rng.getstate())
        self._tensor_extras={}
        return {"status":"registered","registered":True,"contract":contract,"history":outcome.history,
                "evidence_source":evidence_source,"intent_independently_verified":False}

    def call_contract(self, contract_id: str, arguments: dict, width: int, *, version: int = 1) -> dict:
        """Strict named/typed, single-lane execution of an exact stored function ID.

        No search, teaching, raw handles, LLM or inference of absent arguments.
        Validate every portable argument before allocating transient buffers.
        """
        from vectorpro.host import MemoryHostContext
        if type(version) is not int or version != 1:
            raise ValueError("unsupported contract version")
        contract = next((c for c in self.contracts() if c["id"] == contract_id), None)
        if contract is None:
            raise KeyError("unknown or stale contract ID; query contracts again")
        if type(width) is not int or not 1 <= width <= 64:
            raise ValueError("contract width must be an integer in 1..64")
        fields = {p["name"] for p in contract["parameters"]}
        if not isinstance(arguments, dict) or set(arguments) != fields:
            raise ValueError("arguments must provide exactly the contract's named inputs")
        if contract["output"]["type"] not in ("value", "path", "buffer"):
            raise ValueError("legacy capability has no portable output type; use the legacy API")
        portable = []
        for parameter in contract["parameters"]:
            value = arguments[parameter["name"]]
            kind = parameter["type"]
            if kind == "value":
                if type(value) is not int or not 0 <= value < 1 << width:
                    raise ValueError("numeric arguments must be unsigned integers fitting the register width")
            else:
                if not isinstance(value, str):
                    raise ValueError("path/buffer inputs must be portable strings, never raw handles")
                if kind == "path":
                    MemoryHostContext.normalize(value)
                    value = value.encode("utf-8")
                else:
                    if len(value) % 2 or any(c not in "0123456789abcdefABCDEF" for c in value):
                        raise ValueError("buffer input must contain hexadecimal byte pairs")
                    value = bytes.fromhex(value)
                if self.host is None:
                    raise ValueError("path/buffer inputs require a host context")
                if len(value) > self.host.max_buffer_bytes:
                    raise ValueError("portable input exceeds the host byte limit")
            portable.append(value)
        if contract["execution"]["requires_host"] and self.host is None:
            raise ValueError("effectful contracts require a host context")
        handles = sum(isinstance(v, bytes) for v in portable)
        if handles and self.host._next_handle + handles - 1 >= 1 << width:
            raise ValueError("register width cannot represent portable input handles")
        # Native paths (including symlinks) are checked before the vector program runs.
        for parameter, value in zip(contract["parameters"], portable):
            if parameter["type"] == "path" and not isinstance(self.host, MemoryHostContext):
                path = (self.host.root / value.decode("utf-8")).resolve()
                if not path.is_relative_to(self.host.root):
                    raise ValueError("file path escapes the host root")
        rows = [tuple(self.host.put(v) if isinstance(v, bytes) else v for v in portable)]
        event_start = len(self.host.events) if self.host else 0
        result = self.request(contract["name"], rows, width)
        outputs = result.outputs
        if contract["output"]["type"] == "buffer":
            outputs = [{"hex": bytes(self.host.buffers[h]).hex()} for h in outputs]
        elif contract["output"]["type"] == "path":
            outputs = [{"utf8": bytes(self.host.buffers[h]).decode("utf-8")} for h in outputs]
        return {"status": result.status, "contract_id": contract_id, "contract_version": version,
                "capability": result.capability, "outputs": outputs,
                "effects": self.host.events[event_start:] if self.host else []}

    def request(self, name: str, operands: Sequence[Operands], width: int, *,
                plan: LearningPlan | None = None, source: TargetSource | None = None,
                lesson: ExampleLesson | None = None, state_lesson=None) -> RequestResult:
        if state_lesson is not None and (source is not None or lesson is not None):
            raise ValueError("state lessons cannot be combined with numeric teaching sources")
        if state_lesson is not None and (self.host is None or plan is not None):
            raise ValueError("state requests require a host context and use the lesson's shape")
        if source is not None and lesson is not None:
            raise ValueError("supply either an example lesson or a target source")
        if width < 1:
            raise ValueError("width must be positive")
        rows = [tuple(row) for row in operands]
        if not rows:
            raise ValueError("request must contain at least one operand tuple")
        arity = len(rows[0])
        if any(len(row) != arity or any(not isinstance(x, int) or x < 0 or x >= 1 << width
                                      for x in row) for row in rows):
            raise ValueError("operands must have equal arity and fit the unsigned register width")
        if plan is not None and (plan.name != name or plan.arity != arity):
            raise ValueError("learning plan must match request name and arity")
        history = []
        status = "executed"
        if name not in self.registry:
            if state_lesson is not None:
                from vectorpro.learning.stateful import learn_stateful
                if len(state_lesson.input_types) != arity:
                    raise ValueError("state lesson must match request arity")
                self.provide_host_operations()
                outcome = learn_stateful(self.registry, name, state_lesson)
                if outcome.capability is None:
                    return RequestResult("learning_failed", history=outcome.history)
                history = outcome.history
                status = "learned_and_executed"
        if name not in self.registry:
            if plan is None or (source is None and lesson is None):
                return RequestResult("needs_learning_examples")
            outcome = (self.learner.learn_examples(plan, lesson, self._rng) if lesson is not None
                       else self.learner.learn(plan, source, self._rng))
            if not outcome.learned:
                return RequestResult("learning_failed", history=outcome.history)
            history = outcome.history
            status = "learned_and_executed"
        capability = self.registry.get(name)
        if capability.plan.arity != arity or (plan is not None and plan.output != capability.plan.output):
            raise ValueError("request shape does not match the stored capability")
        if capability.executable.effects:
            if self.host is None:
                raise RuntimeError("effectful requests require a HostContext")
            if len(rows) != 1:
                raise ValueError("effectful requests require a single execution lane")
            with self.host.activate():
                outputs = capability.run(rows, width)
        else:
            outputs = capability.run(rows, width)
        return RequestResult(status, outputs, name, history)

    def teach(self, name: str, *, plan=None, lesson=None, state_lesson=None) -> RequestResult:
        """Acquire a capability in isolation, without running it on native inputs."""
        if name in self.registry:
            return RequestResult("already_known", capability=name)
        if state_lesson is not None:
            if plan is not None or lesson is not None:
                raise ValueError("state teaching cannot include numeric lessons/plans")
            from vectorpro.learning.stateful import learn_stateful
            state_lesson.validate()
            self.provide_host_operations()
            outcome = learn_stateful(self.registry, name, state_lesson)
            accepted = outcome.capability is not None
        else:
            if plan is None or lesson is None or plan.name != name:
                raise ValueError("numeric teaching requires matching plan and examples")
            outcome = self.learner.learn_examples(plan, lesson, self._rng)
            accepted = outcome.learned
        return RequestResult("learned" if accepted else "learning_failed",
                             capability=name if accepted else None, history=outcome.history)

    def save(self, path: Path) -> None:
        """Atomically save all capability tensors in one portable data file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": "vectorpro-runtime", "version": 1, "registry": self.registry.to_data(),
                "learning_rng_state": self._rng.getstate(), "contracts": self.contracts()}
        temporary = None
        try:
            if path.suffix == ".pt":
                import torch
                from vectorpro.tensor_codec import encode_tree
                if self._tensor_extras:
                    from vectorpro.semantic_catalog import fingerprint
                    cached = self._tensor_extras.get("registry_digest")
                    if cached is None or not torch.equal(cached, fingerprint(self.registry)):
                        self._tensor_extras = {}  # changed registry: never persist a stale search index
                with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                                 prefix=path.name + ".", delete=False) as stream:
                    temporary = Path(stream.name)
                    torch.save({"version": torch.tensor([1]), **encode_tree(data), **self._tensor_extras}, stream)
            else:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                 prefix=path.name + ".", delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(data, stream)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, *, host: HostContext | None = None,
             config: LearnerConfig | None = None) -> VectorRuntime:
        extras = {}
        if Path(path).suffix == ".pt":
            import torch
            from vectorpro.tensor_codec import decode_tree
            archive = torch.load(path, weights_only=True, map_location="cpu")
            if not isinstance(archive, dict) or not all(isinstance(v, torch.Tensor) for v in archive.values()):
                raise ValueError("tensor runtime payload must contain only tensors")
            if archive.get("version", torch.tensor([])).tolist() != [1]:
                raise ValueError("unsupported tensor runtime version")
            data = decode_tree(archive)
            extras = {k: v for k, v in archive.items() if k not in ("version", "nodes", "bytes", "floats")}
        else:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "vectorpro-runtime" or data.get("version") != 1:
            raise ValueError("unsupported vector runtime file")
        runtime = cls(Registry.from_data(data["registry"]), config, host)
        runtime._tensor_extras = extras
        state = data["learning_rng_state"]
        runtime._rng.setstate((state[0], tuple(state[1]), state[2]))
        if "contracts" in data and data["contracts"] != runtime.contracts():
            raise ValueError("saved contracts do not match the stored implementations")
        return runtime
