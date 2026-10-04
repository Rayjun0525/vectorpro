"""One entry point: execute known requests, otherwise learn from supplied targets.

The runtime accepts structured requests; natural-language/LLM interpretation is
an adapter to add later. Learning never guesses missing target semantics.
"""
from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from vectorpro.host import ARITIES, HostContext
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

    def provide_host_operations(self) -> None:
        """Install primitive descriptions; these are execution machinery, not learned rules."""
        for name, arity in ARITIES.items():
            provenance = {"kind": "host", "operation": name}
            if name in self.registry:
                if self.registry.get(name).provenance != provenance:
                    raise ValueError(f"host primitive name conflict: {name}")
                continue
            plan = LearningPlan(name, f"host primitive {name}", arity, OutputWidth.SAME)
            self.registry.add(Capability(plan, self.registry.build(plan, provenance), provenance))

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
                "learning_rng_state": self._rng.getstate()}
        temporary = None
        try:
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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "vectorpro-runtime" or data.get("version") != 1:
            raise ValueError("unsupported vector runtime file")
        runtime = cls(Registry.from_data(data["registry"]), config, host)
        state = data["learning_rng_state"]
        runtime._rng.setstate((state[0], tuple(state[1]), state[2]))
        return runtime
