"""The learner: turns a plan plus a stream of examples into a registered capability.

Each round the learner gets more examples and tries, cheapest first:

1. **reuse**: an expression over capabilities already in the registry;
2. **learn**: a new unit, trying generic structures simplest first and
   training each from the examples (final results only).

A candidate is accepted when its table-compiled form reproduces every training
and validation example. The learner never sees the target source, the task's
reference function or any local rule, only the examples it was handed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from vectorpro.cells import CellSignature, MLPCell
from vectorpro.execution import BitExecutable
from vectorpro.expr import expr_to_data, render
from vectorpro.learning.examples import ExampleSet, ExampleStream, ExampleTask, TargetSource
from vectorpro.learning.plan import LearningPlan, OutputWidth
from vectorpro.learning.registry import (
    Capability,
    Registry,
    composition_provenance,
    describe_schema,
    unit_provenance,
)
from vectorpro.learning.search import search_composition
from vectorpro.schemas import Direction, MapSchema, ScanSchema
from vectorpro.training import TrainConfig, Trainer
from vectorpro.units import FunctionUnit

Probe = Callable[[BitExecutable], dict]
"""Experimenter-only measurement of a candidate (e.g. accuracy on wide inputs).
Its result is logged and never influences the learner's decisions."""


@dataclass(frozen=True)
class StructureCandidate:
    name: str
    build: Callable[[], FunctionUnit]


def structure_candidates(name: str, arity: int, output: OutputWidth) -> list[StructureCandidate]:
    """Generic, task-agnostic unit structures for a shape, simplest first."""
    candidates: list[StructureCandidate] = []
    if output is OutputWidth.SAME:
        candidates.append(StructureCandidate(
            "map", lambda: FunctionUnit(name, MapSchema(arity), MLPCell(CellSignature(arity, 0, 1)))))
    layouts = {
        OutputWidth.SAME: (1, False),      # one output bit per position
        OutputWidth.PLUS_ONE: (1, True),   # ... plus the final state bit
        OutputWidth.BIT: (0, True),        # only the final state bit
    }
    if output in layouts:
        n_out, emit = layouts[output]
        for direction in Direction:
            schema = ScanSchema(arity, (0,), direction, emit)
            candidates.append(StructureCandidate(
                f"scan-{direction.value}",
                lambda s=schema: FunctionUnit(name, s, MLPCell(CellSignature(arity, 1, n_out)))))
    return candidates


@dataclass(frozen=True)
class LearnerConfig:
    train: TrainConfig = field(default_factory=lambda: TrainConfig(steps=1500))
    restarts: int = 2
    max_depth: int = 2


@dataclass
class Attempt:
    strategy: str
    structure: str | None = None
    executable: BitExecutable | None = None
    provenance: dict | None = None
    train_accuracy: float = 0.0
    validation_accuracy: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.train_accuracy == 1.0 and self.validation_accuracy == 1.0


@dataclass
class LearningOutcome:
    plan: LearningPlan
    capability: Capability | None
    history: list[dict]

    @property
    def learned(self) -> bool:
        return self.capability is not None


class Learner:
    def __init__(self, registry: Registry, config: LearnerConfig | None = None) -> None:
        self.registry = registry
        self.config = config or LearnerConfig()

    def learn(
        self,
        plan: LearningPlan,
        source: TargetSource,
        rng: random.Random,
        probe: Probe | None = None,
    ) -> LearningOutcome:
        stream = ExampleStream(source, plan.arity, rng)
        validation = stream.extend(ExampleSet(plan.validation_width), plan.validation_examples)
        train = ExampleSet(plan.train_width)
        history: list[dict] = []
        for round_no, total in enumerate(plan.rounds, start=1):
            stream.extend(train, total)
            attempt = self._attempt(plan, train, validation)
            entry = {
                "round": round_no,
                "train_examples": len(train),
                "strategy": attempt.strategy,
                "structure": attempt.structure,
                "train_accuracy": attempt.train_accuracy,
                "validation_accuracy": attempt.validation_accuracy,
                "accepted": attempt.accepted,
            }
            if probe and attempt.executable is not None:
                entry["probe"] = probe(attempt.executable)
            history.append(entry)
            if attempt.accepted:
                capability = Capability(plan, attempt.executable, attempt.provenance, history)
                self.registry.add(capability)
                return LearningOutcome(plan, capability, history)
        return LearningOutcome(plan, None, history)

    # -- strategies ---------------------------------------------------------------------

    def _attempt(self, plan: LearningPlan, train: ExampleSet, validation: ExampleSet) -> Attempt:
        reused = self._reuse(plan, train, validation)
        if reused is not None:
            return reused
        return self._learn_unit(plan, train, validation)

    def _reuse(self, plan: LearningPlan, train: ExampleSet, validation: ExampleSet) -> Attempt | None:
        if plan.output is not OutputWidth.SAME:
            return None
        expr = search_composition(
            self.registry.operators(), plan.arity, [train, validation], self.config.max_depth
        )
        if expr is None:
            return None
        provenance = composition_provenance(expr_to_data(expr))
        return self._score(Attempt("reuse", render(expr)), plan, provenance, train, validation)

    def _learn_unit(self, plan: LearningPlan, train: ExampleSet, validation: ExampleSet) -> Attempt:
        best = Attempt("learn")
        task = ExampleTask(train, plan.output, plan.arity)
        trainer = Trainer(self.config.train)
        for candidate in structure_candidates(plan.name, plan.arity, plan.output):
            for _ in range(self.config.restarts):
                unit = candidate.build()
                trainer.fit(unit, task, train.operands, train.width)
                attempt = self._score(
                    Attempt("learn", describe_schema(unit.schema.to_spec())),
                    plan, unit_provenance(unit), train, validation,
                )
                if attempt.accepted:
                    return attempt
                if (attempt.validation_accuracy, attempt.train_accuracy) > (
                    best.validation_accuracy, best.train_accuracy
                ):
                    best = attempt
        return best

    def _score(
        self, attempt: Attempt, plan: LearningPlan, provenance: dict,
        train: ExampleSet, validation: ExampleSet,
    ) -> Attempt:
        """Rebuild the candidate from its stored form and measure it on the examples."""
        executable = self.registry.build(plan, provenance)
        attempt.executable = executable
        attempt.provenance = provenance
        attempt.train_accuracy = train.accuracy(executable)
        attempt.validation_accuracy = validation.accuracy(executable)
        return attempt
