"""Target values, and the only channel through which the learner sees them."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from vectorpro.tasks.base import Task
from vectorpro.learning.plan import LearningPlan

Operands = tuple[int, ...]
TargetSource = Callable[[Operands, int], int]
"""Gives the target value for operands at a width: a human label, a trace of an
existing system, or a checker-backed generator. The learner never sees the
source itself, only the examples it asks for."""


@dataclass
class ExampleSet:
    width: int
    operands: list[Operands] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.operands)

    def accuracy(self, run: Callable[[Sequence[Operands], int], list[int]]) -> float:
        if not self.operands:
            return float("nan")
        got = run(self.operands, self.width)
        return sum(g == t for g, t in zip(got, self.targets)) / len(self.operands)


@dataclass
class ExampleLesson:
    """Finite, data-only training and validation sets; no executable teacher."""
    training: ExampleSet
    validation: ExampleSet

    def validate(self, plan: LearningPlan) -> None:
        if not plan.rounds or any(type(n) is not int or n <= 0 for n in plan.rounds):
            raise ValueError("rounds must contain positive example counts")
        if any(a >= b for a, b in zip(plan.rounds, plan.rounds[1:])):
            raise ValueError("round counts must be strictly increasing")
        if plan.arity < 1 or plan.train_width < 1 or plan.validation_width < 1:
            raise ValueError("plan arity and widths must be positive")
        for examples, width in ((self.training, plan.train_width),
                                (self.validation, plan.validation_width)):
            if examples.width != width or not examples.operands:
                raise ValueError("example sets must be nonempty and match plan widths")
            if len(examples.operands) != len(examples.targets):
                raise ValueError("every input must have exactly one target")
            if len(set(examples.operands)) != len(examples):
                raise ValueError("duplicate example inputs are not allowed")
            for operands, target in zip(examples.operands, examples.targets):
                if len(operands) != plan.arity or any(type(x) is not int or not 0 <= x < 1 << width
                                                    for x in operands):
                    raise ValueError("example inputs must match arity and unsigned width")
                if type(target) is not int or not 0 <= target < 1 << plan.output(width):
                    raise ValueError("example targets must fit the output width")
        if plan.validation_examples < 1 or len(self.validation) < plan.validation_examples:
            raise ValueError("insufficient validation examples for the plan")
        if self.training.width == self.validation.width and set(self.training.operands) & set(self.validation.operands):
            raise ValueError("training and validation inputs must be disjoint")

    def to_dict(self) -> dict:
        return {name: {"width": examples.width, "operands": examples.operands, "targets": examples.targets}
                for name, examples in (("training", self.training), ("validation", self.validation))}

    @classmethod
    def from_dict(cls, data: dict) -> ExampleLesson:
        def build(name):
            record = data[name]
            return ExampleSet(record["width"], [tuple(row) for row in record["operands"]], list(record["targets"]))
        return cls(build("training"), build("validation"))


class ExampleStream:
    """Draws fresh, distinct examples from a target source on request.

    A fraction of draws can be *boundary* inputs, which uniform sampling
    almost never produces: all operands equal, or each operand one of 0, 1,
    the top bit alone, all ones. This is task-agnostic; it protects against
    candidates that are right everywhere except on such boundaries.
    """

    def __init__(self, source: TargetSource, arity: int, rng: random.Random) -> None:
        self.source = source
        self.arity = arity
        self.rng = rng
        self._drawn: set[tuple[int, Operands]] = set()

    def _boundary(self, width: int) -> Operands:
        if self.arity > 1 and self.rng.random() < 0.5:
            return (self.rng.getrandbits(width),) * self.arity
        values = (0, 1, 1 << (width - 1), (1 << width) - 1, self.rng.getrandbits(width))
        return tuple(self.rng.choice(values) for _ in range(self.arity))

    def extend(self, examples: ExampleSet, total: int, boundary_fraction: float = 0.0) -> ExampleSet:
        """Grow ``examples`` to ``total`` distinct examples (fewer if the width runs out)."""
        width = examples.width
        capacity = 1 << (width * self.arity)
        while len(examples) < min(total, capacity):
            if boundary_fraction and self.rng.random() < boundary_fraction:
                operands = self._boundary(width)
            else:
                operands = tuple(self.rng.getrandbits(width) for _ in range(self.arity))
            if (width, operands) in self._drawn:
                continue
            self._drawn.add((width, operands))
            examples.operands.append(operands)
            examples.targets.append(self.source(operands, width))
        return examples


class ExampleTask(Task):
    """Adapts a fixed example set to the ``Task`` interface used by ``Trainer``.

    It answers only for operands that are in the set, so training cannot
    query the target source beyond what was handed over.
    """

    def __init__(self, examples: ExampleSet, output_width: Callable[[int], int], arity: int) -> None:
        self.name = "examples"
        self.arity = arity
        self._targets = dict(zip(examples.operands, examples.targets))
        self._width = examples.width
        self._output_width = output_width

    def reference(self, operands: Sequence[int], width: int) -> int:
        if width != self._width:
            raise KeyError(f"no examples at width {width}")
        return self._targets[tuple(operands)]

    def output_width(self, width: int) -> int:
        return self._output_width(width)
