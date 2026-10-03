"""Target values, and the only channel through which the learner sees them."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from vectorpro.tasks.base import Task

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
