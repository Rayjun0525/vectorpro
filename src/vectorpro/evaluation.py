"""Evaluation harness, agnostic to what is being executed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from vectorpro.data import Operands

Executor = Callable[[Sequence[Operands], int], list[int]]
Reference = Callable[[Operands, int], int]


@dataclass(frozen=True)
class EvalCase:
    name: str
    width: int
    operands: Sequence[Operands]


@dataclass(frozen=True)
class CaseResult:
    name: str
    width: int
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else float("nan")

    def to_dict(self) -> dict:
        return {**asdict(self), "accuracy": self.accuracy}


def evaluate(
    executor: Executor,
    reference: Reference,
    cases: Sequence[EvalCase],
    batch_size: int = 16384,
) -> list[CaseResult]:
    results = []
    for case in cases:
        correct = 0
        for start in range(0, len(case.operands), batch_size):
            batch = case.operands[start : start + batch_size]
            got = executor(batch, case.width)
            correct += sum(g == reference(t, case.width) for g, t in zip(got, batch))
        results.append(CaseResult(case.name, case.width, len(case.operands), correct))
    return results
