"""Declarative train -> verify -> evaluate runs over operand widths.

An experiment declares ``OperationSpec``s; ``run_spec`` does the rest so every
operation is measured under the same protocol and data budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch

from vectorpro.data import all_tuples, edge_pairs, random_tuples, split
from vectorpro.evaluation import EvalCase, evaluate
from vectorpro.execution import BitExecutable
from vectorpro.quantize import HardThreshold, Identity, Quantizer
from vectorpro.tasks.base import LocalRule, Task
from vectorpro.training import TrainConfig, Trainer
from vectorpro.units import iter_units
from vectorpro.verification import verify_local_rule


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    quantizer: Quantizer
    compile: bool = False

    def prepare(self, executable: BitExecutable) -> BitExecutable:
        return executable.compiled() if self.compile else executable


DEFAULT_MODES = (
    ExecutionMode("neural/continuous", Identity()),
    ExecutionMode("neural/quantized", HardThreshold()),
    ExecutionMode("table/quantized", HardThreshold(), compile=True),
)


@dataclass(frozen=True)
class WidthSuite:
    """Train on a few small-width examples; test unseen, wider and edge inputs."""

    train_width: int = 4
    n_train: int = 32
    exhaustive_width: int = 8
    sampled_widths: tuple[int, ...] = (16, 32, 64)
    n_random: int = 10_000

    def split(self, arity: int, rng: random.Random):
        return split(all_tuples(arity, self.train_width), self.n_train, rng)

    def cases(self, arity: int, unseen, rng: random.Random) -> list[EvalCase]:
        cases = [
            EvalCase(f"{self.train_width}bit_unseen", self.train_width, unseen),
            EvalCase(f"{self.exhaustive_width}bit_exhaustive", self.exhaustive_width,
                     all_tuples(arity, self.exhaustive_width)),
        ]
        for width in self.sampled_widths:
            operands = random_tuples(arity, width, self.n_random, rng)
            if arity == 2:
                operands += edge_pairs(width)
            cases.append(EvalCase(f"{width}bit_random+edge", width, operands))
        return cases


@dataclass(frozen=True)
class Derived:
    """An executable built from the trained one and evaluated with no further training."""

    build: Callable[[BitExecutable], BitExecutable]
    task: Task
    cases: Callable[[random.Random], list[EvalCase]]


@dataclass(frozen=True)
class OperationSpec:
    name: str
    task: Task
    build: Callable[[], BitExecutable]
    rules: Mapping[str, LocalRule] = field(default_factory=dict)
    suite: WidthSuite = field(default_factory=WidthSuite)
    derived: Sequence[Derived] = ()
    restarts: int = 1
    """Independent initializations; the one with the lowest training loss is kept."""


@dataclass(frozen=True)
class SeedRun:
    record: dict
    executable: BitExecutable


def verify_units(executable: BitExecutable, rules: Mapping[str, LocalRule]) -> dict[str, dict]:
    """Verify every unit that has an oracle rule; attach the record to the unit."""
    records = {}
    for unit in iter_units(executable):
        if unit.name in rules:
            unit.verification = verify_local_rule(unit.cell, rules[unit.name])
            records[unit.name] = unit.verification.to_dict()
    return records


def evaluate_modes(
    executable: BitExecutable,
    task: Task,
    cases: Sequence[EvalCase],
    modes: Sequence[ExecutionMode] = DEFAULT_MODES,
) -> dict[str, list[dict]]:
    out = {}
    for mode in modes:
        prepared = mode.prepare(executable)
        run = lambda ops, width, p=prepared, q=mode.quantizer: p(ops, width, q)  # noqa: E731
        out[mode.name] = [r.to_dict() for r in evaluate(run, task.reference, cases)]
    return out


def run_spec(
    spec: OperationSpec,
    seed: int,
    config: TrainConfig | None = None,
    modes: Sequence[ExecutionMode] = DEFAULT_MODES,
) -> SeedRun:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    suite = spec.suite
    arity = spec.task.arity

    train, unseen = suite.split(arity, rng)
    trainer = Trainer(config)
    attempts = []
    for _ in range(spec.restarts):
        candidate = spec.build()
        attempts.append((trainer.fit(candidate, spec.task, train, suite.train_width), candidate))
    report, executable = min(attempts, key=lambda a: a[0].final_loss)
    verification = verify_units(executable, spec.rules)

    evaluations = evaluate_modes(executable, spec.task, suite.cases(arity, unseen, rng), modes)
    for derived in spec.derived:
        extra = evaluate_modes(derived.build(executable), derived.task, derived.cases(rng), modes)
        for mode, results in extra.items():
            evaluations[mode] += results

    record = {
        "operation": spec.name,
        "seed": seed,
        "train": {
            "width": suite.train_width,
            "examples": len(train),
            **vars(report),
            "restart_losses": [a[0].final_loss for a in attempts],
        },
        "verification": verification,
        "evaluations": evaluations,
    }
    return SeedRun(record, executable)


def summarize(records: Sequence[dict]) -> str:
    """Per-mode case totals summed over seeds of one operation."""
    lines = [f"== {records[0]['operation']} =="]
    for unit in records[0]["verification"]:
        exact = sum(r["verification"][unit]["exact"] for r in records)
        lines.append(f"  local rule [{unit}] exact on {exact}/{len(records)} seeds")
    for mode, cases in records[0]["evaluations"].items():
        lines.append(f"  [{mode}]")
        for i, case in enumerate(cases):
            correct = sum(r["evaluations"][mode][i]["correct"] for r in records)
            total = sum(r["evaluations"][mode][i]["total"] for r in records)
            lines.append(f"    {case['name']:<22} {correct:>8}/{total:<8} {correct / total:7.2%}")
    return "\n".join(lines)

