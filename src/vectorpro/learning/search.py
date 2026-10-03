"""Composition search: find an expression over known capabilities that fits examples.

Bottom-up enumeration by depth with observational-equivalence pruning: two
expressions that produce identical bits on every example are interchangeable
for this search, so only the first (smallest) is kept. Candidates are run
batched through the capabilities' own executables; no arithmetic is done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from vectorpro.bits import BitCodec
from vectorpro.execution import BitExecutable
from vectorpro.expr import BinOp, Expr, Var
from vectorpro.learning.examples import ExampleSet
from vectorpro.quantize import HardThreshold


@dataclass(frozen=True)
class _Entry:
    expr: Expr
    values: tuple[torch.Tensor, ...]  # one (B, W) tensor per example set
    depth: int


def _key(values: tuple[torch.Tensor, ...]) -> bytes:
    return b"".join(v.to(torch.uint8).numpy().tobytes() for v in values)


def search_composition(
    ops: Mapping[str, BitExecutable],
    arity: int,
    example_sets: Sequence[ExampleSet],
    max_depth: int = 2,
    max_pairs: int = 50_000,
) -> Expr | None:
    """Smallest expression over ``ops`` (``W -> W`` binary) matching every example, or None."""
    if not ops or not example_sets:
        return None
    quantizer = HardThreshold()
    operands = [BitCodec.encode_operands(s.operands, s.width) for s in example_sets]
    targets = tuple(BitCodec.encode(s.targets, s.width) for s in example_sets)
    target_key = _key(targets)

    entries: list[_Entry] = []
    seen: set[bytes] = set()

    def admit(expr: Expr, values: tuple[torch.Tensor, ...], depth: int) -> bool:
        key = _key(values)
        if key in seen:
            return False
        seen.add(key)
        entries.append(_Entry(expr, values, depth))
        return key == target_key

    for i in range(arity):
        if admit(Var(i), tuple(o[:, i] for o in operands), 0):
            return Var(i)

    for depth in range(1, max_depth + 1):
        snapshot = list(entries)
        pairs = [(a, b) for a in snapshot for b in snapshot if max(a.depth, b.depth) == depth - 1]
        for start in range(0, len(pairs), max_pairs):
            chunk = pairs[start : start + max_pairs]
            for name in sorted(ops):
                outputs = []
                for g, s in enumerate(example_sets):
                    left = torch.stack([a.values[g] for a, _ in chunk])  # (P, B, W)
                    right = torch.stack([b.values[g] for _, b in chunk])
                    stacked = torch.stack([left, right], dim=2).reshape(-1, 2, s.width)
                    out = ops[name].execute(stacked, quantizer)
                    outputs.append(out.reshape(len(chunk), -1, s.width))
                for p, (a, b) in enumerate(chunk):
                    expr = BinOp(name, a.expr, b.expr)
                    if admit(expr, tuple(o[p] for o in outputs), depth):
                        return expr
    return None
