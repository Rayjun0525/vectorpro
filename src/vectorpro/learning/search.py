"""Search over compositions of known capabilities.

``Enumerator`` lists expressions bottom-up in order of size (number of
calls), over operators of any arity plus leaves (operands and constants). Two
expressions that produce identical bits on every probe are interchangeable,
so only the first (smallest) is kept: observational-equivalence pruning.
Candidates are evaluated batched through the capabilities' own executables;
the search itself does no arithmetic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from vectorpro.bits import BitCodec
from vectorpro.execution import BitExecutable
from vectorpro.expr import BinOp, Call, Const, Expr, Var
from vectorpro.learning.examples import ExampleSet
from vectorpro.machine.program import CONSTANTS, constant_patterns
from vectorpro.quantize import HardThreshold

SEARCH_CONSTANTS = ("zero", "one", "ones")

Values = tuple[torch.Tensor, ...]  # one (B, W) bit tensor per probe group


@dataclass(frozen=True)
class Operator:
    name: str
    executable: BitExecutable  # W -> W
    arity: int
    cost: int = 1
    commutative: bool = False
    loops: bool = False  # a vector program containing a loop


def detect_commutative(executable: BitExecutable, width: int = 8, n: int = 64, seed: int = 0) -> bool:
    """Whether swapping the two operands never changes the result on random probes."""
    g = torch.Generator().manual_seed(seed)
    a, b = (torch.randint(0, 2, (n, width), generator=g).float() for _ in range(2))
    q = HardThreshold()
    with torch.no_grad():
        return torch.equal(executable.execute(torch.stack([a, b], 1), q),
                           executable.execute(torch.stack([b, a], 1), q))


def _key(values: Values) -> bytes:
    return b"".join(v.to(torch.uint8).numpy().tobytes() for v in values)


def _node(op: Operator, args: Sequence[Expr]) -> Expr:
    return BinOp(op.name, *args) if op.arity == 2 else Call(op.name, tuple(args))


class Enumerator:
    """Yields ``(expr, values)`` for every new behaviour, smallest expressions first."""

    def __init__(
        self,
        operators: Sequence[Operator],
        leaves: Sequence[tuple[Expr, Values]],
        max_size: int = 3,
        budget: int = 2_000_000,
        chunk: int = 4096,
    ) -> None:
        self.operators = sorted(operators, key=lambda o: o.name)
        self.leaves = leaves
        self.max_size = max_size
        self.budget = budget
        self.chunk = chunk
        self.spent = 0
        self.exhausted = False

    def __iter__(self) -> Iterator[tuple[Expr, Values]]:
        quantizer = HardThreshold()
        by_size: list[list[tuple[Expr, Values]]] = [[]]
        seen: set[bytes] = set()

        def admit(expr: Expr, values: Values, bucket: list) -> bool:
            key = _key(values)
            if key in seen:
                return False
            seen.add(key)
            bucket.append((expr, values))
            return True

        for expr, values in self.leaves:
            if admit(expr, values, by_size[0]):
                yield expr, values

        n_groups = len(self.leaves[0][1])
        for size in range(1, self.max_size + 1):
            bucket: list[tuple[Expr, Values]] = []
            by_size.append(bucket)
            for op in self.operators:
                for parts in _compositions(size - 1, op.arity):
                    pools = [by_size[p] for p in parts]
                    combos = itertools.product(*(range(len(p)) for p in pools))
                    if op.commutative and op.arity == 2 and parts[0] == parts[1]:
                        combos = ((i, j) for i, j in combos if i <= j)
                    while True:
                        batch = list(itertools.islice(combos, self.chunk))
                        if not batch:
                            break
                        self.spent += len(batch) * op.cost
                        if self.spent > self.budget:
                            self.exhausted = True
                            return
                        outputs = []
                        for g in range(n_groups):
                            args = [torch.stack([pools[k][c[k]][1][g] for c in batch]) for k in range(op.arity)]
                            p, b, w = args[0].shape
                            stacked = torch.stack(args, dim=2).reshape(p * b, op.arity, w)
                            with torch.no_grad():
                                out = op.executable.execute(stacked, quantizer)
                            outputs.append(out.reshape(p, b, w))
                        for i, c in enumerate(batch):
                            expr = _node(op, [pools[k][c[k]][0] for k in range(op.arity)])
                            values = tuple(o[i] for o in outputs)
                            if admit(expr, values, bucket):
                                yield expr, values


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Ordered ways to split ``total`` into ``parts`` non-negative sizes."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def constant_leaves(widths: Sequence[tuple[int, int]]) -> list[tuple[Expr, Values]]:
    """Constant leaves for groups given as ``(batch, width)``."""
    leaves = []
    for name in SEARCH_CONSTANTS:
        row = CONSTANTS.index(name)
        leaves.append((Const(name), tuple(constant_patterns(w)[row].expand(b, w) for b, w in widths)))
    return leaves


def find_composition(
    operators: Sequence[Operator],
    arity: int,
    example_sets: Sequence[ExampleSet],
    max_size: int = 3,
    budget: int = 2_000_000,
    headrooms: Sequence[int] = (0, 1),
) -> tuple[Expr, int] | None:
    """Smallest ``(expression, headroom)`` matching every example, or None.

    With headroom ``h`` the expression is evaluated on registers ``h`` bits
    wider than the operands, and only the low ``W`` bits must match.
    """
    if not operators or not example_sets:
        return None
    targets = [BitCodec.encode(s.targets, s.width) for s in example_sets]
    for headroom in headrooms:
        operands = []
        for s in example_sets:
            x = BitCodec.encode_operands(s.operands, s.width)
            operands.append(torch.cat([x, x.new_zeros(len(s), arity, headroom)], dim=2))
        leaves = [(Var(i), tuple(o[:, i] for o in operands)) for i in range(arity)]
        leaves += constant_leaves([(len(s), s.width + headroom) for s in example_sets])
        for expr, values in Enumerator(operators, leaves, max_size, budget):
            if all(torch.equal(v[:, : s.width], t) for v, t, s in zip(values, targets, example_sets)):
                return expr, headroom
    return None


def search_composition(
    operators: Sequence[Operator],
    arity: int,
    example_sets: Sequence[ExampleSet],
    max_size: int = 3,
    budget: int = 2_000_000,
) -> Expr | None:
    """Smallest expression over ``operators`` and constants matching every example, or None."""
    found = find_composition(operators, arity, example_sets, max_size, budget, headrooms=(0,))
    return found[0] if found else None
