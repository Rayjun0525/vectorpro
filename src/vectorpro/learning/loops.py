"""Loop search: find a ``BitFold`` body that reproduces the examples.

Bodies are enumerated by size over the registry's operators, deduplicated by
their behaviour on random probe states (as functions of operands, acc, bit
and mask). Each new body is then tried inside every fold configuration
(which operand to walk, in which direction): first on a few examples, then on
all of them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import torch

from vectorpro.bits import BitCodec
from vectorpro.expr import Const, Expr, Var, fold_expr
from vectorpro.learning.examples import ExampleSet
from vectorpro.learning.search import Enumerator, Operator, Values, constant_leaves
from vectorpro.machine.program import CONSTANTS, constant_patterns
from vectorpro.machine.skeletons import BitFold, FoldPrimitives, body_leaves
from vectorpro.quantize import HardThreshold
from vectorpro.schemas import Direction

PROBE_WIDTHS, PROBES_PER_WIDTH = (4, 8), 48
PREFILTER = 8
CHUNK = 256

# What each bit-extraction role must do, as (arity, reference). These are the
# skeleton's own requirements, the same for every task.
PRIMITIVE_ROLES: dict[str, tuple[int, Callable[..., int]]] = {
    "test_and": (2, lambda a, b, w: a & b),
    "to_bit": (2, lambda a, b, w: int(a < b)),
    "to_mask": (2, lambda a, b, w: (a - b) % (1 << w)),
    "shift_down": (1, lambda a, w: a >> 1),
    "shift_up": (1, lambda a, w: (a << 1) % (1 << w)),
}


def find_primitives(operators: Sequence[Operator], seed: int = 0) -> FoldPrimitives | None:
    """Locate, by behaviour on random probes, a capability for every bit-extraction role."""
    rng = random.Random(seed)
    width = 8
    found = {}
    for role, (arity, reference) in PRIMITIVE_ROLES.items():
        ops = [tuple(rng.getrandbits(width) for _ in range(arity)) for _ in range(48)]
        ops += [(0,) * arity, ((1 << width) - 1,) * arity]
        expected = [reference(*o, width) for o in ops]
        match = next((op for op in operators
                      if op.arity == arity and op.executable(ops, width) == expected), None)
        if match is None:
            return None
        found[role] = match.name
    return FoldPrimitives(**found)


def _probe_leaves(arity: int, seed: int) -> list[tuple[Expr, Values]]:
    """Leaf values on random states: operands, acc, a consistent (bit, mask) pair."""
    g = torch.Generator().manual_seed(seed)
    columns: list[list[torch.Tensor]] = [[] for _ in range(arity + 3)]
    for width in PROBE_WIDTHS:
        n = PROBES_PER_WIDTH
        for i in range(arity + 1):  # operands and acc
            columns[i].append(torch.randint(0, 2, (n, width), generator=g).float())
        b = torch.randint(0, 2, (n, 1), generator=g).float()
        bit = torch.cat([b, torch.zeros(n, width - 1)], dim=1)
        columns[arity + 1].append(bit)
        columns[arity + 2].append(b.expand(n, width).clone())
    leaves = [(Var(i), tuple(col)) for i, col in enumerate(columns)]
    return leaves + constant_leaves([(PROBES_PER_WIDTH, w) for w in PROBE_WIDTHS])


def simulate(fold: BitFold, ops: Mapping[str, Operator], operands: torch.Tensor) -> torch.Tensor:
    """Run a fold directly on ``(B, arity, W)`` bits through the operators (search-time check)."""
    return simulate_configs(fold.arity, fold.body, [(fold.over, fold.direction)], ops, operands)[0]


def simulate_configs(
    arity: int,
    body: Expr,
    configs: Sequence[tuple[int, Direction]],
    ops: Mapping[str, Operator],
    operands: torch.Tensor,
) -> torch.Tensor:
    """Run one body under several fold configurations at once: ``(len(configs), B, W)``."""
    batch, _, width = operands.shape
    n = len(configs)
    q = HardThreshold()
    patterns = constant_patterns(width)
    x = operands.repeat(n, 1, 1)  # rows grouped by configuration
    acc = x.new_zeros(n * batch, width)
    for step in range(width):
        cols = []
        for over, direction in configs:
            i = width - 1 - step if direction is Direction.MSB_FIRST else step
            cols.append(operands[:, over, i : i + 1])
        b = torch.cat(cols, dim=0)
        bit = torch.cat([b, b.new_zeros(n * batch, width - 1)], dim=1)
        env = [x[:, k] for k in range(arity)] + [acc, bit, b.expand(n * batch, width)]
        acc = fold_expr(
            body,
            lambda k: env[k],
            lambda o, *args: ops[o].executable.execute(torch.stack(args, dim=1), q),
            lambda name: patterns[CONSTANTS.index(name)].expand(n * batch, width),
        )
    return acc.reshape(n, batch, width)


def _uses(expr: Expr, index: int) -> bool:
    return fold_expr(expr, lambda k: k == index, lambda _, *a: any(a), lambda _: False)


class _BodyBatch:
    """Many fold bodies evaluated together.

    Sub-expressions that do not read the accumulator are shared by all bodies;
    nodes with the same operator and height run in one batched call.
    """

    def __init__(self, bodies: Sequence[Expr], acc_index: int) -> None:
        self.nodes: list[tuple[str, object, tuple[int, ...]]] = []
        index: dict[tuple, int] = {}
        reads_acc: dict[Expr, bool] = {}

        def node(e: Expr, body: int) -> int:
            if e not in reads_acc:
                reads_acc[e] = _uses(e, acc_index)
            key = (e, body if reads_acc[e] else None)
            if key not in index:
                if isinstance(e, Var):
                    n = ("acc", body, ()) if e.index == acc_index else ("var", e.index, ())
                elif isinstance(e, Const):
                    n = ("const", e.name, ())
                else:
                    n = ("op", e.op, tuple(node(a, body) for a in e.args))
                index[key] = len(self.nodes)
                self.nodes.append(n)
            return index[key]

        self.roots = [node(e, b) for b, e in enumerate(bodies)]
        height: list[int] = []
        groups: dict[tuple[int, str], list[int]] = defaultdict(list)
        for i, (kind, payload, children) in enumerate(self.nodes):
            height.append(1 + max(height[c] for c in children) if children else 0)
            if kind == "op":
                groups[(height[i], payload)].append(i)
        self.groups = sorted(groups.items())

    def evaluate(self, env: Sequence[torch.Tensor], acc: torch.Tensor, consts: Mapping[str, torch.Tensor],
                 ops: Mapping[str, Operator]) -> torch.Tensor:
        """``env``: shared leaves ``(R, W)``; ``acc``: ``(N, R, W)`` -> new acc ``(N, R, W)``."""
        q = HardThreshold()
        vals: list[torch.Tensor | None] = [None] * len(self.nodes)
        for i, (kind, payload, _) in enumerate(self.nodes):
            if kind == "var":
                vals[i] = env[payload]
            elif kind == "acc":
                vals[i] = acc[payload]
            elif kind == "const":
                vals[i] = consts[payload]
        for (_, name), ids in self.groups:
            args = torch.stack([torch.stack([vals[c] for c in self.nodes[i][2]], dim=1) for i in ids])
            g, r, a, w = args.shape
            out = ops[name].executable.execute(args.reshape(g * r, a, w), q).reshape(g, r, w)
            for k, i in enumerate(ids):
                vals[i] = out[k]
        return torch.stack([vals[r] for r in self.roots])


def _simulate_batch(
    arity: int, bodies: Sequence[Expr], configs: Sequence[tuple[int, Direction]],
    ops: Mapping[str, Operator], operands: torch.Tensor,
) -> torch.Tensor:
    """All bodies under all configurations: ``(len(bodies), len(configs), B, W)``."""
    batch, _, width = operands.shape
    n_rows = len(configs) * batch
    plan = _BodyBatch(bodies, body_leaves(arity)[0])
    patterns = constant_patterns(width)
    consts = {name: patterns[CONSTANTS.index(name)].expand(n_rows, width) for name in CONSTANTS}
    x = operands.repeat(len(configs), 1, 1)
    acc = x.new_zeros(len(bodies), n_rows, width)
    for step in range(width):
        cols = [operands[:, over, (width - 1 - step) if d is Direction.MSB_FIRST else step].unsqueeze(1)
                for over, d in configs]
        b = torch.cat(cols, dim=0)
        bit = torch.cat([b, b.new_zeros(n_rows, width - 1)], dim=1)
        env = [x[:, k] for k in range(arity)] + [None, bit, b.expand(n_rows, width)]
        acc = plan.evaluate(env, acc, consts, ops)
    return acc.reshape(len(bodies), len(configs), batch, width)


def search_bit_fold(
    operators: Sequence[Operator],
    arity: int,
    example_sets: Sequence[ExampleSet],
    max_size: int = 3,
    budget: int = 2_000_000,
    seed: int = 0,
) -> BitFold | None:
    if not operators or not example_sets:
        return None
    ops = {o.name: o for o in operators}
    encoded = [(BitCodec.encode_operands(s.operands, s.width), BitCodec.encode(s.targets, s.width))
               for s in example_sets]
    first_x, first_y = encoded[0][0][:PREFILTER], encoded[0][1][:PREFILTER]
    acc_i, bit_i, mask_i = body_leaves(arity)
    configs = [(over, d) for over in range(arity) for d in (Direction.MSB_FIRST, Direction.LSB_FIRST)]

    def check(chunk: list[Expr]) -> BitFold | None:
        quick = _simulate_batch(arity, chunk, configs, ops, first_x)
        hits = (quick == first_y).flatten(2).all(dim=2)  # (bodies, configs)
        for b, c in hits.nonzero().tolist():  # row-major: smallest body first
            fold = BitFold(arity, configs[c][0], configs[c][1], chunk[b])
            if all(torch.equal(simulate(fold, ops, x), y) for x, y in encoded):
                return fold
        return None

    pending: list[Expr] = []
    for body, _ in Enumerator(operators, _probe_leaves(arity, seed), max_size, budget):
        # A useful body reads the accumulator and the current bit; others cannot fold.
        if isinstance(body, Var) or not _uses(body, acc_i) or not (_uses(body, bit_i) or _uses(body, mask_i)):
            continue
        pending.append(body)
        if len(pending) == CHUNK:
            if found := check(pending):
                return found
            pending = []
    return check(pending) if pending else None
