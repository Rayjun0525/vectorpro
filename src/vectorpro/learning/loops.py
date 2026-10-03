"""Loop search: find a ``BitFold`` that reproduces the examples.

Bodies are enumerated by size over the registry's operators, deduplicated by
their behaviour on random probe states (as functions of operands, acc, bit,
mask and companions). New bodies are tried in batches inside every fold
configuration (operand walked, direction, register headroom): first on a few
examples, then on all of them.

Two phases:

1. single-accumulator folds;
2. folds with one **companion** taken from the library of loops already
   learned: the companion keeps updating as it did in its own fold, and only
   the result body is searched. This is how a loop whose state needs two
   registers (e.g. quotient and remainder) is reached without searching both
   bodies at once.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import torch

from vectorpro.bits import BitCodec
from vectorpro.expr import BinOp, Const, Expr, Var, fold_expr
from vectorpro.learning.examples import ExampleSet
from vectorpro.learning.search import Enumerator, Operator, Values, constant_leaves
from vectorpro.machine.program import CONSTANTS, constant_patterns
from vectorpro.machine.skeletons import BitFold, FoldPrimitives, body_leaves, companion_leaf
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


def _probe_leaves(arity: int, seed: int, n_companions: int = 0) -> list[tuple[Expr, Values]]:
    """Leaf values on random states: operands, acc, a consistent (bit, mask) pair, companions."""
    g = torch.Generator().manual_seed(seed)
    n_leaves = arity + 3 + n_companions
    columns: list[list[torch.Tensor]] = [[] for _ in range(n_leaves)]
    for width in PROBE_WIDTHS:
        n = PROBES_PER_WIDTH
        for i in [*range(arity + 1), *range(arity + 3, n_leaves)]:  # operands, acc, companions
            columns[i].append(torch.randint(0, 2, (n, width), generator=g).float())
        b = torch.randint(0, 2, (n, 1), generator=g).float()
        columns[arity + 1].append(torch.cat([b, torch.zeros(n, width - 1)], dim=1))
        columns[arity + 2].append(b.expand(n, width).clone())
    leaves = [(Var(i), tuple(col)) for i, col in enumerate(columns)]
    return leaves + constant_leaves([(PROBES_PER_WIDTH, w) for w in PROBE_WIDTHS])


def _uses(expr: Expr, index: int) -> bool:
    return fold_expr(expr, lambda k: k == index, lambda _, *a: any(a), lambda _: False)


class _BodyBatch:
    """Many fold bodies evaluated together.

    Sub-expressions that do not read the accumulator are shared by all bodies;
    nodes with the same operator and height run in one batched call. With
    ``acc_index=None`` every leaf is shared (used for companions).
    """

    def __init__(self, bodies: Sequence[Expr], acc_index: int | None) -> None:
        self.nodes: list[tuple[str, object, tuple[int, ...]]] = []
        index: dict[tuple, int] = {}
        reads_acc: dict[Expr, bool] = {}

        def node(e: Expr, body: int) -> int:
            if e not in reads_acc:
                reads_acc[e] = acc_index is not None and _uses(e, acc_index)
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

    def evaluate(self, env: Sequence[torch.Tensor | None], acc: torch.Tensor | None,
                 consts: Mapping[str, torch.Tensor], ops: Mapping[str, Operator]) -> torch.Tensor:
        """``env``: shared leaves ``(R, W)``; ``acc``: per-body ``(N, R, W)`` -> roots ``(N, R, W)``."""
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
    arity: int,
    bodies: Sequence[Expr],
    configs: Sequence[tuple[int, Direction]],
    ops: Mapping[str, Operator],
    operands: torch.Tensor,
    headroom: int = 0,
    companions: Sequence[Expr] = (),
) -> torch.Tensor:
    """All bodies under all configurations: ``(len(bodies), len(configs), B, W)``."""
    batch, _, out_width = operands.shape
    width = out_width + headroom
    if headroom:
        operands = torch.cat([operands, operands.new_zeros(batch, arity, headroom)], dim=2)
    n_rows = len(configs) * batch
    plan = _BodyBatch(bodies, body_leaves(arity)[0])
    companion_plan = _BodyBatch(companions, None) if companions else None
    patterns = constant_patterns(width)
    consts = {name: patterns[CONSTANTS.index(name)].expand(n_rows, width) for name in CONSTANTS}
    x = operands.repeat(len(configs), 1, 1)
    acc = x.new_zeros(len(bodies), n_rows, width)
    comps = [x.new_zeros(n_rows, width) for _ in companions]
    for step in range(width):
        cols = [operands[:, over, (width - 1 - step) if d is Direction.MSB_FIRST else step].unsqueeze(1)
                for over, d in configs]
        b = torch.cat(cols, dim=0)
        bit = torch.cat([b, b.new_zeros(n_rows, width - 1)], dim=1)
        env = [x[:, k] for k in range(arity)] + [None, bit, b.expand(n_rows, width)] + comps
        new_acc = plan.evaluate(env, acc, consts, ops)
        if companion_plan is not None:
            comps = list(companion_plan.evaluate(env, None, consts, ops))
        acc = new_acc
    return acc[..., :out_width].reshape(len(bodies), len(configs), batch, out_width)


def simulate(fold: BitFold, ops: Mapping[str, Operator], operands: torch.Tensor) -> torch.Tensor:
    """Run a fold directly on ``(B, arity, W)`` bits through the operators (search-time check)."""
    return _simulate_batch(fold.arity, [fold.body], [(fold.over, fold.direction)], ops, operands,
                           fold.headroom, fold.companions)[0, 0]


def search_bit_fold(
    operators: Sequence[Operator],
    arity: int,
    example_sets: Sequence[ExampleSet],
    max_size: int = 3,
    budget: int = 2_000_000,
    seed: int = 0,
    library: Sequence[BitFold] = (),
    headrooms: Sequence[int] = (0, 1),
) -> BitFold | None:
    if not operators or not example_sets:
        return None
    ops = {o.name: o for o in operators}
    encoded = [(BitCodec.encode_operands(s.operands, s.width), BitCodec.encode(s.targets, s.width))
               for s in example_sets]
    first_x, first_y = encoded[0][0][:PREFILTER], encoded[0][1][:PREFILTER]
    acc_i, bit_i, mask_i = body_leaves(arity)

    def check(chunk: list[Expr], configs, headroom: int, companions: tuple[Expr, ...]) -> BitFold | None:
        quick = _simulate_batch(arity, chunk, configs, ops, first_x, headroom, companions)
        hits = (quick == first_y).flatten(2).all(dim=2)  # (bodies, configs)
        for b, c in hits.nonzero().tolist():  # row-major: smallest body first
            fold = BitFold(arity, configs[c][0], configs[c][1], chunk[b], headroom, companions)
            if all(torch.equal(simulate(fold, ops, x), y) for x, y in encoded):
                return fold
        return None

    def run_phase(leaves, keep, settings) -> BitFold | None:
        """``settings``: list of (configs, headroom, companions) every kept body is tried under."""
        if not settings:
            return None
        pending: list[Expr] = []

        def flush() -> BitFold | None:
            for configs, headroom, companions in settings:
                if found := check(pending, configs, headroom, companions):
                    return found
            return None

        for body, _ in Enumerator(operators, leaves, max_size, budget):
            if isinstance(body, Var) or not keep(body):
                continue
            pending.append(body)
            if len(pending) == CHUNK:
                if found := flush():
                    return found
                pending = []
        return flush() if pending else None

    # Phase 1: one accumulator. A useful body reads it and the current bit.
    all_configs = [(over, d) for over in range(arity) for d in (Direction.MSB_FIRST, Direction.LSB_FIRST)]
    found = run_phase(
        _probe_leaves(arity, seed),
        lambda e: _uses(e, acc_i) and (_uses(e, bit_i) or _uses(e, mask_i)),
        [(all_configs, h, ()) for h in headrooms],
    )
    if found:
        return found

    # Phase 2: one companion from the library, and a result body of the form
    # acc <- g(acc, h) where h does not read acc (see _search_with_companion).
    for source in library:
        if source.arity != arity or source.companions:
            continue
        found = _search_with_companion(operators, ops, arity, source, encoded, max_size - 1, budget)
        if found:
            return found
    return None


def _companion_steps(
    arity: int, source: BitFold, ops: Mapping[str, Operator], operands: torch.Tensor
) -> list[list[torch.Tensor | None]]:
    """Per-step leaf values (operands, bit, mask, companion's previous value) for a companion run."""
    batch, _, out_width = operands.shape
    width = out_width + source.headroom
    x = torch.cat([operands, operands.new_zeros(batch, arity, source.headroom)], dim=2)
    companion = _BodyBatch([source.as_companion(0)], None)
    patterns = constant_patterns(width)
    consts = {name: patterns[CONSTANTS.index(name)].expand(batch, width) for name in CONSTANTS}
    c = x.new_zeros(batch, width)
    steps = []
    for step in range(width):
        i = width - 1 - step if source.direction is Direction.MSB_FIRST else step
        b = x[:, source.over, i : i + 1]
        bit = torch.cat([b, b.new_zeros(batch, width - 1)], dim=1)
        env = [x[:, k] for k in range(arity)] + [None, bit, b.expand(batch, width), c]
        steps.append(env)
        c = companion.evaluate(env, None, consts, ops)[0]
    return steps


def _search_with_companion(
    operators: Sequence[Operator],
    ops: Mapping[str, Operator],
    arity: int,
    source: BitFold,
    encoded: Sequence[tuple[torch.Tensor, torch.Tensor]],
    max_size: int,
    budget: int,
) -> BitFold | None:
    """Search ``acc <- g(acc, h)`` beside a companion taken from ``source``.

    ``h`` does not read ``acc``, so its value at every step is fixed by the
    companion's trajectory. ``h`` is enumerated with observational
    equivalence on those actual trajectories (exact, not probe-based), and
    every candidate ``h`` is tried with every binary ``g`` in one batch.
    """
    acc_i = body_leaves(arity)[0]
    comp_i = companion_leaf(arity, 0)
    runs = [(_companion_steps(arity, source, ops, x), y) for x, y in encoded]
    # Leaves: each value is the leaf's trajectory, steps stacked along the batch axis.
    leaf_ids = [k for k in range(comp_i + 1) if k != acc_i]
    leaves = [(Var(k), tuple(torch.cat([env[k] for env in steps]) for steps, _ in runs)) for k in leaf_ids]
    widths = [len(steps) for steps, _ in runs]  # register width = number of steps
    leaves += constant_leaves([(len(steps) * y.shape[0], w) for (steps, y), w in zip(runs, widths)])
    companion = (source.as_companion(0),)
    combiners = [op for op in operators if op.arity == 2]
    q = HardThreshold()

    def check(chunk: list[tuple[Expr, Values]]) -> BitFold | None:
        n = len(chunk)
        for g in combiners:
            for acc_first in (True, False) if not g.commutative else (True,):
                ok = torch.ones(n, dtype=torch.bool)
                for group, ((steps, y), w) in enumerate(zip(runs, widths)):
                    batch = y.shape[0]
                    h = torch.stack([v[group] for _, v in chunk]).reshape(n, w, batch, w)
                    acc = h.new_zeros(n, batch, w)
                    for t in range(w):
                        pair = [acc, h[:, t]] if acc_first else [h[:, t], acc]
                        args = torch.stack(pair, dim=2).reshape(n * batch, 2, w)
                        acc = g.executable.execute(args, q).reshape(n, batch, w)
                    ok &= (acc[..., : y.shape[1]] == y).flatten(1).all(dim=1)
                for b in ok.nonzero().flatten().tolist():
                    h_expr = chunk[b][0]
                    body = BinOp(g.name, Var(acc_i), h_expr) if acc_first else BinOp(g.name, h_expr, Var(acc_i))
                    fold = BitFold(arity, source.over, source.direction, body, source.headroom, companion)
                    if all(torch.equal(simulate(fold, ops, x), y) for x, y in encoded):
                        return fold
        return None

    pending: list[tuple[Expr, Values]] = []
    for expr, values in Enumerator(operators, leaves, max_size, budget):
        if not _uses(expr, comp_i):
            continue
        pending.append((expr, values))
        if len(pending) == CHUNK:
            if found := check(pending):
                return found
            pending = []
    return check(pending) if pending else None
