"""Operand sets for training and evaluation."""

from __future__ import annotations

import itertools
import random
from typing import Sequence

Operands = tuple[int, ...]


def all_tuples(arity: int, width: int) -> list[Operands]:
    return list(itertools.product(range(1 << width), repeat=arity))


def random_tuples(arity: int, width: int, n: int, rng: random.Random) -> list[Operands]:
    return [tuple(rng.getrandbits(width) for _ in range(arity)) for _ in range(n)]


def split(
    tuples: Sequence[Operands], n_train: int, rng: random.Random
) -> tuple[list[Operands], list[Operands]]:
    shuffled = list(tuples)
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def edge_values(width: int) -> list[int]:
    """0, max, every ``2**k`` and ``2**k - 1``, and alternating bit patterns."""
    top = (1 << width) - 1
    alt = int("10" * width, 2) & top
    values = {0, 1, top, alt, alt ^ top}
    for k in range(width):
        values |= {1 << k, (1 << k) - 1, top ^ (1 << k)}
    return sorted(values)


def edge_pairs(width: int) -> list[Operands]:
    """Boundary pairs: overflow, single-bit boundaries and long carry chains."""
    top = (1 << width) - 1
    pairs = set()
    for v in edge_values(width):
        pairs |= {(v, v), (v, 1), (1, v), (v, top), (top, v), (v, top ^ v), (v, (top - v + 1) & top)}
    return sorted(pairs)
