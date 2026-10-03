"""Learning plans: what a requirement becomes instead of code.

A plan says what to learn (name, description, shape), how to teach it (an
example schedule) and when it is done (acceptance). It is plain data, so a
language model can produce it from a human requirement; it never contains a
program.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class OutputWidth(str, Enum):
    """Result width as a function of the operand width ``W``."""

    SAME = "W"
    PLUS_ONE = "W+1"
    DOUBLE = "2W"
    BIT = "1"

    def __call__(self, width: int) -> int:
        return {"W": width, "W+1": width + 1, "2W": 2 * width, "1": 1}[self.value]


@dataclass(frozen=True)
class LearningPlan:
    name: str
    description: str
    arity: int
    output: OutputWidth
    rounds: tuple[int, ...] = (8, 16, 32, 64)
    """Cumulative number of training examples available in each round."""
    train_width: int = 4
    validation_width: int = 8
    validation_examples: int = 32
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {**asdict(self), "output": self.output.value}

    @classmethod
    def from_dict(cls, data: dict) -> LearningPlan:
        """Build from plain data; omitted optional fields take their defaults."""
        fields = dict(data, output=OutputWidth(data["output"]))
        for key in ("rounds", "tags"):
            if key in fields:
                fields[key] = tuple(fields[key])
        return cls(**fields)
