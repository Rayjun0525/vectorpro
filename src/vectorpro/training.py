"""End-to-end training of an executable's cells through its structure.

Supervision is the final result only; no intermediate state or local truth
table is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from vectorpro.bits import BitCodec
from vectorpro.execution import BitExecutable
from vectorpro.quantize import Identity, Quantizer
from vectorpro.tasks.base import Task


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    lr: float = 0.02
    quantizer: Quantizer = field(default_factory=Identity)


@dataclass(frozen=True)
class TrainReport:
    steps: int
    final_loss: float
    train_exact_match: float


class Trainer:
    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    def fit(
        self,
        executable: BitExecutable,
        task: Task,
        operand_tuples: Sequence[Sequence[int]],
        width: int,
    ) -> TrainReport:
        params = executable.parameters()
        if not params:
            raise TypeError(f"{type(executable).__name__} has no trainable parameters")
        if executable.output_width(width) != task.output_width(width):
            raise ValueError("executable and task disagree on output width")

        x = BitCodec.encode_operands(operand_tuples, width)
        y = BitCodec.encode(
            [task.reference(t, width) for t in operand_tuples], task.output_width(width)
        )
        optimizer = torch.optim.Adam(params, lr=self.config.lr)
        loss = torch.tensor(float("nan"))
        for _ in range(self.config.steps):
            optimizer.zero_grad()
            pred = executable.execute(x, self.config.quantizer).clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(pred, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = executable.execute(x, self.config.quantizer)
            exact = ((pred >= 0.5) == (y >= 0.5)).all(dim=-1).float().mean().item()
        return TrainReport(self.config.steps, loss.item(), exact)
