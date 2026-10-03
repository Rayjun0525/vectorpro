from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from vectorpro.cells.base import Cell, CellSignature
from vectorpro.quantize import Quantizer
from vectorpro.schemas.base import Schema


class Direction(Enum):
    LSB_FIRST = "lsb_first"
    MSB_FIRST = "msb_first"


@dataclass(frozen=True)
class ScanSchema(Schema):
    """Sequential scan over bit positions, threading cell state.

    Output layout: per-position outputs (position-major, LSB first),
    followed by the final state if ``emit_final_state``.
    """

    arity: int
    initial_state: tuple[int, ...]
    direction: Direction = Direction.LSB_FIRST
    emit_final_state: bool = True

    def check(self, signature: CellSignature) -> None:
        if signature.n_inputs != self.arity:
            raise ValueError(f"cell takes {signature.n_inputs} inputs, schema arity {self.arity}")
        if signature.n_state != len(self.initial_state):
            raise ValueError(
                f"cell state {signature.n_state} != initial_state {len(self.initial_state)}"
            )

    def output_width(self, width: int, signature: CellSignature) -> int:
        return width * signature.n_outputs + (signature.n_state if self.emit_final_state else 0)

    def to_spec(self) -> dict:
        return {
            "kind": "scan",
            "arity": self.arity,
            "initial_state": list(self.initial_state),
            "direction": self.direction.value,
            "emit_final_state": self.emit_final_state,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> ScanSchema:
        return cls(
            spec["arity"],
            tuple(spec["initial_state"]),
            Direction(spec["direction"]),
            spec["emit_final_state"],
        )

    def run(self, cell: Cell, operands: torch.Tensor, quantizer: Quantizer) -> torch.Tensor:
        self.check(cell.signature)
        batch, _, width = operands.shape
        state = operands.new_tensor(self.initial_state).expand(batch, -1)
        positions = range(width) if self.direction is Direction.LSB_FIRST else reversed(range(width))
        outputs: list[torch.Tensor | None] = [None] * width
        for i in positions:
            y, state = cell.step(operands[:, :, i], state)
            outputs[i] = quantizer(y)
            state = quantizer(state)
        pieces = [torch.stack(outputs, dim=1).reshape(batch, -1)]
        if self.emit_final_state:
            pieces.append(state)
        return torch.cat(pieces, dim=-1)
