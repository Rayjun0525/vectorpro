"""Vector programs: a program's entire logic as tensors.

A program has ``S`` steps over ``R`` registers of ``W`` bits. Step ``s``:

* ``keys[s]``      address vector of the capability to call (resolved by similarity);
* ``reads[s]``     ``(K, R)`` one-hot: argument ``k`` reads register ``r``;
* ``writes[s]``    ``(R+1,)`` one-hot destination; the last slot discards (branch-only step);
* ``cond[s]``      ``(R+1,)`` one-hot register tested for non-zero; the last slot means "always";
* ``next_true[s]`` / ``next_false[s]`` ``(S+1,)`` one-hot next step; the last slot halts.

Registers start from ``init`` (one-hot over ``CONSTANTS``); ``inputs`` loads
operands into registers; ``output`` selects the result register. There are no
names, symbols or code in a program, only these tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

CONSTANTS = ("zero", "one", "top", "ones")
"""Width-generic constant patterns: 0, 1, the top bit alone, all ones."""

MAX_ARGS = 3


def constant_patterns(width: int) -> torch.Tensor:
    """``(len(CONSTANTS), width)`` bit patterns, LSB first."""
    patterns = torch.zeros(len(CONSTANTS), width)
    patterns[1, 0] = 1
    patterns[2, width - 1] = 1
    patterns[3] = 1
    return patterns


@dataclass(frozen=True)
class VectorProgram:
    keys: torch.Tensor        # (S, D)
    reads: torch.Tensor       # (S, K, R)
    writes: torch.Tensor      # (S, R+1)
    cond: torch.Tensor        # (S, R+1)
    next_true: torch.Tensor   # (S, S+1)
    next_false: torch.Tensor  # (S, S+1)
    inputs: torch.Tensor      # (A, R)
    init: torch.Tensor        # (R, len(CONSTANTS))
    output: torch.Tensor      # (R,)

    def __post_init__(self) -> None:
        s, r = self.n_steps, self.n_registers
        expected = {
            "reads": (s, MAX_ARGS, r), "writes": (s, r + 1), "cond": (s, r + 1),
            "next_true": (s, s + 1), "next_false": (s, s + 1),
            "init": (r, len(CONSTANTS)), "output": (r,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}")
        if self.inputs.shape[1] != r:
            raise ValueError("inputs must select among the registers")

    @property
    def n_steps(self) -> int:
        return self.keys.shape[0]

    @property
    def n_registers(self) -> int:
        return self.output.shape[0]

    @property
    def arity(self) -> int:
        return self.inputs.shape[0]

    def to_data(self) -> dict:
        return {f.name: getattr(self, f.name).tolist() for f in fields(self)}

    @classmethod
    def from_data(cls, data: dict) -> VectorProgram:
        return cls(**{f.name: torch.tensor(data[f.name], dtype=torch.float32) for f in fields(cls)})
