"""Integer <-> bit-vector encoding.

Bit tensors are little-endian along the last axis: index 0 is the least
significant bit. Values are float tensors so they can flow through learned
cells; decoding thresholds at 0.5.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class BitCodec:
    """Unsigned fixed-width integer codec."""

    @staticmethod
    def encode(values: Sequence[int], width: int) -> torch.Tensor:
        """Encode unsigned ints to a ``(N, width)`` float tensor of 0/1."""
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        for v in values:
            if v < 0 or v >> width:
                raise ValueError(f"{v} does not fit in {width} unsigned bits")
        nbytes = (width + 7) // 8
        buf = b"".join(v.to_bytes(nbytes, "little") for v in values)
        packed = np.frombuffer(buf, dtype=np.uint8).reshape(len(values), nbytes)
        bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :width]
        return torch.from_numpy(bits.astype(np.float32))

    @staticmethod
    def decode(bits: torch.Tensor, threshold: float = 0.5) -> list[int]:
        """Decode a ``(N, width)`` tensor to unsigned ints."""
        hard = (bits.detach() >= threshold).cpu().numpy().astype(np.uint8)
        packed = np.packbits(hard, axis=1, bitorder="little")
        return [int.from_bytes(row.tobytes(), "little") for row in packed]

    @classmethod
    def encode_operands(
        cls, operand_tuples: Sequence[Sequence[int]], width: int
    ) -> torch.Tensor:
        """Encode tuples of operands to a ``(N, arity, width)`` tensor."""
        if not operand_tuples:
            raise ValueError("operand_tuples is empty")
        arity = len(operand_tuples[0])
        columns = [cls.encode([t[k] for t in operand_tuples], width) for k in range(arity)]
        return torch.stack(columns, dim=1)
