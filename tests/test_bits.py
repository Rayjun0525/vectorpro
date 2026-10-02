import random

import pytest

from vectorpro.bits import BitCodec


@pytest.mark.parametrize("width", [1, 3, 8, 9, 33, 64, 65, 70])
def test_roundtrip(width):
    rng = random.Random(width)
    values = [0, (1 << width) - 1] + [rng.getrandbits(width) for _ in range(50)]
    bits = BitCodec.encode(values, width)
    assert bits.shape == (len(values), width)
    assert BitCodec.decode(bits) == values


def test_little_endian():
    assert BitCodec.encode([6], 4).tolist() == [[0.0, 1.0, 1.0, 0.0]]


@pytest.mark.parametrize("value", [-1, 16])
def test_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        BitCodec.encode([value], 4)


def test_encode_operands_shape():
    assert BitCodec.encode_operands([(1, 2, 3), (4, 5, 6)], 5).shape == (2, 3, 5)
