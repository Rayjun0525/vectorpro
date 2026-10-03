"""Target values for curricula: stand-ins for human labels or traces of an existing system.

Only the learner's example stream reads these, one requested example at a
time; the learner never sees the functions themselves.
"""

from __future__ import annotations


def _mask(w: int) -> int:
    return (1 << w) - 1


TARGETS = {
    "and": lambda o, w: o[0] & o[1],
    "or": lambda o, w: o[0] | o[1],
    "xor": lambda o, w: o[0] ^ o[1],
    "xnor": lambda o, w: ~(o[0] ^ o[1]) & _mask(w),
    "add": lambda o, w: o[0] + o[1],
    "sub": lambda o, w: ((o[0] - o[1]) & _mask(w)) | (int(o[0] < o[1]) << w),
    "lt": lambda o, w: int(o[0] < o[1]),
    "gt": lambda o, w: int(o[0] > o[1]),
    "mux": lambda o, w: (o[0] & ~o[2]) | (o[1] & o[2]),
    "double": lambda o, w: (2 * o[0]) & _mask(w),
    "add3": lambda o, w: (o[0] + o[1] + o[2]) & _mask(w),
    "sub_add": lambda o, w: (o[0] - o[1] + o[2]) & _mask(w),
    "triple_sub": lambda o, w: (3 * o[0] - o[1]) & _mask(w),
    "sum_xor": lambda o, w: ((o[0] + o[1]) & _mask(w)) ^ o[2],
    "mul": lambda o, w: (o[0] * o[1]) & _mask(w),
    "avg": lambda o, w: (o[0] + o[1]) >> 1,
    "max": lambda o, w: max(o[0], o[1]),
    "shl": lambda o, w: (o[0] << 1) & _mask(w),
    "shr": lambda o, w: o[0] >> 1,
    "mul_add": lambda o, w: (o[0] * o[1] + o[2]) & _mask(w),
    "sum_mod": lambda o, w: ((o[0] + o[1]) & _mask(w)) % o[2] if o[2] else (o[0] + o[1]) & _mask(w),
    "diff_div": lambda o, w: ((o[0] - o[1]) & _mask(w)) // o[2] if o[2] else _mask(w),
    "square": lambda o, w: (o[0] * o[0]) & _mask(w),
    "div": lambda o, w: o[0] // o[1] if o[1] else _mask(w),
    "neg": lambda o, w: (-o[0]) & _mask(w),
    "nand": lambda o, w: ~(o[0] & o[1]) & _mask(w),
    "min": lambda o, w: min(o[0], o[1]),
    "popcount": lambda o, w: bin(o[0]).count("1"),
    "reverse": lambda o, w: int(format(o[0], f"0{w}b")[::-1], 2),
    "mod": lambda o, w: o[0] % o[1] if o[1] else o[0],
}
