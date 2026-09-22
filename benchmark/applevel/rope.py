"""RoPE tables, shared by the exporter and by the decode driver, which needs
rows past the exported prompt length.  Plain Python so PyPy can import it."""

import math


def rope_tables(seq, dh, heads, theta):
    cos, sin = [], []
    half = dh // 2
    inv = [1.0 / (theta ** (2.0 * i / dh)) for i in range(half)]
    for pos in range(seq):
        row = [math.cos(pos * f) for f in inv]
        row = row + row
        srow = [math.sin(pos * f) for f in inv]
        srow = srow + srow
        cos.extend(row * heads)
        sin.extend(srow * heads)
    return cos, sin


def rope_perm(dh, heads):
    d = dh * heads
    p = [0.0] * (d * d)
    half = dh // 2
    for h in range(heads):
        o = h * dh
        for j in range(dh):
            if j < half:
                p[(o + j + half) * d + o + j] = -1.0
            else:
                p[(o + j - half) * d + o + j] = 1.0
    return p
