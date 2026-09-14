"""INPUT_SEED: the same model on a different input, derived the same way on
every side.

The correctness check compares each system's logits against ours on one stored
input.  One input cannot tell "the same computation" from "a computation that
agrees on this vector", so `bench.sh correctness` runs the same check on five
more inputs per model.  Those inputs have to be bit-identical across pypy2,
torch and jax without the three sharing an RNG, so they are derived by integer
arithmetic from what is already in the checkpoint:

    INPUT_SEED=0   the stored input, unchanged - the default everywhere, so
                   nothing about an existing run moves.
    text, k>0      ids[i] = (tokens[(i + k) % seq] + k) % vocab: the stored
                   sequence rotated left by k, every id shifted by k.  Always
                   a valid id, always the same length, and a different
                   sequence for each k.
    vision, k>0    pixel[i] += 0.05 * (2*u(i, k) - 1), u in [0, 1) from an
                   integer hash of (flat index, seed): additive uniform noise
                   on the normalized pixel values, +-0.05.  Indexed by the
                   position in the flat `image` region of weights.bin, which
                   is the same region on all three sides whatever shape each
                   of them reads it as, so the layout ([3, s, s] or [s, s, 3])
                   does not enter into it.

Every side adds the noise in double and stores float32, so the three buffers
are bit-identical rather than merely close.

Self-check: python3 inputs.py
"""

import os

M = 2147483647


def seed():
    try:
        return int(os.environ.get('INPUT_SEED', '0') or 0)
    except ValueError:
        return 0


def tokens(toks, vocab, k):
    n = len(toks)
    if not k or not n:
        return list(toks)
    return [(toks[(i + k) % n] + k) % vocab for i in range(n)]


def noise(i, k):
    h = (i * 2654435761 + k * 40503 + 12345) % M
    h = (h * 1103515245 + 12345) % M
    return 0.05 * (2.0 * h / float(M) - 1.0)


def perturb_cfg(cfg, k):
    """cfg['tokens'] in place, for the text models.  It is the token count for
    vit/mixer, so only a list is an input."""
    if k and isinstance(cfg.get('tokens'), list):
        cfg['tokens'] = tokens(cfg['tokens'], cfg['vocab'], k)


def image_span(cfg):
    """(offset, count) of the image inside weights.bin, or None."""
    entry = cfg.get('index', {}).get('image')
    if not entry:
        return None
    off, shape = entry
    n = 1
    for d in shape:
        n *= d
    return off, n


def reference_name(dtype, k):
    """The per-seed reference.  Seed 0 keeps the stored name, so the reference
    every other stage compares against is never rewritten by this sweep."""
    base = 'logits_pypy' if dtype == 'float32' else 'logits_pypy_%s' % dtype
    return '%s%s.bin' % (base, '_seed%d' % k if k else '')


def _demo():
    assert seed() >= 0
    t = [1, 2, 3, 4]
    assert tokens(t, 10, 0) == t
    assert tokens(t, 10, 1) == [3, 4, 5, 2]          # rotate left 1, +1, mod 10
    assert tokens([9], 10, 1) == [0]                 # wraps in the vocabulary
    assert len(set(tuple(tokens(t, 10, k)) for k in range(1, 6))) == 5
    for k in range(1, 6):
        vals = [noise(i, k) for i in range(1000)]
        assert all(-0.05 <= v <= 0.05 for v in vals)
        assert len(set(vals)) > 900                  # not a constant
        assert noise(7, k) != noise(7, k + 1)        # the seed moves it
    assert noise(3, 0) == noise(3, 0)                # deterministic
    assert reference_name('float32', 0) == 'logits_pypy.bin'
    assert reference_name('float32', 3) == 'logits_pypy_seed3.bin'
    assert reference_name('float16', 3) == 'logits_pypy_float16_seed3.bin'
    print('inputs.py ok')


if __name__ == '__main__':
    _demo()
