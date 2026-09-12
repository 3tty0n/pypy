"""Shared per-iteration warm-up trace, used by common.py (ours),
torch_common.py and jax_models.py so the trace/steady-state logic is written
once instead of three times.

Enabled by WARMUP_TRACE=N in the environment: N forwards, one
`iter=<i> us=<latency>` line each, then one summary line.
"""
import os


def trace_n():
    return int(os.environ.get('WARMUP_TRACE', '0') or '0')


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def steady_at(us, steady_us, window=20, tol=0.05):
    """First iteration i (0-indexed, matching the printed `iter=` numbers)
    such that every one of the next `window` forwards is within `tol` of
    steady_us, or 'none' if that never happens within the trace."""
    lo, hi = steady_us * (1 - tol), steady_us * (1 + tol)
    for i in range(len(us) - window + 1):
        if all(lo <= x <= hi for x in us[i:i + window]):
            return i
    return 'none'


def report_trace(us, compiled=None):
    """us: per-forward latencies in microseconds, in order. Prints the
    `total_us=... steady_us=... steady_at=...` summary line.

    compiled, when given (the "ours" driver only): per-forward count of
    kernels actually compiled through the Triton subprocess (a disk-cache
    miss, from _metatensor.kernel_compile_count() deltas). Adds
    cold_compiles (the trace total) and first_forward_compiles (iter 0)."""
    last = us[-50:] if len(us) >= 50 else us
    steady_us = _median(last)
    line = 'total_us=%.1f steady_us=%.1f steady_at=%s' % (
        sum(us), steady_us, steady_at(us, steady_us))
    if compiled is not None:
        line += ' cold_compiles=%d first_forward_compiles=%d' % (
            sum(compiled), compiled[0] if compiled else 0)
    print(line)
