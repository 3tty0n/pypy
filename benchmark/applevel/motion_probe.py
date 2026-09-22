"""The blocked-layout change rule, derived against hand-written recovery.

The rule is benchmark/motion/layout_rule.py: the same values addressed as
blocks of heads instead of as a matrix,

    blocked[b*rows + i, j] = matrix[i, b*(cols/heads) + j]

It is written in the interpreter's vocabulary and knows nothing about
storage, kernels or what produced the matrix; benchmark/paper/audit_rules.py
checks that mechanically and rejects a contaminated copy of the same rule.

The two representations are genuinely two: a [rows, cols] matrix and a
[heads*rows, cols/heads] stack of blocks, with different shapes and a
different element order.  Up to SWITCH the step reads the region in the
matrix layout; from SWITCH it reads it in the blocked layout, through the
rule.  The `if` on the step index is constant on the warm trace, so the first
step past SWITCH fails its guard.  Each layout is read by a weighted sum whose
weights have that layout's shape, so reading the wrong order is caught.

  derived      the rule applied to the value where it is, so the region the
               meta-tracer is specialising produces the blocked form itself
  handwritten  the value brought to canonical form first, then the rule
  direct       the hand-written adapter that keeps the descriptor: the region
               has already run (its sum is launched first), so the rule reaches
               its value as one more output of the launched kernel
  drain        the same program as direct under METATENSOR_DRAIN=1

Needs a binary whose gathers are fusion nodes; on one without, derived and
handwritten are the same program, which was the first attempt's null.

Both layouts are checked against the rule's own arithmetic on plain floats.

    motion_probe.py ARM STEPS SWITCH [--rows N] [--cols N] [--heads N]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'motion'))

import _metatensor
import layout_rule

ROWS, COLS, HEADS = 256, 96, 12
ARMS = ('derived', 'handwritten', 'direct', 'drain')


def median(xs):
    s = sorted(xs)
    k = len(s)
    return 0.0 if not k else (s[k // 2] if k % 2
                              else 0.5 * (s[k // 2 - 1] + s[k // 2]))


def leaf_values(n):
    return [((i * 41 + 7) % 89) * 0.01 - 0.4 for i in range(n)]


def weight_values(n):
    return [((i * 13 + 2) % 29) * 0.05 - 0.6 for i in range(n)]


def run(arm, steps, switch, rows, cols, heads):
    n = rows * cols
    leaf = _metatensor.tensor(leaf_values(n), [rows, cols], False, 'float64')
    one = _metatensor.scalar(1.0, 'float64')
    wv = weight_values(n)
    weights = {
        'matrix': _metatensor.tensor(wv, [rows, cols], False, 'float64'),
        'blocked': _metatensor.tensor(
            wv, list(layout_rule.blocked_shape(rows, cols, heads)), False,
            'float64'),
    }
    us = []
    got = {}
    anchor = None
    l0, k0 = _metatensor.launch_count(), _metatensor.kernel_count()
    lsw = ksw = 0
    for i in range(steps):
        if i == switch:
            lsw, ksw = _metatensor.launch_count(), _metatensor.kernel_count()
        t0 = time.time()
        region = leaf.mul(leaf).add(one).relu()   # the region being produced
        layout = 'matrix' if i < switch else 'blocked'
        if arm == 'handwritten':
            region = region.force()
        elif arm == 'direct' or arm == 'drain':
            anchor = region.sum().force()
        out = region
        if layout == 'blocked':
            out = layout_rule.apply(region, heads)
        got[layout] = out.mul(weights[layout]).sum().item()
        us.append((time.time() - t0) * 1e6)
    if anchor is not None:
        got['anchor'] = anchor.item()
    nafter = steps - switch
    counters = ((lsw - l0) / float(switch or 1),
                (_metatensor.launch_count() - lsw) / float(nafter or 1),
                ksw - k0, _metatensor.kernel_count() - ksw)
    retained = -1
    if hasattr(_metatensor, 'live_bytes'):
        retained = _metatensor.live_bytes()
    return us, got, retained, counters


def flag(name, default):
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    argv = []
    rest = sys.argv[1:]
    while rest:
        a = rest.pop(0)
        if a.startswith('--'):
            rest = rest[1:]
        else:
            argv.append(a)
    arm = argv[0] if argv else 'derived'
    steps = int(argv[1]) if len(argv) > 1 else 200
    switch = int(argv[2]) if len(argv) > 2 else steps // 2
    rows, cols, heads = (flag('--rows', ROWS), flag('--cols', COLS),
                         flag('--heads', HEADS))
    if arm not in ARMS:
        sys.stderr.write('motion_probe: unknown arm %r\n' % arm)
        return 2
    drain = os.environ.get('METATENSOR_DRAIN', '') not in ('', '0')
    if drain != (arm == 'drain'):
        sys.stderr.write('motion_probe: arm %s needs METATENSOR_DRAIN %s\n'
                         % (arm, 'set' if arm == 'drain' else 'unset'))
        return 2

    us, got, retained, c = run(arm, steps, switch, rows, cols, heads)

    region = [max(v * v + 1.0, 0.0) for v in leaf_values(rows * cols)]
    wv = weight_values(rows * cols)
    blocked = layout_rule.reference(region, rows, cols, heads)
    want = {
        'matrix': sum([a * b for a, b in zip(region, wv)]),
        'blocked': sum([a * b for a, b in zip(blocked, wv)]),
        'anchor': sum(region),
    }
    ok = 1
    for what in sorted(got):
        if abs(got[what] - want[what]) > 1e-9 * max(1.0, abs(want[what])):
            sys.stderr.write('motion_probe: %s %r, the rule says %r\n'
                             % (what, got[what], want[what]))
            ok = 0

    after = median(us[switch + 8:]) if steps > switch + 8 else median(us)
    print('motion arm=%s steps=%d switch=%d rows=%d cols=%d heads=%d '
          'step_us=%.1f at_us=%.1f retained_bytes=%d '
          'launches_before=%.2f launches_after=%.2f '
          'kernels_before=%d kernels_after=%d pass=%d'
          % (arm, steps, switch, rows, cols, heads, after,
             us[switch] if switch < len(us) else 0.0, retained,
             c[0], c[1], c[2], c[3], ok))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
