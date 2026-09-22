"""A declared change rule against the hand-written guard-recovery path.

The rule is benchmark/motion/layout_rule.py: an equation saying what it means
for the same values to be addressed as blocks rather than as a matrix.  It is
written in the interpreter's vocabulary and knows nothing about storage,
devices, kernels or what will have produced the matrix; benchmark/paper/
audit_rules.py checks that mechanically, and rejects a deliberately
contaminated copy of the same rule.

Two ways to carry the change across a guard, on the same rule:

  derived      the rule is applied to the value where it is.  The value is
               still inside the region that is producing it, so the
               meta-tracer specialises the rule along with the region and the
               region produces the blocked form directly.  Nobody wrote a
               transition; it is what specialisation left behind.
  handwritten  the value is first brought to canonical form through the
               existing recovery path, and the rule is applied to that.  This
               is what a system that carries no descriptor across the
               boundary has to do, and it is the path MetaTensor's guard
               recovery already takes.

Both arms compute the same numbers, and both are checked against the rule's
own arithmetic on plain floats - not against each other - so an arm that is
wrong is caught by the equation rather than by agreement.

The change crosses a real guard: up to SWITCH the step does not ask for the
blocked form, from SWITCH it does, and the `if` on the step index is constant
on the warm trace, so the first step past SWITCH fails its guard.

Reported per arm, over RUNS fresh processes: the median step latency after
the transition and the device bytes still held at the end of the run, each
with a distribution-free interval from the order statistics.

`step latency` is one decoder-shaped forward at a fixed length.  It is not an
autoregressive decode step: MetaTensor has no in-place key/value cache, so
there is no next-token loop to measure and the number should not be read as
one.

    motion_probe.py ARM STEPS SWITCH [--rows N] [--cols N] [--heads N]
    ARM: derived | handwritten
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'motion'))

import _metatensor
import layout_rule

ROWS, COLS, HEADS = 256, 96, 12


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def leaf_values(rows, cols):
    return [((i * 41 + 7) % 89) * 0.01 - 0.4 for i in range(rows * cols)]


def run(arm, steps, switch, rows, cols, heads):
    vals = leaf_values(rows, cols)
    leaf = _metatensor.tensor(vals, [rows, cols], False, 'float64')
    one = _metatensor.scalar(1.0, 'float64')
    us = []
    blocked_sum = None
    # Launches and distinct kernels, split at the transition.  This is the
    # evidence for whether specialisation absorbed the rule into the region
    # it was tracing, and it does not depend on a clock.
    l0 = _metatensor.launch_count()
    k0 = _metatensor.kernel_count()
    l_at_switch = k_at_switch = 0
    for i in range(steps):
        if i == switch:
            l_at_switch = _metatensor.launch_count()
            k_at_switch = _metatensor.kernel_count()
        t0 = time.time()
        a = leaf.mul(leaf)            # the region being produced
        b = a.add(one)
        plain = b.relu().sum().item()
        if i >= switch:
            if arm == 'derived':
                # the rule, applied where the value is
                blocked = layout_rule.apply(a, heads)
            else:
                # the value brought to canonical form first, then the rule
                blocked = layout_rule.apply(a.force(), heads)
            blocked_sum = blocked.sum().item()
        us.append((time.time() - t0) * 1e6)
    # Device bytes the allocator is still holding.  mem_total() is the
    # card's capacity and answers a different question; an interpreter
    # without live_bytes reports -1 rather than a number that looks right.
    retained = -1
    if hasattr(_metatensor, 'live_bytes'):
        retained = _metatensor.live_bytes()
    n_after = steps - switch
    counters = (
        (l_at_switch - l0) / float(switch or 1),           # launches/step before
        (_metatensor.launch_count() - l_at_switch) / float(n_after or 1),
        k_at_switch - k0,                                  # kernels before
        _metatensor.kernel_count() - k_at_switch)          # kernels after
    return us, plain, blocked_sum, retained, counters


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    arm = argv[0] if argv else 'derived'
    steps = int(argv[1]) if len(argv) > 1 else 200
    switch = int(argv[2]) if len(argv) > 2 else steps // 2
    rows, cols, heads = ROWS, COLS, HEADS
    for flag in ('--rows', '--cols', '--heads'):
        if flag in sys.argv:
            v = int(sys.argv[sys.argv.index(flag) + 1])
            if flag == '--rows':
                rows = v
            elif flag == '--cols':
                cols = v
            else:
                heads = v
    if arm not in ('derived', 'handwritten'):
        sys.stderr.write('motion_probe: unknown arm %r\n' % arm)
        return 2

    us, plain, blocked_sum, retained, counters = run(arm, steps, switch,
                                                    rows, cols, heads)

    # Against the rule's own arithmetic, not against the other arm.
    vals = leaf_values(rows, cols)
    want_plain = 0.0
    squared = [0.0] * (rows * cols)
    for k, v in enumerate(vals):
        squared[k] = v * v
        want_plain += squared[k] + 1.0
    want_blocked = sum(layout_rule.reference(squared, rows, cols, heads))
    ok = 1
    for got, want, what in ((plain, want_plain, 'plain'),
                            (blocked_sum, want_blocked, 'blocked')):
        if got is None or abs(got - want) > 1e-9 * max(1.0, abs(want)):
            sys.stderr.write('motion_probe: %s %r, the rule says %r\n'
                             % (what, got, want))
            ok = 0

    after = median(us[switch + 8:]) if steps > switch + 8 else median(us)
    print('motion arm=%s steps=%d switch=%d rows=%d cols=%d heads=%d '
          'step_us=%.1f at_us=%.1f retained_bytes=%d '
          'launches_before=%.2f launches_after=%.2f '
          'kernels_before=%d kernels_after=%d pass=%d'
          % (arm, steps, switch, rows, cols, heads, after,
             us[switch] if switch < len(us) else 0.0, retained,
             counters[0], counters[1], counters[2], counters[3], ok))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
