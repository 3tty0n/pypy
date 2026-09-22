#!/usr/bin/env python3
"""Independent checker for the leave histories.

It imports nothing from the probe, the rule, the group module or the
adapter.  What a correct run must produce is computed here from the history
table and the input definition alone:

  h1      B leaves after contributing, preempt (retain): step K keeps g_B
  h2      B leaves after contributing, fault (revoke):   step K drops g_B
  h3      B leaves before contributing, fault:           step K never had it
  h2post  B faults after step K committed: g_B stays in step K (a committed
          step is outside the contract), nothing is revoked
  none    nobody leaves

and in every history with a leave, B contributes nothing after step K.

Inputs (restated, not imported): worker k's shard is
((i*(5+2k) + k) mod 9 - 4) / 2 for i < n, the step scalar is 1 + step mod 4,
g = relu(x*s) + x, w starts at zero and a commit does w -= g/8.  All of it is
exact in binary64, so committed values are compared bitwise.

    check_leave.py RESULT_FILE [RESULT_FILE ...]
    check_leave.py --pair A B     the two files must agree field by field

Each RESULT_FILE is one `leave ...` line as leave_probe.py --save writes it.
Exit status is non-zero on any disagreement.
"""
import sys

EXPECTED_LOG = {
    'h1': ['leave:B:preempt'],
    'h2': ['leave:B:fault', 'revoke:B'],
    'h3': ['leave:B:fault'],
    'h2post': ['leave:B:fault'],
    'none': [],
}
COMPARED = ('commits', 'log', 'w_at', 'w_end')


def parse(path):
    fields = {}
    for tok in open(path).read().split():
        if '=' in tok:
            key, val = tok.split('=', 1)
            fields[key] = val
    return fields


def shard(n, k):
    return [float(((i * (5 + 2 * k) + k) % 9) - 4) * 0.5 for i in range(n)]


def b_contributes(history, step, k):
    if history == 'none' or step < k:
        return True
    if step == k:
        return history in ('h1', 'h2post')
    return False


def model(history, steps, k, n):
    xa, xb = shard(n, 0), shard(n, 1)
    w = [0.0] * n
    w_at = None
    for step in range(steps):
        s = float(1 + step % 4)
        g = [max(x * s, 0.0) + x for x in xa]
        if b_contributes(history, step, k):
            gb = [max(x * s, 0.0) + x for x in xb]
            g = [a + b for a, b in zip(g, gb)]
        w = [wi - gi * 0.125 for wi, gi in zip(w, g)]
        if step == k:
            w_at = list(w)
    return {
        'commits': str(steps),
        'log': '|'.join(EXPECTED_LOG[history]) or '-',
        'w_at': ','.join(float.hex(x) for x in w_at),
        'w_end': ','.join(float.hex(x) for x in w),
    }


def check(path):
    got = parse(path)
    want = model(got['history'], int(got['steps']), int(got['at']),
                 int(got['n']))
    bad = [key for key in COMPARED if got.get(key) != want[key]]
    label = '%s %s' % (got['history'], got.get('impl', '?'))
    if bad:
        print('MISMATCH %-16s %s  (%s)' % (label, ', '.join(bad), path))
        return False
    print('ok       %-16s commits=%s log=%s  (%s)'
          % (label, got['commits'], got['log'], path))
    return True


def pair(a, b):
    fa, fb = parse(a), parse(b)
    bad = [key for key in COMPARED if fa.get(key) != fb.get(key)]
    if bad:
        print('DIFFER   %s vs %s: %s' % (a, b, ', '.join(bad)))
        return False
    print('same     %s vs %s' % (a, b))
    return True


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    if argv[1] == '--pair':
        return 0 if pair(argv[2], argv[3]) else 1
    ok = all([check(p) for p in argv[1:]])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
