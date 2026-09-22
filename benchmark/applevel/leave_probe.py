"""Two workers, an uncommitted gradient aggregate, and a worker that leaves.

    leave_probe.py HISTORY [--impl motion|adapter] [--steps S] [--at K]
                           [--n N] [--save FILE] [--nostats] [--times]
                           [--policy retain-only] [--sync]

Workers A and B each compute a gradient from their own shard, the step
commits w -= lr * (g_A + g_B), and at step K (after the loop is hot, so in
compiled code) worker B leaves:

  h1      B leaves after g_B entered the aggregate, reason preempt
          declared retain  -> step K commits g_A + g_B
  h2      the same, reason fault, declared revoke
                           -> step K commits g_A alone
  h3      B leaves before contributing, reason fault
                           -> step K commits g_A alone, nothing to revoke
  h2post  B faults right after step K committed: outside the contract (a
          committed step cannot be revoked); reported, not repaired
  none    nobody leaves

From the step after the leave on, only A contributes.

--impl motion runs the change rule through _metatensor.group (the generated
transition); --impl adapter runs benchmark/motion/adapter_leave.py (the
hand-written one).  Both see the same notices at the same points: three
membership checks per step - before A's contribution, before B's, and at the
start of the commit - numbered from 1.

Inputs are dyadic rationals and lr = 1/8, so every sum is exact: the checker
compares committed values bitwise, and revocation rebuilds the value from
the surviving contributions in contribution order.  Nothing depends on a
tolerance.

--policy retain-only declares retain for every reason, so nothing is kept:
the steady state to compare the retaining one against.  (A fault is then
retained too, and h2 is no longer what the checker expects.)

The result line and --save carry what the checker compares: w after step K,
w at the end, the commit count and the effect log.  --times adds per-step
wall times (and a host read of w every step is then skipped).
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'motion'))

import _metatensor as mt
import adapter_leave

try:
    import pypyjit
except ImportError:
    pypyjit = None
if '--nostats' in sys.argv:
    pypyjit = None

HISTORIES = {
    'h1': ('B', 'preempt', 3),
    'h2': ('B', 'fault', 3),
    'h3': ('B', 'fault', 2),
    'h2post': ('B', 'fault', 4),     # first check of step K+1
    'none': None,
}
POLICY = [('preempt', 'retain'), ('fault', 'revoke')]  # [policy]
LR = 0.125


def flag(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def shard(n, k):
    return [float(((i * (5 + 2 * k) + k) % 9) - 4) * 0.5 for i in range(n)]


def grad(x, s):
    return (x * s).relu() + x


def counters():
    loops = bridges = 0
    if pypyjit is not None:
        try:
            c = pypyjit.get_stats_snapshot().counters
            loops = c['TOTAL_COMPILED_LOOPS']
            bridges = c['TOTAL_COMPILED_BRIDGES']
        except Exception:
            pass
    return loops, bridges, mt.launch_count()


# --sync ends every timed step with a host read (w.sum().item()), so the time
# includes the device finishing the step; without it a step is timed to the
# end of its launches, which are asynchronous: host dispatch time.
SYNC = '--sync' in sys.argv
STEADY = [0, 0]


def mark(step, k):
    """Launch count at the start of the hot window [k/2, k) and at its end:
    the steady state before the leave, JIT warm and nothing failing."""
    if step == k // 2:
        STEADY[0] = mt.launch_count()
    elif step == k:
        STEADY[1] = mt.launch_count()


def event_poll(history, k):
    ev = HISTORIES[history]
    if ev is None:
        return None
    who, reason, when = ev
    return who, reason, 3 * k + when


def run_motion(history, steps, k, n, times):
    g = mt.group(['A', 'B'])
    policy = POLICY
    if flag('--policy', '') == 'retain-only':
        policy = [(reason, 'retain') for reason, _ in POLICY]
    for reason, action in policy:  # [motion-policy]
        g.declare(reason, action)  # [motion-policy]
    ev = event_poll(history, k)
    if ev is not None:
        g.arm(*ev)
    xa = mt.tensor(shard(n, 0))
    xb = mt.tensor(shard(n, 1))
    w = mt.tensor([0.0] * n)
    w_k = None
    for step in range(steps):
        mark(step, k)
        t0 = time.time()
        w = motion_step(g, xa, xb, w, mt.scalar(float(1 + step % 4),
                                                'float64'))
        if times is not None:
            if SYNC:
                w.sum().item()
            else:
                w.force()
            times.append((time.time() - t0) * 1e6)
        if step == k:
            w_k = w.tolist()
    return w_k, w.tolist(), g.commits(), g.log(), g.retains()


def motion_step(g, xa, xb, w, s):
    """One training step.  A function, as a training step is: its locals
    die when it returns, so nothing of the step is live at the loop's
    back-edge (this interpreter keeps dead locals alive there, which would
    make the JIT materialise the aggregate every step)."""
    agg = g.aggregate()
    agg.sync()
    agg.contribute('A', grad(xa, s))
    agg.sync()
    agg.contribute('B', grad(xb, s))
    return g.commit(agg, w, LR)


def run_adapter(history, steps, k, n, times):
    ch = adapter_leave.Channel()
    ev = event_poll(history, k)
    if ev is not None:
        ch.arm(*ev)
    agg = adapter_leave.ManualAggregator(['A', 'B'], POLICY, ch)
    xa = mt.tensor(shard(n, 0))
    xb = mt.tensor(shard(n, 1))
    w = mt.tensor([0.0] * n)
    lr = mt.scalar(LR, 'float64')
    w_k = None
    for step in range(steps):
        mark(step, k)
        t0 = time.time()
        w = adapter_step(agg, step, xa, xb, w, lr,
                         mt.scalar(float(1 + step % 4), 'float64'))
        if times is not None:
            if SYNC:
                w.sum().item()
            else:
                w.force()
            times.append((time.time() - t0) * 1e6)
        if step == k:
            w_k = w.tolist()
    return w_k, w.tolist(), agg.commits, agg.events, True


def adapter_step(agg, step, xa, xb, w, lr, s):
    agg.begin(step)
    agg.check()
    agg.contribute('A', grad(xa, s))
    agg.check()
    agg.contribute('B', grad(xb, s))
    return agg.commit(w, lr)


def median(xs):
    s = sorted(xs)
    m = len(s)
    return 0.0 if not m else (s[m // 2] if m % 2
                              else 0.5 * (s[m // 2 - 1] + s[m // 2]))


def hexlist(xs):
    return ','.join([float.hex(x) for x in xs])


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    skip = set()
    for i, a in enumerate(sys.argv):
        if a in ('--impl', '--steps', '--at', '--n', '--save', '--policy'):
            skip.add(sys.argv[i + 1])
    argv = [a for a in argv if a not in skip]
    history = argv[0] if argv else 'h2'
    if history not in HISTORIES:
        sys.stderr.write('leave_probe: unknown history %r\n' % history)
        return 2
    impl = flag('--impl', 'motion')
    steps = int(flag('--steps', '200'))
    k = int(flag('--at', str(steps // 2)))
    n = int(flag('--n', '64'))
    times = [] if '--times' in sys.argv else None

    c0 = counters()
    run = run_motion if impl == 'motion' else run_adapter
    w_k, w_end, commits, log, retains = run(history, steps, k, n, times)
    c1 = counters()

    fields = [('history', history), ('impl', impl), ('steps', steps),
              ('at', k), ('n', n), ('commits', commits),
              ('retains', int(retains)),
              ('loops', c1[0] - c0[0]), ('bridges', c1[1] - c0[1]),
              ('launches_per_step', '%.3f' % ((c1[2] - c0[2]) /
                                              float(steps))),
              ('steady_launches', '%.3f' % ((STEADY[1] - STEADY[0]) /
                                            float(k - k // 2))),
              ('log', '|'.join([e.replace(' ', ':') for e in log]) or '-'),
              ('w_at', hexlist(w_k)), ('w_end', hexlist(w_end))]
    if times is not None:
        fields.append(('step_us', '%.1f' % median(times[k + 10:] or times)))
        fields.append(('before_us', '%.1f' % median(times[k // 2:k])))
        fields.append(('event_us', '%.1f' % times[k]))
    line = ' '.join(['%s=%s' % kv for kv in fields])
    print('leave ' + line)
    save = flag('--save', None)
    if save:
        f = open(save, 'w')
        f.write(line + '\n')
        f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
