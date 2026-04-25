"""
Dispatch-pair profiler for the meta-tracing interpreter.

Implements Idea 1 (meta-interpreter warmup speedup): records adjacent opcode
pair frequencies per jitcode while the slow-path dispatch loop runs, so a
later pass can emit AOT-fused "super-jitcode" handlers for the hottest pairs
(Piumarta & Riccardi style selective inlining, lifted to the meta level).

Activation:
  PYPY_DISPATCH_PROFILE=1      enable profiling
  PYPY_DISPATCH_PROFILE_TOP=N  number of pairs to report (default 50)
  PYPY_DISPATCH_PROFILE_OUT=path  export JSON report at exit

Designed to be a no-op when disabled so it can sit in the slow loop without
measurable overhead. Untranslated (pure-CPython) use only for now; the
translated build reads the profile at codegen time to feed GenExtension.
"""

import os
import atexit
import json

DISPATCH_PROFILE_ENABLED = os.environ.get('PYPY_DISPATCH_PROFILE', '') == '1'
_TOP = int(os.environ.get('PYPY_DISPATCH_PROFILE_TOP', '50') or '50')
_OUT = os.environ.get('PYPY_DISPATCH_PROFILE_OUT', '')

# Super-op table: (op1, op2) -> fused handler name. Populated from a prior
# profile run (see load_hot_pairs_from_file). Empty by default so the slow
# loop does one extra dict lookup per op; the hit rate is what makes it
# amortize.
_hot_pair_handlers = {}


class DispatchProfiler(object):
    """Singleton profiler. Keeps pair counts keyed by (jitcode_name, op1, op2).

    The slow dispatch loop in pyjitpl.run_one_step calls record_pair(prev, cur)
    once per iteration after dispatching `prev`. We keep two levels of maps
    to keep the inner loop cheap; aggregation happens at report time.
    """

    _instance = None

    def __init__(self):
        # (jitcode_name, prev_op, cur_op) -> count
        self.pair_counts = {}
        # (jitcode_name, prev_op, cur_op, next_op) -> count.
        # Sliding-window triples, derived cheaply from record_pair via the
        # `_last_pair` short-term cursor below. Triples tell us when a
        # hot pair consistently extends into a third op (e.g.
        # add -> lt -> guard_true), which is exactly what chain-fusion
        # needs to prioritize in super_op_table.
        self.triple_counts = {}
        # (jitcode_name, op) -> count, for sanity-check vs opcode_counters
        self.single_counts = {}
        # jitcode_name -> number of times its dispatch loop was entered
        self.jitcode_entries = {}
        # sliding-window cursor: jitcode_name -> (prev_prev, prev) so we
        # can form a triple when the next pair arrives. Cleared per
        # record_entry so triples don't cross jitcode frame boundaries.
        self._last_pair = {}
        # populated by pyjitpl.MetaInterpStaticData.setup_insns when the flag
        # is active; lets the atexit report translate op ids to names.
        self._opcode_names = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            if DISPATCH_PROFILE_ENABLED:
                atexit.register(cls._instance._atexit_report)
        return cls._instance

    def _atexit_report(self):
        self.print_report(self._opcode_names)
        if _OUT:
            self.dump_json(_OUT, self._opcode_names)

    def record_entry(self, jitcode_name):
        if not DISPATCH_PROFILE_ENABLED:
            return
        self.jitcode_entries[jitcode_name] = \
            self.jitcode_entries.get(jitcode_name, 0) + 1
        # Reset the triple cursor so triples don't span jitcode frames.
        self._last_pair.pop(jitcode_name, None)

    def record_pair(self, jitcode_name, prev_op, cur_op):
        if not DISPATCH_PROFILE_ENABLED:
            return
        key = (jitcode_name, prev_op, cur_op)
        self.pair_counts[key] = self.pair_counts.get(key, 0) + 1
        skey = (jitcode_name, prev_op)
        self.single_counts[skey] = self.single_counts.get(skey, 0) + 1
        # Triple bookkeeping: if we saw pair (x, prev) on the previous
        # call, then (x, prev, cur) is a triple hit right now.
        last = self._last_pair.get(jitcode_name)
        if last is not None:
            prev_prev = last[1]
            # Only count when the sliding window is contiguous, i.e.
            # last_cur == current prev_op. Otherwise there was a jitcode
            # frame boundary or a skipped op.
            if last[1] == prev_op:
                tkey = (jitcode_name, last[0], prev_op, cur_op)
                self.triple_counts[tkey] = (
                    self.triple_counts.get(tkey, 0) + 1)
                prev_prev = last[0]
            _ = prev_prev  # silence unused
        self._last_pair[jitcode_name] = (prev_op, cur_op)

    def top_pairs(self, n=_TOP, opcode_names=None):
        items = sorted(self.pair_counts.items(), key=lambda kv: -kv[1])
        out = []
        for (jc, a, b), c in items[:n]:
            if opcode_names is not None:
                na = opcode_names[a] if 0 <= a < len(opcode_names) else str(a)
                nb = opcode_names[b] if 0 <= b < len(opcode_names) else str(b)
            else:
                na, nb = str(a), str(b)
            out.append((jc, na, nb, c))
        return out

    def total_pair_hits(self):
        return sum(self.pair_counts.values())

    def total_triple_hits(self):
        return sum(self.triple_counts.values())

    def top_triples(self, n=_TOP, opcode_names=None):
        items = sorted(self.triple_counts.items(), key=lambda kv: -kv[1])
        out = []
        for (jc, a, b, c), count in items[:n]:
            if opcode_names is not None:
                na = opcode_names[a] if 0 <= a < len(opcode_names) else str(a)
                nb = opcode_names[b] if 0 <= b < len(opcode_names) else str(b)
                nc = opcode_names[c] if 0 <= c < len(opcode_names) else str(c)
            else:
                na, nb, nc = str(a), str(b), str(c)
            out.append((jc, na, nb, nc, count))
        return out

    def print_report(self, opcode_names=None):
        if not self.pair_counts:
            return
        total = self.total_pair_hits()
        print("\n" + "=" * 80)
        print("DISPATCH-PAIR PROFILER REPORT (pairs=%d, total=%d)" %
              (len(self.pair_counts), total))
        print("=" * 80)
        print("%-35s %-25s %-25s %12s %7s" %
              ("jitcode", "op_prev", "op_cur", "count", "share%"))
        print("-" * 110)
        for jc, na, nb, c in self.top_pairs(_TOP, opcode_names):
            pct = 100.0 * c / total if total else 0.0
            print("%-35s %-25s %-25s %12d %6.2f%%" % (jc[:35], na[:25], nb[:25], c, pct))
        print("=" * 80)

    def dump_json(self, path, opcode_names=None):
        payload = {
            'total_pair_hits': self.total_pair_hits(),
            'total_triple_hits': self.total_triple_hits(),
            'jitcode_entries': self.jitcode_entries,
            'pairs': [
                {'jitcode': jc, 'prev': a, 'cur': b, 'count': c}
                for (jc, a, b), c in self.pair_counts.items()
            ],
            'triples': [
                {'jitcode': jc, 'a': a, 'b': b, 'c': c, 'count': count}
                for (jc, a, b, c), count in self.triple_counts.items()
            ],
        }
        if opcode_names is not None:
            payload['opcode_names'] = list(opcode_names)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def get_profiler():
    return DispatchProfiler.get_instance()


def record_pair(jitcode_name, prev_op, cur_op):
    """Fast path used from pyjitpl.run_one_step. Gated by the module flag so
    a disabled profile is a single truthy check."""
    if DISPATCH_PROFILE_ENABLED:
        DispatchProfiler.get_instance().record_pair(jitcode_name, prev_op, cur_op)


def record_entry(jitcode_name):
    if DISPATCH_PROFILE_ENABLED:
        DispatchProfiler.get_instance().record_entry(jitcode_name)


# --- super-jitcode hot pair loading ----------------------------------------

def load_hot_pairs_from_file(path):
    """Load a hot-pair report from a prior profile run. Returns a set of
    (jitcode_name, prev_op_name, cur_op_name) tuples. Consumed by the
    GenExtension code generator to emit fused handlers at translation time.

    Format: the JSON emitted by dump_json above.
    """
    with open(path, 'r') as f:
        payload = json.load(f)
    names = payload.get('opcode_names') or []
    out = set()
    for p in payload.get('pairs', []):
        if not names:
            continue
        a, b = p['prev'], p['cur']
        if 0 <= a < len(names) and 0 <= b < len(names):
            out.add((p['jitcode'], names[a], names[b]))
    return out


def register_hot_pair(op1, op2, handler):
    """Register a fused handler for a (prev_op, cur_op) pair. Used by the
    GenExtension post-pass once hot pairs are known. Stored in a global so
    the dispatch loop can check without touching jitcode state."""
    _hot_pair_handlers[(op1, op2)] = handler


def lookup_hot_pair(op1, op2):
    return _hot_pair_handlers.get((op1, op2))


def clear_hot_pairs():
    _hot_pair_handlers.clear()
