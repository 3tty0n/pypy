"""Lever C spec/correctness test: classical loop unrolling (loop_unroll_factor)
composed with invariant variable-index hoisting (VIH).

The hot inner loop reads a loop-invariant variable index a[k] and a per-iteration
index a[jj].  Under unrolling the body is traced K times; with VIH on, a[k] must
be hoisted once into the short preamble and shared across the K unrolled copies
(not re-read per copy), while producing results identical to the interpreter for
every (factor, VIH) combination.
"""
from rpython.rlib.jit import JitDriver, set_param, dont_look_inside
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats
from rpython.jit.metainterp.resoperation import rop


@dont_look_inside
def _residual(x):
    return x + 1


def _max_body_array_reads():
    """Max over compiled loops of getarrayitem_gc reads after the last LABEL."""
    best = 0
    for loop in get_stats().get_all_loops():
        ops = loop.operations
        label_idx = -1
        for i, op in enumerate(ops):
            if op.getopnum() == rop.LABEL:
                label_idx = i
        cnt = 0
        for op in ops[label_idx + 1:]:
            if op.getopname().startswith('getarrayitem_gc'):
                cnt += 1
        if cnt > best:
            best = cnt
    return best


class UnrollVIHTests(object):
    def _make(self):
        driver = JitDriver(greens=[], reds=['n', 'jj', 'k', 'a', 'total'])

        def inner(n, k, a):
            total = 0.0
            jj = 0
            while jj < n:
                driver.jit_merge_point(n=n, jj=jj, k=k, a=a, total=total)
                # a[k] invariant in this loop; a[jj] varies per iteration.
                total += a[k] * a[jj]
                jj += 1
            return total

        def f(outer, n, k, factor, vih, metric=1, phase=0):
            set_param(driver, 'threshold', 1)
            set_param(driver, 'loop_unroll_factor', factor)
            set_param(driver, 'loop_unroll_warmup', 2)
            set_param(driver, 'loop_unroll_trial', 1)
            set_param(driver, 'loop_unroll_min_gain', 0)
            set_param(driver, 'loop_unroll_metric', metric)
            set_param(driver, 'loop_unroll_phase', phase)
            set_param(driver, 'enable_invariant_varindex_hoist', vih)
            a = [0.0] * n
            i = 0
            while i < n:
                a[i] = float(i * 3 + 1)
                i += 1
            tot = 0.0
            outer_i = 0
            while outer_i < outer:
                tot += inner(n, k, a)
                outer_i += 1
            return tot

        return f

    def _interp(self, outer, n, k):
        a = [float(i * 3 + 1) for i in range(n)]
        per = 0.0
        for jj in range(n):
            per += a[k] * a[jj]
        return per * outer

    def test_unroll_vih_correctness(self):
        # The full grid must equal the interpreter: unrolling x VIH must never
        # change results, in either A/B outcome (min_gain=0 lets fK be adopted).
        f = self._make()
        expected = self._interp(60, 40, 4)
        for factor in [1, 2, 4]:
            for vih in [0, 1]:
                res = self.meta_interp(f, [60, 40, 4, factor, vih, 1, 0])
                assert res == expected

    def test_unroll_vih_robust_metric_correct(self):
        # The robust decision metric (loop_unroll_metric=2: adopt fK only if min
        # AND mean agree) must also produce interpreter-identical results.
        f = self._make()
        expected = self._interp(60, 40, 4)
        for factor in [1, 2, 4]:
            res = self.meta_interp(f, [60, 40, 4, factor, 1, 2, 0])
            assert res == expected

    def test_unroll_vih_phase_metric_correct(self):
        # Exclusive-phase trial (loop_unroll_phase>0) must also produce
        # interpreter-identical results in both A/B outcomes.
        f = self._make()
        expected = self._interp(60, 40, 4)
        for factor in [1, 2, 4]:
            res = self.meta_interp(f, [60, 40, 4, factor, 1, 2, 8])
            assert res == expected

    def test_unroll_vih_does_not_increase_body_reads(self):
        # With VIH on, unrolling must not multiply the invariant read across the
        # K unrolled bodies: body array-reads under (factor>1, VIH on) must not
        # exceed (factor==1, VIH on) x K.  This is the optimisation-effectiveness
        # gate -- it fails if the hoist is duplicated per peeled copy.
        f = self._make()
        self.meta_interp(f, [60, 40, 4, 1, 1, 1, 0])
        base = _max_body_array_reads()
        self.meta_interp(f, [60, 40, 4, 2, 1, 1, 0])
        unrolled = _max_body_array_reads()
        assert unrolled <= base * 2


    def test_unroll_gate_call_loop_correct(self):
        # A loop with a residual call is non-numeric: the content gate keeps it
        # out of the A/B trial.  Results must stay correct under unrolling with
        # the gate both on (numeric_only=1) and off (0).
        driver = JitDriver(greens=[], reds=['n', 'res'])

        def f(start, factor, numeric_only):
            set_param(driver, 'threshold', 1)
            set_param(driver, 'loop_unroll_factor', factor)
            set_param(driver, 'loop_unroll_warmup', 2)
            set_param(driver, 'loop_unroll_numeric_only', numeric_only)
            n = start
            res = 0
            while n > 0:
                driver.jit_merge_point(n=n, res=res)
                res = _residual(res)   # residual call -> non-numeric loop
                n -= 1
            return res

        for factor in [1, 2, 4]:
            for numeric_only in [0, 1]:
                assert self.meta_interp(f, [30, factor, numeric_only]) == 30


class TestLLtype(UnrollVIHTests, LLJitMixin):
    pass
