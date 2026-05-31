"""Meta-interp test for enable_invariant_varindex_hoist: a loop-invariant
variable-index array read (a[k] with k invariant in the inner loop) should be
hoisted into the short preamble so the peeled loop body no longer re-reads it,
while producing identical results.
"""
from rpython.rlib.jit import JitDriver, set_param
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats
from rpython.jit.metainterp.resoperation import rop


def _loop_body_array_reads():
    """Max over compiled loops of getarrayitem_gc reads AFTER the last LABEL
    (i.e. in the peeled loop body)."""
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


class TestVarindexHoist(LLJitMixin):
    def _run(self, fb):
        myjitdriver = JitDriver(greens=[], reds=['n', 'jj', 'k', 'a', 'total'])

        def f(n, k, fb):
            set_param(myjitdriver, 'enable_invariant_varindex_hoist', fb)
            a = [0.0] * n
            i = 0
            while i < n:
                a[i] = float(i * 3 + 1)
                i += 1
            total = 0.0
            jj = 0
            while jj < n:
                myjitdriver.jit_merge_point(n=n, jj=jj, k=k, a=a, total=total)
                # a[k] is loop-invariant (k fixed); a[jj] varies per iteration.
                total += a[k] * a[jj]
                jj += 1
            return total
        res = self.meta_interp(f, [60, 4, fb])
        expected = 0.0
        a = [float(i * 3 + 1) for i in range(60)]
        for jj in range(60):
            expected += a[4] * a[jj]
        assert res == expected
        return _loop_body_array_reads()

    def test_invariant_varindex_is_hoisted(self):
        off = self._run(0)
        on = self._run(1)
        # flag off: both a[k] and a[jj] read in the body (>=2).
        # flag on: a[k] hoisted, only a[jj] remains -> strictly fewer.
        assert off >= 2
        assert on < off
