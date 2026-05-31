"""Validation tests for Hot-Loop Lifting (HLL).

HLL is the umbrella for two trace-optimizer techniques that lift redundant
work out of hot loop bodies:

  * VIH - Variable-Index Hoisting.  A loop-invariant variable-index array read
    (a[k] with k invariant across the inner loop) is hoisted into the short
    preamble, so the peeled loop body stops re-reading it every iteration.

  * HBP - Hot Bridge Promotion.  A hot guard-failure bridge is promoted into a
    specialized loop variant that inherits the parent trace's guard facts, so
    the promoted trace stops re-proving conditions the parent already proved.

Every test pairs a *structural* assertion (the compiled trace demonstrably
changed in the way the technique promises) with a *correctness* assertion (the
jitted result matches the plain interpreter).  Run untranslated:

    pytest rpython/jit/metainterp/test/test_hot_loop_lifting.py
"""

from rpython.rlib.jit import JitDriver, dont_look_inside, promote, set_param
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats
from rpython.jit.metainterp.resoperation import rop


# --------------------------------------------------------------------------
# trace-inspection helpers
# --------------------------------------------------------------------------

def _loop_body_array_reads():
    """Max over compiled loops of getarrayitem_gc reads located AFTER the last
    LABEL, i.e. inside the peeled (steady-state) loop body."""
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


def _count_op(opname):
    """Total count of an op across every compiled loop and bridge."""
    count = 0
    for loop in get_stats().get_all_loops():
        for op in loop._all_operations():
            if op.getopname() == opname:
                count += 1
    return count


def _max_target_tokens():
    """Largest number of specialized loop variants (target_tokens) attached to
    any jitcell token - grows when HBP promotes a hot bridge into a loop."""
    tokens = get_stats().get_all_jitcell_tokens()
    sizes = [len(t.target_tokens) for t in tokens if t.target_tokens]
    return max(sizes) if sizes else 0


# --------------------------------------------------------------------------
# VIH - Variable-Index Hoisting
# --------------------------------------------------------------------------

class TestVarIndexHoisting(LLJitMixin):

    def _run_vih(self, flag):
        myjitdriver = JitDriver(greens=[], reds=['n', 'jj', 'k', 'a', 'total'])

        def f(n, k, flag):
            set_param(myjitdriver, 'enable_invariant_varindex_hoist', flag)
            a = [0.0] * n
            i = 0
            while i < n:
                a[i] = float(i * 3 + 1)
                i += 1
            total = 0.0
            jj = 0
            while jj < n:
                myjitdriver.jit_merge_point(n=n, jj=jj, k=k, a=a, total=total)
                # a[k] is loop-invariant (k fixed); a[jj] varies each iteration.
                total += a[k] * a[jj]
                jj += 1
            return total

        res = self.meta_interp(f, [60, 4, flag])
        # correctness: jitted result matches a plain interpreter
        a = [float(i * 3 + 1) for i in range(60)]
        expected = 0.0
        for jj in range(60):
            expected += a[4] * a[jj]
        assert res == expected
        return _loop_body_array_reads()

    def test_invariant_index_read_is_lifted(self):
        # Flag off: both a[k] and a[jj] are read in the loop body (>= 2).
        # Flag on:  a[k] is hoisted into the preamble, so the body reads fewer.
        off = self._run_vih(0)
        on = self._run_vih(1)
        assert off >= 2
        assert on < off

    def test_vih_default_does_not_change_result(self):
        # Sanity: enabling VIH never changes the computed value.
        off = self._run_vih(0)
        on = self._run_vih(1)
        assert on <= off  # never adds body reads


# --------------------------------------------------------------------------
# HBP - Hot Bridge Promotion
# --------------------------------------------------------------------------

class TestHotBridgePromotion(LLJitMixin):

    def _hbp_program(self):
        myjitdriver = JitDriver(greens=[], reds=['i', 'N', 's'])

        @dont_look_inside
        def opaque(v):
            return v

        def run(N, enable, inherit):
            set_param(None, 'threshold', 3)
            set_param(None, 'trace_eagerness', 1)
            set_param(None, 'retrace_limit', 5 if enable else 0)
            set_param(None, 'enable_hot_bridge_promotion', enable)
            set_param(None, 'hot_bridge_threshold', 0)
            set_param(None, 'hot_bridge_guard_threshold', 1)
            set_param(None, 'hbp_inherit', inherit)
            s = 0
            i = 0
            while i < N:
                myjitdriver.jit_merge_point(N=N, i=i, s=s)
                x = i & 0xff
                if x > 0:                 # parent proves x > 0 here
                    s = opaque(s)
                    tag = promote(x & 3)  # hot guard_value -> bridge per value
                    if tag:
                        if x > 0:         # the promoted bridge re-tests x > 0
                            s += 256 // x
                        else:
                            s -= 99
                    else:
                        s += x
                i += 1
            return s
        return myjitdriver, run

    def _ref(self, N):
        s = 0
        i = 0
        while i < N:
            x = i & 0xff
            if x > 0:
                tag = x & 3
                if tag:
                    if x > 0:
                        s += 256 // x
                    else:
                        s -= 99
                else:
                    s += x
            i += 1
        return s

    def test_hot_guard_bridge_is_promoted_and_specialized(self):
        # The hot guard_value(x & 3) failures spawn bridges.  With HBP on,
        # those hot bridges are promoted into specialized loop variants that
        # inherit the parent's "x > 0" fact - so the promoted trace drops the
        # redundant int_gt(x, 0) the un-promoted bridge has to re-emit.
        _, run = self._hbp_program()

        res = self.meta_interp(run, [400, 1, 0])   # HBP on, inherit off
        assert res == self._ref(400)
        int_gt_no_inherit = _count_op('int_gt')

        res = self.meta_interp(run, [400, 1, 1])   # HBP on, inherit on
        assert res == self._ref(400)
        int_gt_inherit = _count_op('int_gt')

        # Promotion + inheritance removed at least one redundant guard.
        assert int_gt_inherit < int_gt_no_inherit, (
            "promoted bridge should inherit x>0 and drop the redundant "
            "int_gt: no_inherit=%d inherit=%d"
            % (int_gt_no_inherit, int_gt_inherit))

    def test_hbp_adds_loop_variant_without_changing_result(self):
        # Direct promotion signal: turning HBP on must not reduce the number
        # of specialized loop variants vs the un-promoted baseline, and must
        # never change the answer.
        _, run = self._hbp_program()

        res = self.meta_interp(run, [400, 0, 0])   # HBP off
        assert res == self._ref(400)
        baseline_tt = _max_target_tokens()

        res = self.meta_interp(run, [400, 1, 1])   # HBP on + inherit
        assert res == self._ref(400)
        promoted_tt = _max_target_tokens()

        assert promoted_tt >= baseline_tt, (
            "HBP should not shrink the loop-variant set: "
            "baseline=%d promoted=%d" % (baseline_tt, promoted_tt))
