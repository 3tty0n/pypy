"""Phase 1 bridge reuse (docs/bridge_prefix_stitching_design.md).

When several guards share their resume data via ResumeGuardCopiedDescr.prev,
the first to compile a bridge registers it on the shared 'prev'; later siblings
stitch to that bridge instead of tracing their own.  These tests run on the
llgraph backend (whose stitch_bridge is the trivial
`faildescr._llgraph_bridge = target[0].lltrace`) and check that, with the flag
on, results stay correct and the stitch path is actually exercised.
"""
import py
from rpython.rlib.jit import JitDriver
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp import compile as compile_mod


class Phase1Mixin(object):
    def run_with_flag(self, func, args):
        """Run meta_interp with PHASE1_BRIDGE_REUSE on, counting stitch_bridge
        calls on the (llgraph) cpu.  Returns (result, n_stitches)."""
        old_flag = compile_mod.PHASE1_BRIDGE_REUSE
        counter = {'n': 0}

        # wrap the cpu's stitch_bridge to count and still perform it
        from rpython.jit.backend.llgraph.runner import LLGraphCPU
        old_stitch = LLGraphCPU.stitch_bridge

        def counting_stitch(self, faildescr, target):
            counter['n'] += 1
            return old_stitch(self, faildescr, target)

        compile_mod.PHASE1_BRIDGE_REUSE = True
        LLGraphCPU.stitch_bridge = counting_stitch
        try:
            res = self.meta_interp(func, args)
        finally:
            compile_mod.PHASE1_BRIDGE_REUSE = old_flag
            LLGraphCPU.stitch_bridge = old_stitch
        return res, counter['n']


class BaseTests(Phase1Mixin):
    def test_two_paths_correct_with_flag(self):
        # a simple bridge-forming loop: with the flag on, the result must be
        # identical to the pure-python result and nothing may crash.
        myjitdriver = JitDriver(greens=[], reds=['x', 'y', 'res'])

        def g(y, x):
            if y & 1:
                return x - 2
            return x + 3

        def f(x, y):
            res = 0
            while y > 0:
                myjitdriver.can_enter_jit(x=x, y=y, res=res)
                myjitdriver.jit_merge_point(x=x, y=y, res=res)
                res += g(y, x)
                y -= 1
            return res

        expected = f(6, 50)
        res, n_stitch = self.run_with_flag(f, [6, 50])
        assert res == expected

    def test_polymorphic_dispatch_correct_with_flag(self):
        # a polymorphic loop that produces several guard-class bridges; with the
        # flag on the result must stay correct, exercising the reuse path when
        # sibling guards share resume data.
        myjitdriver = JitDriver(greens=[], reds=['n', 'i', 'res', 'objs'])

        class W(object):
            pass

        class A(W):
            def val(self):
                return 1

        class B(W):
            def val(self):
                return 2

        class C(W):
            def val(self):
                return 3

        def make(n):
            objs = [None] * 6
            objs[0] = A(); objs[1] = B(); objs[2] = C()
            objs[3] = A(); objs[4] = B(); objs[5] = C()
            return objs

        def f(n):
            objs = make(n)
            res = 0
            i = 0
            while i < n:
                myjitdriver.can_enter_jit(n=n, i=i, res=res, objs=objs)
                myjitdriver.jit_merge_point(n=n, i=i, res=res, objs=objs)
                w = objs[i % 6]
                res += w.val()
                i += 1
            return res

        expected = f(300)
        res, n_stitch = self.run_with_flag(f, [300])
        assert res == expected


class TestLLtype(BaseTests, LLJitMixin):
    pass
