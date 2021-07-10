import math
import sys

import py
import weakref

from rpython.rlib import rgc
from rpython.jit.codewriter.policy import StopAtXPolicy
from rpython.jit.metainterp import history
from rpython.jit.metainterp.test.support import LLJitMixin, noConst
from rpython.jit.metainterp.warmspot import get_stats
from rpython.jit.metainterp.pyjitpl import MetaInterp
from rpython.rlib import rerased
from rpython.rlib.jit import (JitDriver, we_are_jitted, hint, dont_look_inside,
    loop_invariant, elidable, promote, jit_debug, assert_green,
    AssertGreenFailed, unroll_safe, current_trace_length, look_inside_iff,
    isconstant, isvirtual, set_param, record_exact_class)
from rpython.rlib.longlong2float import float2longlong, longlong2float
from rpython.rlib.rarithmetic import ovfcheck, is_valid_int, int_force_ge_zero
from rpython.rtyper.lltypesystem import lltype, rffi


class BasicTests:
    def test_basic(self):
        myjitdriver = JitDriver(greens = [], reds = ['y', 'res', 'x'])

        @dont_look_inside
        def add(x, y):
            return x + y

        @dont_look_inside
        def minus(x, y):
            return x - y

        def interp(x, y):
            res = 0
            while y > 0:
                myjitdriver.can_enter_jit(x=x, y=y, res=res)
                myjitdriver.jit_merge_point(x=x, y=y, res=res)
                res = add(res, add(x, 1))
                y = minus(y, 1)
            return res
        res = self.meta_interp(interp, [10, 5])
        assert res == 55
        self.check_trace_count(1)

    def test_branching(self):
        @dont_look_inside
        def lt(x, y):
            return x < y

        @dont_look_inside
        def gt(x, y):
            return x > y

        @dont_look_inside
        def add(x, y):
            return x + y

        @dont_look_inside
        def sub(x, y):
            return x - y

        @dont_look_inside
        def emit_jump(x):
            return x

        myjitdriver = JitDriver(greens = [],
                                reds = ['y', 'x', 'res'])
        def interp(x, y):
            res = 0
            while True:
                myjitdriver.can_enter_jit(x=x, y=y, res=res)
                myjitdriver.jit_merge_point(x=x, y=y, res=res)
                y = sub(y, 1)
                res = add(res, x)
                if lt(y, 0):
                    return res
                else:
                    if we_are_jitted():
                        res = emit_jump(res)
                        # XXX: pseudo-reproduction of method-traversing
                        return res

        res = self.meta_interp(interp, [10, 10])

class TestLLtype(BasicTests, LLJitMixin):
    pass
