import math
import sys

import py
import weakref

from rpython.rlib import rgc
from rpython.jit.codewriter.policy import StopAtXPolicy
from rpython.jit.metainterp import history
from rpython.jit.metainterp.test.support import LLJitMixin, noConst, get_stats
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


def compile_threaded_code():
    pass

class BasicTests:
    def test_minilang_1(self):

        @dont_look_inside
        def lt(x, y):
            if x < y:
                return 1
            else:
                return 0

        @dont_look_inside
        def add(x, y):
            return x + y

        @dont_look_inside
        def sub(x, y):
            return x - y

        @dont_look_inside
        def is_true(x):
            return x != 0

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x):
            return x

        ADD = -1
        SUB = 0
        LT = 1
        JUMP = 2
        JUMP_IF = 3
        JUMP = 4
        EXIT = 5
        myjitdriver = JitDriver(greens=['pc', 'bytecode',], reds=['x', 'res'],
                                threaded_code_gen=True)
        def interp(x):
            pc = 0
            res = x
            bytecode = [LT, JUMP_IF, 6, SUB, JUMP, 0, EXIT]
            while True:
                myjitdriver.can_enter_jit(pc=pc, bytecode=bytecode, x=x, res=res)
                myjitdriver.jit_merge_point(pc=pc, bytecode=bytecode, x=x, res=res)
                op = bytecode[pc]
                pc += 1
                if op == ADD:
                    res = add(res, 1)
                elif op == SUB:
                    res = sub(res, 1)
                elif op == JUMP:
                    t = bytecode[pc]
                    pc = t
                elif op == JUMP_IF:
                    t = bytecode[pc]
                    if is_true(x):
                        pc = t
                        if we_are_jitted():
                            pc = emit_jump(pc, x, res)
                            pc += 1
                    else:
                        pc += 1
                        if we_are_jitted():
                            pc = emit_jump(t, x, res)
                            pc = t
                            res = emit_ret(res)
                elif op == LT:
                    x = lt(res, 0)
                elif op == EXIT:
                    return res

        interp.oopspec = 'jit.not_in_trace()'
        res = self.meta_interp(interp, [10])
        # get_stats().loops[0].operations

class TestLLtype(BasicTests, LLJitMixin):
    pass
