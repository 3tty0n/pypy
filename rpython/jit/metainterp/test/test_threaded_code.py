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
    def test_basic_fun_1(self):
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
        def ret(x):
            return x

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x):
            return x

        def f(x, y):
            if lt(x, y):
                return add(x, y)
            else:
                return sub(x, y)

        myjitdriver = JitDriver(greens=[], reds=['x', 'y'],
                                threaded_code_gen=True)

        def interp(x, y):
            while True:
                myjitdriver.can_enter_jit(x=x, y=y)
                myjitdriver.jit_merge_point(x=x, y=y)
                x = sub(x, 1)
                y = lt(x, y)
                if is_true(y):
                    if we_are_jitted():
                        x = emit_ret(x)
                    else:
                        return ret(x)
                else:
                    if we_are_jitted():
                        x = emit_jump(x, y, None)
                        return x
                    continue

        interp.oopspec = 'jit.not_in_trace()'
        res = self.meta_interp(interp, [20, 2])


    def test_minilang_1(self):

        class TStack:
            _immutable_fields_ = ['pc', 'next']

            def __init__(self, pc, next):
                self.pc = pc
                self.next = next

            def __repr__(self):
                return "TStack(%d, %s)" % (self.pc, repr(self.next))

            def t_pop(self):
                return self.pc, self.next

        memoization = {}

        @elidable
        def t_empty():
            return None

        @elidable
        def t_is_empty(tstack):
            return tstack is None or tstack.pc == -100

        @elidable
        def t_push(pc, next):
            key = pc, next
            if key in memoization:
                return memoization[key]
            result = TStack(pc, next)
            memoization[key] = result
            return result

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
        def ret(x):
            return x

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x, y):
            return x

        ADD = 0
        SUB = 1
        LT = 2
        JUMP = 3
        JUMP_IF = 4
        EXIT = 5
        NOP = -100
        inst_set = {
            0: "ADD",
            1: "SUB",
            2: "LT",
            3: "JUMP",
            4: "JUMP_IF",
            5: "EXIT",
            -100: "NOP"
        }
        def opcode_to_string(pc, bytecode, tstack):
            op = bytecode[pc]
            name = inst_set.get(op)
            return "%s: %s, tstack top: %s" % (pc, name, tstack.pc)

        myjitdriver = JitDriver(greens=['pc', 'bytecode', 'tstack'], reds=['x', 'res',],
                                get_printable_location=opcode_to_string,
                                threaded_code_gen=True
                                )
        def interp(x):
            # set_param(myjitdriver, 'threshold', 2)
            tstack = TStack(-100, None)
            pc = 0
            res = x
            bytecode = [NOP, LT, JUMP_IF, 6, SUB, JUMP, 1, EXIT]
            while True:
                myjitdriver.can_enter_jit(pc=pc, bytecode=bytecode, x=x, res=res, tstack=tstack)
                myjitdriver.jit_merge_point(pc=pc, bytecode=bytecode, x=x, res=res, tstack=tstack)
                op = bytecode[pc]
                pc += 1
                if op == ADD:
                    res = add(res, 1)
                elif op == SUB:
                    res = sub(res, 1)
                elif op == JUMP:
                    t = int(bytecode[pc])
                    if we_are_jitted():
                        if t_is_empty(tstack):
                            pc = t
                        else:
                            pc, tstack = tstack.t_pop()
                            pc = emit_jump(pc, t, None)
                    else:
                        # if t < pc:
                        #     myjitdriver.can_enter_jit(pc=pc, bytecode=bytecode, x=x, res=res, tstack=tstack)
                        pc = t
                elif op == JUMP_IF:
                    t = int(bytecode[pc])
                    if is_true(x):
                        if we_are_jitted():
                            pc += 1
                            tstack = t_push(pc, tstack)
                        # else:
                            # if t < pc:
                            #     myjitdriver.can_enter_jit(pc=pc, bytecode=bytecode, x=x, res=res, tstack=tstack)
                        pc = t
                    else:
                        if we_are_jitted():
                            tstack = t_push(t, tstack)
                        pc += 1
                elif op == LT:
                    x = lt(res, 0)
                elif op == EXIT:
                    if we_are_jitted():
                        if t_is_empty(tstack):
                            return ret(res)
                        else:
                            pc, tstack = tstack.t_pop()
                            pc = emit_ret(pc, ret)
                    else:
                        return ret(res)

        interp.oopspec = 'jit.not_in_trace()'
        res = self.meta_interp(interp, [10])
        # get_stats().loops[0].operations

class TestLLtype(BasicTests, LLJitMixin):
    pass
