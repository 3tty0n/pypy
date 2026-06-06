"""TLA interpreter frames and JIT drivers (extracted from tla.py).

Contains tier1driver, tier2driver, tier2vdriver, tier3driver, the tier-2
residual helpers (_t2_*), and Frame / JitFrame / JitFrame3.

Depends only on interp_helpers and object; imported by tla.run().
"""
import math
import os
import sys

from rpython.rlib import jit
from rpython.rlib.jit import JitDriver, we_are_jitted, hint, we_are_blackholing
from rpython.rlib.objectmodel import we_are_translated_to_c
from rpython.rlib.rarithmetic import r_uint
from rpython.rlib.rrandom import Random

from rpython.jit.tl.threadedcode.traverse_stack import TStack, t_empty, t_push
from rpython.jit.tl.threadedcode.tlib import emit_jump, emit_ret
from rpython.jit.tl.threadedcode.object import W_Object, W_IntObject, \
    W_FloatObject, W_StringObject, W_ListObject, OperationError
from rpython.jit.tl.threadedcode.bytecode import *

from rpython.jit.tl.threadedcode.interp_helpers import (
    ContinueInTracingJIT, ContinueInThreadedJIT,
    TRACE_THRESHOLD, MAX_INTERP_DEPTH,
    get_printable_location, get_printable_location_tier1,
    _construct_value, _branch_reaches_backedge,
    _entry_has_foreign_call_assembler, _entry_has_wide_call_assembler,
    _compute_stackdepth, _power_01, _construct_float,
    _tier1_confirm_enter_jit,
)


tier1driver = JitDriver(
    greens=['pc', 'entry', 'bytecode', 'tstack'], reds=['self'],
    get_printable_location=get_printable_location_tier1,
    confirm_enter_jit=_tier1_confirm_enter_jit,
    threaded_code_gen=True, conditions=["is_true"], is_recursive=True)


tier2driver = JitDriver(
    greens=['pc', 'bytecode',], reds=['self'],
    get_printable_location=get_printable_location, is_recursive=True)


# Virtualizable tier-2 driver (JitFrame._interp when jitted=True).
tier2vdriver = JitDriver(
    greens=['pc', 'bytecode',], reds=['self'],
    get_printable_location=get_printable_location,
    is_recursive=True, virtualizables=['self'])


def get_printable_location_tier3(pc, hybrid, bytecode):
    return get_printable_location(pc, bytecode)


# Tier-3/4 driver (JitFrame3).  `hybrid` is a green mode bit so adaptive
# profiling/hybrid traces do not occupy plain tier-3 JitCells.
tier3driver = JitDriver(
    greens=['pc', 'hybrid', 'bytecode'], reds=['self'],
    get_printable_location=get_printable_location_tier3,
    is_recursive=True, virtualizables=['self'])


class Frame(object):

    def __init__(self, bytecode, stack=None, stackpos=0, depth=0):
        if stack is None:
            stack = [None] * 64
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos
        self.depth = depth
        self._clean_stack = None
        self._clean_pos = 0
        self._clean_pc = -1
        self._frame_poisoned = False
        self._ever_poisoned = False

    @jit.unroll_safe
    def copy_frame(self, argnum, retaddr, dummy=False):

        oldstack = self.stack
        oldstackpos = self.stackpos
        framepos = oldstackpos - argnum - 1
        assert framepos >= 0

        newstack = [None] * len(self.stack)
        for i in range(framepos, oldstackpos):
            # j = oldstackpos - i - 1
            newstack[i - framepos] = oldstack[i]
        newstack[argnum + 1] = W_IntObject(retaddr)

        bytecode = jit.promote(self.bytecode)
        return Frame(bytecode, newstack, argnum + 2, depth=self.depth + 1)

    @jit.dont_look_inside
    def push(self, w_x):
        self.stack[self.stackpos] = w_x
        self.stackpos += 1

    @jit.push_raw_helper
    def _push(self, w_x):
        stackpos = jit.promote(self.stackpos)
        self.stack[stackpos] = w_x
        self.stackpos += 1

    @jit.dont_look_inside
    def pop(self):
        stackpos = self.stackpos - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    @jit.dont_look_inside
    @jit.pop_raw_helper
    def _pop(self):
        stackpos = jit.promote(self.stackpos) - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    @jit.dont_look_inside
    def take(self, n):
        assert len(self.stack) is not 0
        w_x = self.stack[self.stackpos - n - 1]
        assert w_x is not None
        return w_x

    def _take(self, n):
        assert len(self.stack) is not 0
        stackpos = jit.promote(self.stackpos)
        w_x = self.stack[stackpos - n - 1]
        assert w_x is not None
        return w_x

    @jit.dont_look_inside
    def drop(self, n):
        for _ in range(n):
            self.pop()

    @jit.unroll_safe
    def _drop(self, n):
        for _ in range(n):
            self._pop()

    @jit.not_in_trace
    def dump(self):
        sys.stderr.write("stackpos: %d " % self.stackpos)
        sys.stderr.write("[")
        for i in range(self.stackpos):
            w_x = self.stack[i]
            if isinstance(w_x, W_Object):
                sys.stderr.write(w_x.getrepr() + ", ")
        sys.stderr.write("]\n")

    @jit.dont_look_inside
    @jit.condition_helper
    def is_true(self, recorded, dummy):
        w_x = self.pop()
        if dummy and not we_are_blackholing():
            return recorded
        return w_x.is_true()

    def _is_true(self):
        w_x = self._pop()
        return w_x.is_true()

    def _CONST_INT(self, pc, neg=False):
        if isinstance(pc, int):
            bytecode = jit.promote(self.bytecode)
            x = ord(bytecode[pc])
            if neg:
                self._push(W_IntObject(-x))
            else:
                self._push(W_IntObject(x))
        else:
            raise OperationError

    def _CONST_FLOAT(self, pc, neg=False):
        if isinstance(pc, int):
            bytecode = jit.promote(self.bytecode)
            x = _construct_float(bytecode, pc)
            if neg:
                self._push(W_FloatObject(-x))
            else:
                self._push(W_FloatObject(x))
        else:
            raise OperationError

    def _CONST_N(self, pc):
        if isinstance(pc, int):
            bytecode = jit.promote(self.bytecode)
            x = _construct_value(bytecode, pc)
            self._push(W_IntObject(x))
        else:
            raise OperationError

    @jit.dont_look_inside
    @jit.push_helper
    def PUSH(self, w_x, dummy):
        if dummy and not we_are_blackholing():
            return
        self.push(w_x)

    def _PUSH(self, w_x):
        self._push(w_x)

    @jit.dont_look_inside
    @jit.pop_helper
    def POP(self, dummy):
        if dummy and not we_are_blackholing():
            return self.take(0)
        return self.pop()

    def _POP(self):
        return self._pop()

    @jit.dont_look_inside
    @jit.drop_helper
    def DROP(self, n, dummy):
        if dummy and not we_are_blackholing():
            return
        for _ in range(n):
            self.pop()

    @jit.unroll_safe
    def _DROP(self, n):
        for _ in range(n):
            self._pop()

    def _POP1(self):
        v = self._pop()
        _ = self._pop()
        self._push(v)

    def _ADD(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.add(w_y, True)
        else:
            w_z = w_x.add(w_y, False)
        self._push(w_z)

    def _SUB(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.sub(w_y, True)
        else:
            w_z = w_x.sub(w_y, False)
        self._push(w_z)

    def _MUL(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.mul(w_y, True)
        else:
            w_z = w_x.mul(w_y, False)
        self._push(w_z)

    def _DIV(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.div(w_y, True)
        else:
            w_z = w_x.div(w_y, False)
        self._push(w_z)

    def _MOD(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.mod(w_y, True)
        else:
            w_z = w_x.mod(w_y, False)
        self._push(w_z)

    def _DUP(self):
        w_x = self._pop()
        self._push(w_x)
        self._push(w_x)

    def _DUPN(self, pc):
        bytecode = jit.promote(self.bytecode)
        n = ord(bytecode[pc])
        w_x = self._take(n)
        self._push(w_x)

    def _LT(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.le(w_y, True)
        else:
            w_z = w_x.le(w_y, False)
        self._push(w_z)

    def _GT(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_z = w_x.ge(w_y, True)
        else:
            w_z = w_x.ge(w_y, False)
        self._push(w_z)

    def _EQ(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            self._push(w_x.eq(w_y, True))
        else:
            self._push(w_x.eq(w_y, False))

    def _NE(self):
        w_y = self._pop()
        w_x = self._pop()
        if we_are_jitted():
            w_eq = w_x.eq(w_y, True)
        else:
            w_eq = w_x.eq(w_y, False)
        if w_eq.intvalue:
            self._push(W_IntObject(1))
        else:
            self._push(W_IntObject(0))

    @jit.enable_shallow_tracing
    def CONST_INT(self, pc):
        bytecode = jit.promote(self.bytecode)
        self._push(W_IntObject(ord(bytecode[pc])))

    @jit.enable_shallow_tracing
    def CONST_NEG_INT(self, pc):
        bytecode = jit.promote(self.bytecode)
        self._push(W_IntObject(-ord(bytecode[pc])))

    @jit.enable_shallow_tracing
    def CONST_N(self, pc):
        bytecode = jit.promote(self.bytecode)
        self._push(W_IntObject(_construct_value(bytecode, pc)))

    @jit.enable_shallow_tracing
    def DUP(self):
        w_x = self._pop()
        self._push(w_x)
        self._push(w_x)

    @jit.enable_shallow_tracing
    def DUPN(self, pc):
        bytecode = jit.promote(self.bytecode)
        self._push(self._take(ord(bytecode[pc])))

    @jit.enable_shallow_tracing
    def POP1(self):
        v = self._pop()
        self._pop()
        self._push(v)

    @jit.enable_shallow_tracing
    def LT(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.le_inline(w_y))

    @jit.enable_shallow_tracing
    def GT(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.ge_inline(w_y))

    @jit.enable_shallow_tracing
    def EQ(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.eq_inline(w_y))

    @jit.enable_shallow_tracing
    def ADD(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.add_inline(w_y))

    @jit.enable_shallow_tracing
    def SUB(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.sub_inline(w_y))

    @jit.enable_shallow_tracing
    def MUL(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.mul_inline(w_y))

    @jit.enable_shallow_tracing
    def DIV(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.div_inline(w_y))

    @jit.enable_shallow_tracing
    def MOD(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(w_x.mod_inline(w_y))

    @jit.enable_shallow_tracing
    def INT_TO_FLOAT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            self._push(W_FloatObject(float(w_x.intvalue)))
        else:
            self._push(W_FloatObject(0.0))

    @jit.enable_shallow_tracing
    def FLOAT_TO_INT(self):
        w_x = self._pop()
        if isinstance(w_x, W_FloatObject):
            self._push(W_IntObject(int(w_x.floatvalue)))
        else:
            self._push(W_IntObject(0))

    @jit.enable_shallow_tracing
    def ABS_FLOAT(self):
        w_x = self._pop()
        if isinstance(w_x, W_FloatObject):
            self._push(W_FloatObject(abs(w_x.floatvalue)))
        else:
            self._push(W_FloatObject(0.0))

    @jit.enable_shallow_tracing
    def SQRT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            self._push(W_FloatObject(math.sqrt(w_x.intvalue)))
        elif isinstance(w_x, W_FloatObject):
            self._push(W_FloatObject(math.sqrt(w_x.floatvalue)))
        else:
            self._push(W_FloatObject(0.0))

    @jit.enable_shallow_tracing
    def SIN(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            self._push(W_FloatObject(math.sin(w_x.intvalue)))
        elif isinstance(w_x, W_FloatObject):
            self._push(W_FloatObject(math.sin(w_x.floatvalue)))
        else:
            self._push(W_FloatObject(0.0))

    @jit.enable_shallow_tracing
    def COS(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            self._push(W_FloatObject(math.cos(w_x.intvalue)))
        elif isinstance(w_x, W_FloatObject):
            self._push(W_FloatObject(math.cos(w_x.floatvalue)))
        else:
            self._push(W_FloatObject(0.0))

    @jit.enable_shallow_tracing
    def BUILD_LIST(self):
        size = self._pop()
        init = self._pop()
        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self._push(W_ListObject(lst))

    @jit.enable_shallow_tracing
    def LOAD(self):
        w_index = self._pop()
        w_lst = self._pop()
        if isinstance(w_lst, W_ListObject) and isinstance(w_index, W_IntObject):
            idx = int(w_index.intvalue)
            if 0 <= idx < len(w_lst.listvalue):
                self._push(w_lst.listvalue[idx])
                return
        self._push(W_IntObject(0))

    @jit.enable_shallow_tracing
    def STORE(self):
        w_index = self._pop()
        w_lst = self._pop()
        w_x = self._pop()
        if isinstance(w_lst, W_ListObject) and isinstance(w_index, W_IntObject):
            idx = int(w_index.intvalue)
            if 0 <= idx < len(w_lst.listvalue):
                w_lst.listvalue[idx] = w_x
            self._push(w_lst)
            return
        self._push(w_lst if isinstance(w_lst, W_ListObject) else W_IntObject(0))

    def _ADD_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue + w_y.intvalue))
        else:
            self._push(w_x.add_inline(w_y))

    def _SUB_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue - w_y.intvalue))
        else:
            self._push(w_x.sub_inline(w_y))

    def _MUL_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(int(w_x.intvalue * w_y.intvalue)))
        else:
            self._push(w_x.mul_inline(w_y))

    def _DIV_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue // w_y.intvalue))
        else:
            self._push(w_x.div_inline(w_y))

    def _MOD_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue % w_y.intvalue))
        else:
            self._push(w_x.mod_inline(w_y))

    def _LT_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue <= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.le_inline(w_y))

    def _GT_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue >= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.ge_inline(w_y))

    def _EQ_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue == w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.eq_inline(w_y))

    def _NE_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue != w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            w_eq = w_x.eq(w_y, False)
            if w_eq.intvalue:
                self._push(W_IntObject(0))
            else:
                self._push(W_IntObject(1))

    @jit.dont_look_inside
    def CALL(self, oldframe, t, argnum, dummy):
        if dummy and not we_are_blackholing():
            return
        w_x = self.interp(t)
        oldframe.drop(argnum)
        if w_x:
            oldframe.push(w_x)

    def CALL_ASSEMBLER(self, oldframe, t, argnum, bytecode,
                       tstack, dummy):
        "Special handler to be compiled to call_assembler_r"
        w_x = self.interp_CALL_ASSEMBLER(t, t, bytecode,
                                         tstack, dummy)
        oldframe.DROP(argnum, dummy)
        if w_x:
            oldframe.PUSH(w_x, dummy)

    def _CALL(self, oldframe, t, argnum):
        w_x = self._interp(t)
        oldframe._drop(argnum)
        if w_x:
            oldframe._push(w_x)

    @jit.dont_look_inside
    @jit.ret_helper
    def RET(self, n, dummy):
        if dummy and not we_are_blackholing():
            return
        v = self.pop()
        return v

    def _RET(self, n):
        v = self._pop()
        return v

    def _PRINT(self):
        v = self._take(0)
        # print v.getrepr()

    @jit.dont_look_inside
    @jit.reset_helper
    def FRAME_RESET(self, o, l, n, dummy):
        ret = self.stack[self.stackpos - n - 1]
        old_base = self.stackpos - n
        new_base = self.stackpos - o - n - l - 1

        for i in range(n):
            self.stack[new_base + i] = self.stack[old_base + i]
            self.stack[old_base + i] = None

        self.stack[new_base + n] = ret
        self.stackpos = new_base + n + 1

    @jit.unroll_safe
    def _FRAME_RESET(self, o, l, n):
        stackpos = jit.promote(self.stackpos)
        ret = self.stack[stackpos - n - 1]
        old_base = stackpos - n
        new_base = stackpos - o - n - l - 1

        for i in range(n):
            self.stack[new_base + i] = self.stack[old_base + i]
            self.stack[old_base + i] = None

        self.stack[new_base + n] = ret
        self.stackpos = new_base + n + 1

    def _BUILD_LIST(self):
        size = self._pop()
        init = self._pop()

        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self._push(W_ListObject(lst))

    def _LOAD(self):
        w_index = self._pop()
        w_lst = self._pop()

        assert isinstance(w_index, W_IntObject)
        assert isinstance(w_lst, W_ListObject)

        w_x = w_lst.listvalue[int(w_index.intvalue)]
        self._push(w_x)

    def _STORE(self):
        w_index = self._pop()
        w_lst = self._pop()
        w_x = self._pop()

        assert isinstance(w_lst, W_ListObject)
        assert isinstance(w_index, W_IntObject)

        w_lst.listvalue[int(w_index.intvalue)] = w_x
        self._push(w_lst)

    def _RAND_INT(self):
        raise NotImplementedError

    def _COS(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.cos(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.cos(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SIN(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.sin(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.sin(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SQRT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(math.sqrt(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_x = W_FloatObject(math.sqrt(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_x)

    def _INT_TO_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_IntObject)
        w_x = W_FloatObject(float(w_x.intvalue))
        self._push(w_x)

    def _FLOAT_TO_INT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        w_x = W_IntObject(int(w_x.floatvalue))
        self._push(w_x)

    def _ABS_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        self._push(W_FloatObject(abs(w_x.floatvalue)))

    def _interp(self, pc=0):
        "tracing interpreter"
        if self.depth > MAX_INTERP_DEPTH:
            raise OperationError
        bytecode = self.bytecode

        while pc < len(bytecode):
            tier2driver.jit_merge_point(bytecode=bytecode, pc=pc, self=self)

            # print get_printable_location(pc, bytecode)

            opcode = ord(bytecode[pc])
            pc += 1

            if opcode == CONST_INT:
                self._CONST_INT(pc)
                pc += 1

            elif opcode == CONST_NEG_INT:
                self._CONST_INT(pc, neg=True)
                pc += 1

            elif opcode == CONST_FLOAT:
                self._CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self._CONST_FLOAT(pc, neg=True)
                pc += 9

            elif opcode == CONST_N:
                self._CONST_N(pc)
                pc += 4

            elif opcode == POP:
                self._POP()

            elif opcode == POP1:
                self._POP1()

            elif opcode == DUP:
                self._DUP()

            elif opcode == DUPN:
                self._DUPN(pc)
                pc += 1

            elif opcode == LT:
                self._LT_real()

            elif opcode == GT:
                self._GT_real()

            elif opcode == EQ:
                self._EQ_real()

            elif opcode == ADD:
                self._ADD_real()

            elif opcode == SUB:
                self._SUB_real()

            elif opcode == DIV:
                self._DIV_real()

            elif opcode == MUL:
                self._MUL_real()

            elif opcode == MOD:
                self._MOD_real()

            elif opcode == BUILD_LIST:
                self._BUILD_LIST()

            elif opcode == LOAD:
                self._LOAD()

            elif opcode == STORE:
                self._STORE()

            elif opcode == RAND_INT:
                self._RAND_INT()

            elif opcode == SIN:
                self._SIN()

            elif opcode == COS:
                self._COS()

            elif opcode == RAND_INT:
                self._RAND_INT()

            elif opcode == ABS_FLOAT:
                self._ABS_FLOAT()

            elif opcode == SQRT:
                self._SQRT()

            elif opcode == INT_TO_FLOAT:
                self._INT_TO_FLOAT()

            elif opcode == FLOAT_TO_INT:
                self._FLOAT_TO_INT()

            elif opcode == CALL_ASSEMBLER:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == CALL_N:
                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == RET:
                argnum = hint(ord(bytecode[pc]), promote=True)
                pc += 1
                w_x = self._RET(argnum)
                return w_x

            elif opcode == JUMP:
                t = ord(bytecode[pc])
                if t < pc:
                    if not we_are_jitted():
                        if bytecode.counts[pc-1] < TRACE_THRESHOLD:
                            raise ContinueInThreadedJIT(pc-1)

                    tier2driver.can_enter_jit(bytecode=bytecode, pc=t, self=self)

                pc = t

            elif opcode == JUMP_N:
                t = _construct_value(bytecode, pc)
                if t < pc:
                    if not we_are_jitted():
                        if bytecode.counts[pc-1] < TRACE_THRESHOLD:
                            raise ContinueInThreadedJIT(pc-1)

                    tier2driver.can_enter_jit(bytecode=bytecode, pc=t, self=self)

                pc = t

            elif opcode == JUMP_IF:
                t = ord(bytecode[pc])
                pc += 1

                if self._is_true():
                    if t < pc:
                        tier2driver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                    pc = t

            elif opcode == JUMP_IF_N:
                t = _construct_value(bytecode, pc)
                pc += 4

                if self._is_true():
                    if t < pc:
                        tier2driver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                    pc = t

            elif opcode == EXIT:
                return self._POP()

            elif opcode == PRINT:
                self._PRINT()

            elif opcode == FRAME_RESET:
                old_arity = ord(bytecode[pc])
                local_size = ord(bytecode[pc+1])
                new_arity = ord(bytecode[pc+2])
                pc += 3
                self._FRAME_RESET(old_arity, local_size, new_arity)

            elif opcode == NOP:
                continue

            else:
                assert False, 'Unknown opcode: %s' % bytecodes[opcode]

    @jit.dont_look_inside
    def interp_CALL_ASSEMBLER(self, pc, entry, bytecode, tstack, dummy):
        if dummy and not we_are_blackholing():
            return self.take(0)

        return self.interp(pc)


    def interp(self, pc=0):
        if self.depth > MAX_INTERP_DEPTH:
            raise OperationError
        tstack = t_empty()
        entry = pc
        bytecode = jit.promote(self.bytecode)

        while pc < len(bytecode):
            tier1driver.jit_merge_point(bytecode=bytecode, entry=entry,
                                        pc=pc, tstack=tstack, self=self)

            # print get_printable_location_tier1(pc, entry, bytecode, tstack)
            # self.dump()

            opcode = ord(bytecode[pc])
            pc += 1

            if opcode == CONST_INT:
                self.CONST_INT(pc)
                pc += 1

            elif opcode == CONST_NEG_INT:
                self.CONST_NEG_INT(pc)
                pc += 1

            elif opcode == CONST_FLOAT:
                self._CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self._CONST_FLOAT(pc, neg=True)
                pc += 9

            elif opcode == CONST_N:
                self.CONST_N(pc)
                pc += 4

            elif opcode == POP:
                self._POP()

            elif opcode == POP1:
                self.POP1()

            elif opcode == DUP:
                self.DUP()

            elif opcode == DUPN:
                self.DUPN(pc)
                pc += 1

            elif opcode == LT:
                self.LT()

            elif opcode == GT:
                self.GT()

            elif opcode == EQ:
                self.EQ()

            elif opcode == ADD:
                self.ADD()

            elif opcode == SUB:
                self.SUB()

            elif opcode == DIV:
                self.DIV()

            elif opcode == MUL:
                self.MUL()

            elif opcode == MOD:
                self.MOD()

            elif opcode == BUILD_LIST:
                self.BUILD_LIST()

            elif opcode == LOAD:
                self.LOAD()

            elif opcode == STORE:
                self.STORE()

            elif opcode == SIN:
                self.SIN()

            elif opcode == COS:
                self.COS()

            elif opcode == RAND_INT:
                self._RAND_INT()

            elif opcode == ABS_FLOAT:
                self.ABS_FLOAT()

            elif opcode == SQRT:
                self.SQRT()

            elif opcode == INT_TO_FLOAT:
                self.INT_TO_FLOAT()

            elif opcode == FLOAT_TO_INT:
                self.FLOAT_TO_INT()

            elif opcode == CALL:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL(self, t, argnum, dummy=True)
                else:
                    entry = t
                    if (t < pc and
                            not _entry_has_wide_call_assembler(bytecode, t)):
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=t, pc=t, tstack=tstack, self=frame)
                    frame.CALL(self, t, argnum, dummy=False)

            elif opcode == CALL_N:

                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=True)
                else:
                    if (t < pc and t == entry and
                            not _entry_has_wide_call_assembler(bytecode, t)):
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=t, pc=t,
                            tstack=t_empty(), self=frame)
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=t, pc=t,
                            tstack=t_empty(), self=frame)
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=False)

            elif opcode == CALL_ASSEMBLER:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=True)
                else:
                    if (t < pc and t == entry and
                            not _entry_has_wide_call_assembler(bytecode, t)):
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=t, pc=t,
                            tstack=t_empty(), self=frame)
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=t, pc=t,
                            tstack=t_empty(), self=frame)
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=False)

            elif opcode == RET:
                argnum = hint(ord(bytecode[pc]), promote=True)
                pc += 1
                if we_are_jitted():
                    if tstack.t_is_empty():
                        w_x = self.POP(dummy=True)
                        pc = emit_ret(entry, w_x)
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=pc, tstack=tstack, self=self)
                    else:
                        w_x = self.POP(dummy=True)
                        pc, tstack = tstack.t_pop()
                        pc = emit_ret(pc, w_x)
                else:
                    return self.RET(argnum, dummy=False)

            elif opcode == JUMP:
                t = ord(bytecode[pc])

                if we_are_jitted():
                    if tstack.t_is_empty():
                        if t < pc:
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                        pc = t
                    else:
                        pc, tstack = tstack.t_pop()

                    if t < pc:
                        emit_jump(pc, t)
                else:
                    if t < pc:
                        if bytecode.counts[pc-1] == TRACE_THRESHOLD:
                            raise ContinueInTracingJIT(pc-1)
                        bytecode.counts[pc-1] += 1
                    if (t < pc and tstack.t_is_empty() and
                            not _entry_has_wide_call_assembler(bytecode, entry) and
                            not _entry_has_foreign_call_assembler(
                                bytecode, entry)):
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                    pc = t

            elif opcode == JUMP_N:
                t = _construct_value(bytecode, pc)
                pc += 4

                if we_are_jitted():
                    if tstack.t_is_empty():
                        if t < pc:
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                        pc = t
                    else:
                        pc, tstack = tstack.t_pop()

                    if t < pc:
                        pc = emit_jump(pc, t)
                else:
                    if (t < pc and tstack.t_is_empty() and
                            not _entry_has_wide_call_assembler(bytecode, entry) and
                            not _entry_has_foreign_call_assembler(
                                bytecode, entry)):
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                    pc = t

            elif opcode == JUMP_IF:
                jumpif_pc = pc - 1
                target = ord(bytecode[pc])
                pc += 1

                if we_are_jitted():
                    recorded = (_branch_reaches_backedge(bytecode, target, jumpif_pc) or
                                not _branch_reaches_backedge(bytecode, pc, jumpif_pc))
                    if self.is_true(recorded, dummy=True):
                        tstack = t_push(pc, tstack)
                        pc = target
                    else:
                        tstack = t_push(target, tstack)
                else:
                    if self.is_true(True, dummy=False):
                        if (target < pc and tstack.t_is_empty() and
                                not _entry_has_wide_call_assembler(bytecode, target) and
                                not _entry_has_foreign_call_assembler(
                                    bytecode, target)):
                            entry = target
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=target, tstack=tstack, self=self)
                        pc = target

            elif opcode == JUMP_IF_N:
                jumpif_pc = pc - 1
                target = _construct_value(bytecode, pc)
                pc += 4

                if we_are_jitted():
                    recorded = (_branch_reaches_backedge(bytecode, target, jumpif_pc) or
                                not _branch_reaches_backedge(bytecode, pc, jumpif_pc))
                    if self.is_true(recorded, dummy=True):
                        tstack = t_push(pc, tstack)
                        pc = target
                    else:
                        tstack = t_push(target, tstack)

                else:
                    if self.is_true(True, dummy=False):
                        if (target < pc and tstack.t_is_empty() and
                                not _entry_has_wide_call_assembler(bytecode, target) and
                                not _entry_has_foreign_call_assembler(
                                    bytecode, target)):
                            entry = target
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=target, tstack=tstack, self=self)
                        pc = target

            elif opcode == EXIT:
                if we_are_jitted():
                    if tstack.t_is_empty():
                        w_x = self.POP(dummy=True)
                        pc = entry
                        pc = emit_ret(pc, w_x)
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=pc, tstack=tstack, self=self)
                    else:
                        w_x = self.POP(dummy=True)
                        pc, tstack = tstack.t_pop()
                        pc = emit_ret(pc, w_x)
                else:
                    return self.POP(dummy=False)

            elif opcode == PRINT:
                self._PRINT()

            elif opcode == FRAME_RESET:
                old_arity = ord(bytecode[pc])
                local_size = ord(bytecode[pc+1])
                new_arity = ord(bytecode[pc+2])
                pc += 3
                if we_are_jitted():
                    self.FRAME_RESET(old_arity, local_size, new_arity, dummy=True)
                else:
                    self.FRAME_RESET(old_arity, local_size, new_arity, dummy=False)

            elif opcode == NOP:
                continue

            else:
                assert False, 'Unknown opcode: %s' % bytecodes[opcode]


# Tier-2 residual operation helpers (_t2_*).
_INTCACHE_LO = -1
_INTCACHE_HI = 1024
_int_cache = [W_IntObject(_i) for _i in range(_INTCACHE_LO, _INTCACHE_HI + 1)]
_BOX_FALSE = _int_cache[0 - _INTCACHE_LO]
_BOX_TRUE = _int_cache[1 - _INTCACHE_LO]

@jit.elidable
def _intbox(n):
    if _INTCACHE_LO <= n <= _INTCACHE_HI:
        return _int_cache[n - _INTCACHE_LO]
    return W_IntObject(n)

class _T4Cfg(object):
    def __init__(self):
        self.ratio = 1
        self.freeze = 512
        self.minn = 50
        self.bailfloor = 8        # DRR: min off-type bails before a re-decision
        # CB4 adaptive hybrid-tier controller knobs (integer; instance fields so
        # TLA_* overrides are read at runtime, not constant-folded).
        self.cbmodel = 0          # master gate: 0 => legacy controller
        self.cnt_base = 100       # tier-up threshold: cnt_base + cnt_slope*len
        self.cnt_slope = 10
        self.cnt_maxinv = 3       # commit floor (profiling passes before commit)
        self.c_inl = 10           # cost: inlined op
        self.c_res = 64           # cost: residual call per op
        self.c_br_cmp = 200       # cost: off-type bridge for an inlined compare
        self.c_br_ari = 8         # cost: off-type bridge for inlined arithmetic
        self.recomp_base = 5000   # recompile cost: recomp_base + recomp_slope*len
        self.recomp_slope = 50
        self.horizon = 8          # future-execution multiplier on the saving
        self.reopt_base = 50      # reopt backoff: reopt_base * reopt_mult**retry
        self.reopt_mult = 2
        self.reopt_cap = 6        # max reopts (and backoff shift cap)

_t4cfg = _T4Cfg()

def _t4_set_poly(bytecode, site, value):
    if bytecode.poly[site] == value:
        return
    new = bytecode.poly[:]
    new[site] = value
    bytecode.poly = new

def _t4_configure():
    r = os.environ.get('TLA_POLY_RATIO')
    if r:
        _t4cfg.ratio = int(r)
    f = os.environ.get('TLA_FREEZE')
    if f:
        _t4cfg.freeze = int(f)
    m = os.environ.get('TLA_PROFILE_MIN')
    if m:
        _t4cfg.minn = int(m)
    bf = os.environ.get('TLA_BAIL_FLOOR')
    if bf:
        _t4cfg.bailfloor = int(bf)
    # CB4 adaptive controller knobs
    cm = os.environ.get('TLA_ADAPTIVE_MODEL')
    if cm:
        _t4cfg.cbmodel = int(cm)
    v = os.environ.get('TLA_CB_CNT_BASE')
    if v:
        _t4cfg.cnt_base = int(v)
    v = os.environ.get('TLA_CB_CNT_SLOPE')
    if v:
        _t4cfg.cnt_slope = int(v)
    v = os.environ.get('TLA_CB_CNT_MAXINV')
    if v:
        _t4cfg.cnt_maxinv = int(v)
    v = os.environ.get('TLA_CB_INL')
    if v:
        _t4cfg.c_inl = int(v)
    v = os.environ.get('TLA_CB_RES')
    if v:
        _t4cfg.c_res = int(v)
    v = os.environ.get('TLA_CB_BR_CMP')
    if v:
        _t4cfg.c_br_cmp = int(v)
    v = os.environ.get('TLA_CB_BR_ARI')
    if v:
        _t4cfg.c_br_ari = int(v)
    v = os.environ.get('TLA_CB_RECOMP_BASE')
    if v:
        _t4cfg.recomp_base = int(v)
    v = os.environ.get('TLA_CB_RECOMP_SLOPE')
    if v:
        _t4cfg.recomp_slope = int(v)
    v = os.environ.get('TLA_CB_HORIZON')
    if v:
        _t4cfg.horizon = int(v)
    v = os.environ.get('TLA_CB_REOPT_BASE')
    if v:
        _t4cfg.reopt_base = int(v)
    v = os.environ.get('TLA_CB_REOPT_MULT')
    if v:
        _t4cfg.reopt_mult = int(v)
    v = os.environ.get('TLA_CB_REOPT_CAP')
    if v:
        _t4cfg.reopt_cap = int(v)

@jit.elidable
def _t2_add(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _intbox(w_x.intvalue + w_y.intvalue)
    return w_x.add_inline(w_y)

@jit.elidable
def _t2_sub(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _intbox(w_x.intvalue - w_y.intvalue)
    return w_x.sub_inline(w_y)

@jit.elidable
def _t2_mul(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _intbox(w_x.intvalue * w_y.intvalue)
    return w_x.mul_inline(w_y)

@jit.dont_look_inside
def _t2_div(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _intbox(w_x.intvalue // w_y.intvalue)
    return w_x.div_inline(w_y)

@jit.dont_look_inside
def _t2_mod(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _intbox(w_x.intvalue % w_y.intvalue)
    return w_x.mod_inline(w_y)

@jit.elidable
def _t2_le(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _BOX_TRUE if w_x.intvalue <= w_y.intvalue else _BOX_FALSE
    if isinstance(w_x, W_FloatObject) and isinstance(w_y, W_FloatObject):
        return _BOX_TRUE if w_x.floatvalue <= w_y.floatvalue else _BOX_FALSE
    return _BOX_FALSE

@jit.elidable
def _t2_ge(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _BOX_TRUE if w_x.intvalue >= w_y.intvalue else _BOX_FALSE
    if isinstance(w_x, W_FloatObject) and isinstance(w_y, W_FloatObject):
        return _BOX_TRUE if w_x.floatvalue >= w_y.floatvalue else _BOX_FALSE
    return _BOX_FALSE

@jit.elidable
def _t2_eq(w_x, w_y):
    if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
        return _BOX_TRUE if w_x.intvalue == w_y.intvalue else _BOX_FALSE
    if isinstance(w_x, W_FloatObject) and isinstance(w_y, W_FloatObject):
        return _BOX_TRUE if w_x.floatvalue == w_y.floatvalue else _BOX_FALSE
    return _BOX_FALSE


class JitFrameBase(object):
    """Shared base for JitFrame and JitFrame3."""


class JitFrame(JitFrameBase):
    """Virtualizable frame for tiers 0 (jitted=False) and 2 (jitted=True)."""
    _virtualizable_ = ['stackpos', 'stack[*]']
    _immutable_fields_ = ['hybrid', 'inline_arith', 'jitted']

    def __init__(self, bytecode, stack=None, stackpos=0, depth=0, stacksize=64,
                 hybrid=False, inline_arith=False, jitted=True):
        self = jit.hint(self, access_directly=True, fresh_virtualizable=True)
        if stack is None:
            stack = [None] * stacksize
        self.stacksize = stacksize
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos
        self.depth = depth
        self.hybrid = hybrid
        self.inline_arith = inline_arith
        self.jitted = jitted

    def _push(self, w_x):
        stackpos = self.stackpos
        assert stackpos >= 0
        self.stack[stackpos] = w_x
        self.stackpos = stackpos + 1

    def _pop(self):
        stackpos = self.stackpos - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    def _take(self, n):
        idx = self.stackpos - n - 1
        assert idx >= 0
        w_x = self.stack[idx]
        assert w_x is not None
        return w_x

    @jit.unroll_safe
    def _drop(self, n):
        for _ in range(n):
            self._pop()

    @jit.unroll_safe
    def copy_frame(self, argnum, retaddr):
        framepos = self.stackpos - argnum - 1
        assert framepos >= 0
        assert argnum >= 0
        bytecode = jit.promote(self.bytecode)
        newframe = JitFrame(bytecode, None, argnum + 2, self.depth + 1,
                            stacksize=self.stacksize, jitted=self.jitted)
        count = argnum + 1
        j = 0
        while j < count:
            newframe.stack[j] = self.stack[framepos + j]
            j += 1
        newframe.stack[argnum + 1] = W_IntObject(retaddr)
        return newframe

    @jit.not_in_trace
    def dump(self):
        sys.stderr.write("stackpos: %d " % self.stackpos)
        sys.stderr.write("[")
        for i in range(self.stackpos):
            w_x = self.stack[i]
            if isinstance(w_x, W_Object):
                sys.stderr.write(w_x.getrepr() + ", ")
        sys.stderr.write("]\n")

    def _is_true(self):
        w_x = self._pop()
        return w_x.is_true()

    def _CONST_INT(self, pc, neg=False):
        bytecode = jit.promote(self.bytecode)
        x = ord(bytecode[pc])
        if neg:
            self._push(_intbox(-x))
        else:
            self._push(_intbox(x))

    def _CONST_FLOAT(self, pc, neg=False):
        bytecode = jit.promote(self.bytecode)
        x = _construct_float(bytecode, pc)
        if neg:
            self._push(W_FloatObject(-x))
        else:
            self._push(W_FloatObject(x))

    def _CONST_N(self, pc):
        bytecode = jit.promote(self.bytecode)
        x = _construct_value(bytecode, pc)
        self._push(_intbox(x))

    def _POP(self):
        return self._pop()

    def _POP1(self):
        v = self._pop()
        _ = self._pop()
        self._push(v)

    def _DUP(self):
        w_x = self._pop()
        self._push(w_x)
        self._push(w_x)

    def _DUPN(self, pc):
        bytecode = jit.promote(self.bytecode)
        n = ord(bytecode[pc])
        w_x = self._take(n)
        self._push(w_x)

    def _ADD(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_add(w_x, w_y))

    def _SUB(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_sub(w_x, w_y))

    def _MUL(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_mul(w_x, w_y))

    def _DIV(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_div(w_x, w_y))

    def _MOD(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_mod(w_x, w_y))

    def _LT(self):
        # LT dispatches <= (matches Frame; every lang program relies on it).
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_le(w_x, w_y))

    def _GT(self):
        # GT dispatches >= (matches Frame).
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_ge(w_x, w_y))

    def _EQ(self):
        w_y = self._pop()
        w_x = self._pop()
        self._push(_t2_eq(w_x, w_y))

    def _BUILD_LIST(self):
        size = self._pop()
        init = self._pop()
        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self._push(W_ListObject(lst))

    def _LOAD(self):
        w_index = self._pop()
        w_lst = self._pop()
        assert isinstance(w_index, W_IntObject)
        assert isinstance(w_lst, W_ListObject)
        w_x = w_lst.listvalue[int(w_index.intvalue)]
        self._push(w_x)

    def _STORE(self):
        w_index = self._pop()
        w_lst = self._pop()
        w_x = self._pop()
        assert isinstance(w_lst, W_ListObject)
        assert isinstance(w_index, W_IntObject)
        w_lst.listvalue[int(w_index.intvalue)] = w_x
        self._push(w_lst)

    def _RAND_INT(self):
        raise NotImplementedError

    def _COS(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.cos(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.cos(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SIN(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.sin(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.sin(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SQRT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(math.sqrt(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_x = W_FloatObject(math.sqrt(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_x)

    def _INT_TO_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_IntObject)
        w_x = W_FloatObject(float(w_x.intvalue))
        self._push(w_x)

    def _FLOAT_TO_INT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        w_x = W_IntObject(int(w_x.floatvalue))
        self._push(w_x)

    def _ABS_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        self._push(W_FloatObject(abs(w_x.floatvalue)))

    def _PRINT(self):
        v = self._take(0)
        # print v.getrepr()

    def _RET(self, n):
        v = self._pop()
        return v

    @jit.unroll_safe
    def _FRAME_RESET(self, o, l, n):
        stackpos = jit.promote(self.stackpos)
        ret_idx = stackpos - n - 1
        old_base = stackpos - n
        new_base = stackpos - o - n - l - 1
        assert n >= 0
        assert ret_idx >= 0
        assert old_base >= 0
        assert new_base >= 0
        ret = self.stack[ret_idx]

        i = 0
        while i < n:
            self.stack[new_base + i] = self.stack[old_base + i]
            self.stack[old_base + i] = None
            i += 1

        self.stack[new_base + n] = ret
        self.stackpos = new_base + n + 1

    def _CALL(self, oldframe, t, argnum):
        w_x = self._interp(t)
        oldframe._drop(argnum)
        if w_x:
            oldframe._push(w_x)

    def _interp(self, pc=0):
        if self.depth > MAX_INTERP_DEPTH:
            raise OperationError
        bytecode = self.bytecode

        while pc < len(bytecode):
            if self.jitted:
                tier2vdriver.jit_merge_point(bytecode=bytecode, pc=pc, self=self)

            # print get_printable_location(pc, bytecode)

            opcode = ord(bytecode[pc])
            pc += 1

            if opcode == CONST_INT:
                self._CONST_INT(pc)
                pc += 1

            elif opcode == CONST_NEG_INT:
                self._CONST_INT(pc, neg=True)
                pc += 1

            elif opcode == CONST_FLOAT:
                self._CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self._CONST_FLOAT(pc, neg=True)
                pc += 9

            elif opcode == CONST_N:
                self._CONST_N(pc)
                pc += 4

            elif opcode == POP:
                self._POP()

            elif opcode == POP1:
                self._POP1()

            elif opcode == DUP:
                self._DUP()

            elif opcode == DUPN:
                self._DUPN(pc)
                pc += 1

            elif opcode == LT:
                self._LT()

            elif opcode == GT:
                self._GT()

            elif opcode == EQ:
                self._EQ()

            elif opcode == ADD:
                self._ADD()

            elif opcode == SUB:
                self._SUB()

            elif opcode == DIV:
                self._DIV()

            elif opcode == MUL:
                self._MUL()

            elif opcode == MOD:
                self._MOD()

            elif opcode == BUILD_LIST:
                self._BUILD_LIST()

            elif opcode == LOAD:
                self._LOAD()

            elif opcode == STORE:
                self._STORE()

            elif opcode == RAND_INT:
                self._RAND_INT()

            elif opcode == SIN:
                self._SIN()

            elif opcode == COS:
                self._COS()

            elif opcode == ABS_FLOAT:
                self._ABS_FLOAT()

            elif opcode == SQRT:
                self._SQRT()

            elif opcode == INT_TO_FLOAT:
                self._INT_TO_FLOAT()

            elif opcode == FLOAT_TO_INT:
                self._FLOAT_TO_INT()

            elif opcode == CALL_ASSEMBLER:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == CALL_N:
                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == RET:
                argnum = hint(ord(bytecode[pc]), promote=True)
                pc += 1
                w_x = self._RET(argnum)
                return w_x

            elif opcode == JUMP:
                t = ord(bytecode[pc])
                if self.jitted and t < pc:
                    tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                pc = t

            elif opcode == JUMP_N:
                t = _construct_value(bytecode, pc)
                if self.jitted and t < pc:
                    tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                pc = t

            elif opcode == JUMP_IF:
                t = ord(bytecode[pc])
                pc += 1

                if self._is_true():
                    if self.jitted and t < pc:
                        tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                    pc = t

            elif opcode == JUMP_IF_N:
                t = _construct_value(bytecode, pc)
                pc += 4

                if self._is_true():
                    if self.jitted and t < pc:
                        tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                    pc = t

            elif opcode == EXIT:
                return self._POP()

            elif opcode == PRINT:
                self._PRINT()

            elif opcode == FRAME_RESET:
                old_arity = ord(bytecode[pc])
                local_size = ord(bytecode[pc+1])
                new_arity = ord(bytecode[pc+2])
                pc += 3
                self._FRAME_RESET(old_arity, local_size, new_arity)

            elif opcode == NOP:
                continue

            else:
                assert False, 'Unknown opcode: %s' % bytecodes[opcode]


class JitFrame3(JitFrameBase):
    """Virtualizable frame for tiers 3 and 4 (inline arithmetic)."""
    _virtualizable_ = ['stackpos', 'stack[*]']
    _immutable_fields_ = ['hybrid', 'inline_arith']

    def __init__(self, bytecode, stack=None, stackpos=0, depth=0, stacksize=64,
                 hybrid=False, inline_arith=False):
        self = jit.hint(self, access_directly=True, fresh_virtualizable=True)
        if stack is None:
            stack = [None] * stacksize
        self.stacksize = stacksize
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos
        self.depth = depth
        self.hybrid = hybrid
        self.inline_arith = inline_arith

    def _push(self, w_x):
        stackpos = self.stackpos
        assert stackpos >= 0
        self.stack[stackpos] = w_x
        self.stackpos = stackpos + 1

    def _pop(self):
        stackpos = self.stackpos - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    def _take(self, n):
        idx = self.stackpos - n - 1
        assert idx >= 0
        w_x = self.stack[idx]
        assert w_x is not None
        return w_x

    @jit.unroll_safe
    def _drop(self, n):
        for _ in range(n):
            self._pop()

    @jit.unroll_safe
    def copy_frame(self, argnum, retaddr):
        framepos = self.stackpos - argnum - 1
        assert framepos >= 0
        assert argnum >= 0
        bytecode = jit.promote(self.bytecode)
        newframe = JitFrame3(bytecode, None, argnum + 2, self.depth + 1,
                            stacksize=self.stacksize, hybrid=self.hybrid,
                            inline_arith=self.inline_arith)
        count = argnum + 1
        j = 0
        while j < count:
            newframe.stack[j] = self.stack[framepos + j]
            j += 1
        newframe.stack[argnum + 1] = W_IntObject(retaddr)
        return newframe

    @jit.not_in_trace
    def dump(self):
        sys.stderr.write("stackpos: %d " % self.stackpos)
        sys.stderr.write("[")
        for i in range(self.stackpos):
            w_x = self.stack[i]
            if isinstance(w_x, W_Object):
                sys.stderr.write(w_x.getrepr() + ", ")
        sys.stderr.write("]\n")

    def _is_true(self):
        w_x = self._pop()
        return w_x.is_true()

    def _CONST_INT(self, pc, neg=False):
        bytecode = jit.promote(self.bytecode)
        x = ord(bytecode[pc])
        if neg:
            self._push(W_IntObject(-x))
        else:
            self._push(W_IntObject(x))

    def _CONST_FLOAT(self, pc, neg=False):
        bytecode = jit.promote(self.bytecode)
        x = _construct_float(bytecode, pc)
        if neg:
            self._push(W_FloatObject(-x))
        else:
            self._push(W_FloatObject(x))

    def _CONST_N(self, pc):
        bytecode = jit.promote(self.bytecode)
        x = _construct_value(bytecode, pc)
        self._push(W_IntObject(x))

    def _POP(self):
        return self._pop()

    def _POP1(self):
        v = self._pop()
        _ = self._pop()
        self._push(v)

    def _DUP(self):
        w_x = self._pop()
        self._push(w_x)
        self._push(w_x)

    def _DUPN(self, pc):
        bytecode = jit.promote(self.bytecode)
        n = ord(bytecode[pc])
        w_x = self._take(n)
        self._push(w_x)

    def _profile(self, bytecode, site, w_x, w_y, is_cmp):
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            a = bytecode.cnt_a[site] + 1
            bytecode.cnt_a[site] = a
            b = bytecode.cnt_b[site]
        else:
            a = bytecode.cnt_a[site]
            b = bytecode.cnt_b[site] + 1
            bytecode.cnt_b[site] = b
        if is_cmp:
            if a != 0 and b != 0:
                _t4_set_poly(bytecode, site, 1)
        else:
            total = a + b
            if _t4cfg.minn <= total <= _t4cfg.freeze:
                minority = a if a < b else b
                if minority * _t4cfg.ratio >= total:
                    _t4_set_poly(bytecode, site, 1)
                else:
                    _t4_set_poly(bytecode, site, 0)
            # DRR: deopt-rate re-decision.  Off-type guard failures replay this
            # op under the blackhole; the old freeze window cuts off that signal,
            # but here we use it -- once such bails reach the floor AND dominate
            # the inlined runs (a rate, not a raw count), residualise the site so
            # a genuinely polymorphic hot site stops paying a bridge per off-type
            # hit.  Monotonic 0->1, latched by redecided so it fires at most once
            # per site (no churn / re-invalidation).  Inert at ratio==1 (the
            # shipped default), where inline_arith short-circuits this whole path.
            if we_are_blackholing():
                nb = bytecode.bails[site] + 1
                bytecode.bails[site] = nb
                if (bytecode.redecided[site] == 0 and
                        nb >= _t4cfg.bailfloor and
                        nb * _t4cfg.ratio >= bytecode.inl_runs[site] + nb):
                    _t4_set_poly(bytecode, site, 1)
                    bytecode.redecided[site] = 1

    def _ADD(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid and not self.inline_arith:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, False)
                if bytecode.poly[site] == 0:
                    bytecode.inl_runs[site] += 1
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_add(w_x, w_y)); return
        self._push(w_x.add_inline(w_y))

    def _SUB(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid and not self.inline_arith:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, False)
                if bytecode.poly[site] == 0:
                    bytecode.inl_runs[site] += 1
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_sub(w_x, w_y)); return
        self._push(w_x.sub_inline(w_y))

    def _MUL(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid and not self.inline_arith:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, False)
                if bytecode.poly[site] == 0:
                    bytecode.inl_runs[site] += 1
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_mul(w_x, w_y)); return
        self._push(w_x.mul_inline(w_y))

    def _DIV(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid and not self.inline_arith:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, False)
                if bytecode.poly[site] == 0:
                    bytecode.inl_runs[site] += 1
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_div(w_x, w_y)); return
        self._push(w_x.div_inline(w_y))

    def _MOD(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid and not self.inline_arith:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, False)
                if bytecode.poly[site] == 0:
                    bytecode.inl_runs[site] += 1
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_mod(w_x, w_y)); return
        self._push(w_x.mod_inline(w_y))

    def _LT(self, site):
        # LT dispatches <= (matches Frame; every lang program relies on it).
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, True)
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_le(w_x, w_y)); return
        self._push(w_x.le_inline(w_y))

    def _GT(self, site):
        # GT dispatches >= (matches Frame).
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, True)
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_ge(w_x, w_y)); return
        self._push(w_x.ge_inline(w_y))

    def _EQ(self, site):
        w_y = self._pop()
        w_x = self._pop()
        if self.hybrid:
            bytecode = jit.promote(self.bytecode)
            if not we_are_jitted():
                self._profile(bytecode, site, w_x, w_y, True)
            if jit.promote(bytecode.poly[site]):
                self._push(_t2_eq(w_x, w_y)); return
        self._push(w_x.eq_inline(w_y))

    def _BUILD_LIST(self):
        size = self._pop()
        init = self._pop()
        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self._push(W_ListObject(lst))

    def _LOAD(self):
        w_index = self._pop()
        w_lst = self._pop()
        assert isinstance(w_index, W_IntObject)
        assert isinstance(w_lst, W_ListObject)
        w_x = w_lst.listvalue[int(w_index.intvalue)]
        self._push(w_x)

    def _STORE(self):
        w_index = self._pop()
        w_lst = self._pop()
        w_x = self._pop()
        assert isinstance(w_lst, W_ListObject)
        assert isinstance(w_index, W_IntObject)
        w_lst.listvalue[int(w_index.intvalue)] = w_x
        self._push(w_lst)

    def _RAND_INT(self):
        raise NotImplementedError

    def _COS(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.cos(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.cos(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SIN(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.sin(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.sin(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    def _SQRT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(math.sqrt(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_x = W_FloatObject(math.sqrt(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_x)

    def _INT_TO_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_IntObject)
        w_x = W_FloatObject(float(w_x.intvalue))
        self._push(w_x)

    def _FLOAT_TO_INT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        w_x = W_IntObject(int(w_x.floatvalue))
        self._push(w_x)

    def _ABS_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        self._push(W_FloatObject(abs(w_x.floatvalue)))

    def _PRINT(self):
        v = self._take(0)
        # print v.getrepr()

    def _RET(self, n):
        v = self._pop()
        return v

    @jit.unroll_safe
    def _FRAME_RESET(self, o, l, n):
        stackpos = jit.promote(self.stackpos)
        ret_idx = stackpos - n - 1
        old_base = stackpos - n
        new_base = stackpos - o - n - l - 1
        assert n >= 0
        assert ret_idx >= 0
        assert old_base >= 0
        assert new_base >= 0
        ret = self.stack[ret_idx]

        i = 0
        while i < n:
            self.stack[new_base + i] = self.stack[old_base + i]
            self.stack[old_base + i] = None
            i += 1

        self.stack[new_base + n] = ret
        self.stackpos = new_base + n + 1

    def _CALL(self, oldframe, t, argnum):
        w_x = self._interp(t)
        oldframe._drop(argnum)
        if w_x:
            oldframe._push(w_x)

    def _interp(self, pc=0):
        "tier-2 inlined tracing interpreter (virtualizable frame)"
        if self.depth > MAX_INTERP_DEPTH:
            raise OperationError
        bytecode = self.bytecode
        hybrid = self.hybrid

        while pc < len(bytecode):
            tier3driver.jit_merge_point(bytecode=bytecode, pc=pc,
                                        hybrid=hybrid, self=self)

            # print get_printable_location(pc, bytecode)

            opcode = ord(bytecode[pc])
            pc += 1

            if opcode == CONST_INT:
                self._CONST_INT(pc)
                pc += 1

            elif opcode == CONST_NEG_INT:
                self._CONST_INT(pc, neg=True)
                pc += 1

            elif opcode == CONST_FLOAT:
                self._CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self._CONST_FLOAT(pc, neg=True)
                pc += 9

            elif opcode == CONST_N:
                self._CONST_N(pc)
                pc += 4

            elif opcode == POP:
                self._POP()

            elif opcode == POP1:
                self._POP1()

            elif opcode == DUP:
                self._DUP()

            elif opcode == DUPN:
                self._DUPN(pc)
                pc += 1

            elif opcode == LT:
                self._LT(pc - 1)

            elif opcode == GT:
                self._GT(pc - 1)

            elif opcode == EQ:
                self._EQ(pc - 1)

            elif opcode == ADD:
                self._ADD(pc - 1)

            elif opcode == SUB:
                self._SUB(pc - 1)

            elif opcode == DIV:
                self._DIV(pc - 1)

            elif opcode == MUL:
                self._MUL(pc - 1)

            elif opcode == MOD:
                self._MOD(pc - 1)

            elif opcode == BUILD_LIST:
                self._BUILD_LIST()

            elif opcode == LOAD:
                self._LOAD()

            elif opcode == STORE:
                self._STORE()

            elif opcode == RAND_INT:
                self._RAND_INT()

            elif opcode == SIN:
                self._SIN()

            elif opcode == COS:
                self._COS()

            elif opcode == ABS_FLOAT:
                self._ABS_FLOAT()

            elif opcode == SQRT:
                self._SQRT()

            elif opcode == INT_TO_FLOAT:
                self._INT_TO_FLOAT()

            elif opcode == FLOAT_TO_INT:
                self._FLOAT_TO_INT()

            elif opcode == CALL_ASSEMBLER:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == CALL_N:
                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == RET:
                argnum = hint(ord(bytecode[pc]), promote=True)
                pc += 1
                w_x = self._RET(argnum)
                return w_x

            elif opcode == JUMP:
                t = ord(bytecode[pc])
                if t < pc:
                    tier3driver.can_enter_jit(bytecode=bytecode, pc=t,
                                              hybrid=hybrid, self=self)
                pc = t

            elif opcode == JUMP_N:
                t = _construct_value(bytecode, pc)
                if t < pc:
                    tier3driver.can_enter_jit(bytecode=bytecode, pc=t,
                                              hybrid=hybrid, self=self)
                pc = t

            elif opcode == JUMP_IF:
                t = ord(bytecode[pc])
                pc += 1

                if self._is_true():
                    if t < pc:
                        tier3driver.can_enter_jit(bytecode=bytecode, pc=t,
                                                  hybrid=hybrid, self=self)
                    pc = t

            elif opcode == JUMP_IF_N:
                t = _construct_value(bytecode, pc)
                pc += 4

                if self._is_true():
                    if t < pc:
                        tier3driver.can_enter_jit(bytecode=bytecode, pc=t,
                                                  hybrid=hybrid, self=self)
                    pc = t

            elif opcode == EXIT:
                return self._POP()

            elif opcode == PRINT:
                self._PRINT()

            elif opcode == FRAME_RESET:
                old_arity = ord(bytecode[pc])
                local_size = ord(bytecode[pc+1])
                new_arity = ord(bytecode[pc+2])
                pc += 3
                self._FRAME_RESET(old_arity, local_size, new_arity)

            elif opcode == NOP:
                continue

            else:
                assert False, 'Unknown opcode: %s' % bytecodes[opcode]
