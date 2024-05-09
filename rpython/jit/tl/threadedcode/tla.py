import math
import sys

from rpython.rlib import jit
from rpython.rlib.jit import JitDriver, we_are_jitted, hint
from rpython.rlib.rarithmetic import r_uint
from rpython.rlib.rrandom import Random

from rpython.jit.tl.threadedcode.hints import (
    enable_shallow_tracing,
    enable_shallow_tracing_argn,
    enable_shallow_tracing_with_value
)
from rpython.jit.tl.threadedcode.traverse_stack import TStack, t_empty, t_push
from rpython.jit.tl.threadedcode.tlib import emit_jump, emit_ret
from rpython.jit.tl.threadedcode.object import (
    W_Object,
    W_IntObject,
    W_FloatObject,
    W_StringObject,
    W_ListObject,
    OperationError
)
from rpython.jit.tl.threadedcode.bytecode import *


TRACE_THRESHOLD = -1

class ContinueInTracingJIT(Exception):
    def __init__(self, pc):
        self.pc = pc

class ContinueInThreadedJIT(Exception):
    def __init__(self, pc):
        self.pc = pc

def get_printable_location_tier1(pc, entry, bytecode, tstack):
    op = ord(bytecode[pc])
    name = bytecodes[op]

    if hasarg[op]:
        arg = str(ord(bytecode[pc + 1]))
    else:
        arg = ''

    if tstack.t_is_empty():
        return "%s: %s %s, tstack: None" % (pc, name, arg)
    else:
        return "%s: %s %s, tstack: %d" % (pc, name, arg, tstack.pc)

def get_printable_location(pc, bytecode):
    op = ord(bytecode[pc])
    name = bytecodes[op]
    if hasarg[op]:
        arg = str(ord(bytecode[pc + 1]))
    else:
        arg = ''
    return "%s: %s %s" % (pc, name, arg)

def _construct_value(bytecode, pc):
    a = ord(bytecode[pc])
    b = ord(bytecode[pc+1])
    c = ord(bytecode[pc+2])
    d = ord(bytecode[pc+3])
    return a << 24 | b << 16 | c << 8 | d

@jit.unroll_safe
def _power_01(n):
    acc = 1
    for i in range(n):
        acc = acc * 0.1
    return acc

@jit.unroll_safe
def _construct_float(bytecode, pc):
    literals = [0] * 9
    for i in range(9):
        assert pc + i < len(bytecode)
        literals[i] = ord(bytecode[pc+i])

    int_val = _construct_value(bytecode, pc)
    float_val = _construct_value(bytecode, pc+4)

    decimal = literals[8]
    return float(int_val + (float_val * _power_01(decimal)))

tier1driver = JitDriver(
    greens=['pc', 'entry', 'bytecode', 'tstack'], reds=['self'],
    get_printable_location=get_printable_location_tier1,
    threaded_code_gen=True, conditions=["is_true"])


tier2driver = JitDriver(
    greens=['pc', 'bytecode',], reds=['self'],
    get_printable_location=get_printable_location, is_recursive=True)


class Frame(object):
    def __init__(self, bytecode, stack=[None] * 64, stackpos=0):
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos

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
        return Frame(bytecode, newstack, argnum + 2)

    @enable_shallow_tracing
    def push(self, w_x):
        self.stack[self.stackpos] = w_x
        self.stackpos += 1

    def _push(self, w_x):
        stackpos = jit.promote(self.stackpos)
        self.stack[stackpos] = w_x
        self.stackpos += 1

    @enable_shallow_tracing_with_value(W_Object())
    def pop(self):
        stackpos = self.stackpos - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    def _pop(self):
        stackpos = jit.promote(self.stackpos) - 1
        assert stackpos >= 0
        self.stackpos = stackpos
        res = self.stack[stackpos]
        self.stack[stackpos] = None
        return res

    @enable_shallow_tracing_with_value(W_Object())
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

    @enable_shallow_tracing
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

    @enable_shallow_tracing_with_value(True)
    def is_true(self):
        w_x = self.pop()
        return w_x.is_true()

    def _is_true(self):
        w_x = self._pop()
        return w_x.is_true()

    @enable_shallow_tracing
    def CONST_INT(self, pc, neg=False):
        if isinstance(pc, int):
            x = ord(self.bytecode[pc])
            if neg:
                self.push(W_IntObject(-x))
            else:
                self.push(W_IntObject(x))
        else:
            raise OperationError

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

    @enable_shallow_tracing
    def CONST_FLOAT(self, pc, neg=False):
        if isinstance(pc, int):
            x = _construct_float(self.bytecode, pc)
            if neg:
                self.push(W_FloatObject(-x))
            else:
                self.push(W_FloatObject(x))
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

    @enable_shallow_tracing
    def CONST_N(self, pc):
        if isinstance(pc, int):
            bytecode = jit.promote(self.bytecode)
            x = _construct_value(bytecode, pc)
            self.push(W_IntObject(x))
        else:
            raise OperationError

    def _CONST_N(self, pc):
        if isinstance(pc, int):
            bytecode = jit.promote(self.bytecode)
            x = _construct_value(bytecode, pc)
            self._push(W_IntObject(x))
        else:
            raise OperationError

    @enable_shallow_tracing
    def PUSH(self, w_x):
        self.push(w_x)

    def _PUSH(self, w_x):
        self.push(w_x)

    @jit.dont_look_inside
    def POP(self, dummy=False):
        if dummy:
            return self.take(0)
        return self.pop()

    def _POP(self):
        return self._pop()

    @enable_shallow_tracing
    def DROP(self, n):
        for _ in range(n):
            self.pop()

    @jit.unroll_safe
    def _DROP(self, n):
        for _ in range(n):
            self._pop()

    @enable_shallow_tracing
    def POP1(self):
        v = self.pop()
        _ = self.pop()
        self.push(v)

    def _POP1(self):
        v = self._pop()
        _ = self._pop()
        self._push(v)

    @enable_shallow_tracing
    def ADD(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.add(w_y)
        self.push(w_z)

    def _ADD(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.add(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def SUB(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.sub(w_y)
        self.push(w_z)

    def _SUB(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.sub(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def MUL(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.mul(w_y)
        self.push(w_z)

    def _MUL(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.mul(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def DIV(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.div(w_y)
        self.push(w_z)

    def _DIV(self):
        w_y = self._pop()
        w_x = self.pop()
        w_z = w_x.div(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def MOD(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.mod(w_y)
        self.push(w_z)

    def _MOD(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.mod(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def DUP(self):
        w_x = self.pop()
        self.push(w_x)
        self.push(w_x)

    def _DUP(self):
        w_x = self._pop()
        self._push(w_x)
        self._push(w_x)

    @enable_shallow_tracing
    def DUPN(self, pc):
        n = ord(self.bytecode[pc])
        w_x = self.take(n)
        self.push(w_x)

    def _DUPN(self, pc):
        bytecode = jit.promote(self.bytecode)
        n = ord(bytecode[pc])
        w_x = self._take(n)
        self._push(w_x)

    @enable_shallow_tracing
    def LT(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.le(w_y)
        self.push(w_z)

    def _LT(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.le(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def GT(self):
        w_y = self.pop()
        w_x = self.pop()
        w_z = w_x.ge(w_y)
        self.push(w_z)

    def _GT(self):
        w_y = self._pop()
        w_x = self._pop()
        w_z = w_x.ge(w_y)
        self._push(w_z)

    @enable_shallow_tracing
    def EQ(self):
        w_y = self.pop()
        w_x = self.pop()
        self.push(w_x.eq(w_y))

    def _EQ(self):
        w_y = self._pop()
        w_x = self._pop()
        self.push(w_x.eq(w_y))

    @enable_shallow_tracing
    def NE(self):
        w_y = self.pop()
        w_x = self.pop()
        if w_x.eq(w_y).intvalue:
            self.push(W_IntObject(1))
        else:
            self.push(W_IntObject(0))

    def _NE(self):
        w_y = self._pop()
        w_x = self._pop()
        if w_x.eq(w_y).intvalue:
            self._push(W_IntObject(1))
        else:
            self._push(W_IntObject(0))

    @jit.dont_look_inside
    def CALL(self, oldframe, t, argnum, dummy=True):
        if dummy:
            return
        w_x = self.interp(t)
        oldframe.drop(argnum)
        if w_x:
            oldframe.push(w_x)

    @jit.call_assembler
    def CALL_ASSEMBLER(self, oldframe, t, argnum, bytecode,
                       tstack, dummy):
        "Special handler to be compiled to call_assembler_r"
        w_x = self.interp_CALL_ASSEMBLER(t, t, bytecode,
                                         tstack, dummy)
        oldframe.DROP(argnum)
        if w_x:
            oldframe.PUSH(w_x)

    def _CALL(self, oldframe, t, argnum):
        w_x = self._interp(t)
        oldframe._drop(argnum)
        if w_x:
            oldframe._push(w_x)

    @enable_shallow_tracing_with_value(W_Object())
    def RET(self, n):
        v = self.pop()
        return v

    def _RET(self, n):
        v = self._pop()
        return v

    @enable_shallow_tracing
    def PRINT(self):
        v = self.take(0)
        print v.getrepr()

    def _PRINT(self):
        v = self._take(0)
        # print v.getrepr()

    @enable_shallow_tracing
    def FRAME_RESET(self, o, l, n):
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

    @enable_shallow_tracing
    def BUILD_LIST(self):
        size = self.pop()
        init = self.pop()

        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self.push(W_ListObject(lst))

    def _BUILD_LIST(self):
        size = self._pop()
        init = self._pop()

        assert isinstance(size, W_IntObject)
        lst = [init] * int(size.intvalue)
        self.push(W_ListObject(lst))

    @enable_shallow_tracing
    def LOAD(self):
        w_index = self.pop()
        w_lst = self.pop()

        assert isinstance(w_index, W_IntObject)
        assert isinstance(w_lst, W_ListObject)

        assert w_index.intvalue < len(w_lst.listvalue)
        w_x = w_lst.listvalue[int(w_index.intvalue)]
        self.push(w_x)

    def _LOAD(self):
        w_index = self._pop()
        w_lst = self._pop()

        assert isinstance(w_index, W_IntObject)
        assert isinstance(w_lst, W_ListObject)

        w_x = w_lst.listvalue[int(w_index.intvalue)]
        self._push(w_x)

    @enable_shallow_tracing
    def STORE(self):
        w_index = self.pop()
        w_lst = self.pop()
        w_x = self.pop()

        assert isinstance(w_lst, W_ListObject)
        assert isinstance(w_index, W_IntObject)

        w_lst.listvalue[int(w_index.intvalue)] = w_x
        self.push(w_lst)

    def _STORE(self):
        w_index = self._pop()
        w_lst = self._pop()
        w_x = self._pop()

        assert isinstance(w_lst, W_ListObject)
        assert isinstance(w_index, W_IntObject)

        w_lst.listvalue[int(w_index.intvalue)] = w_x
        self._push(w_lst)

    @enable_shallow_tracing
    def RAND_INT(self):
        raise NotImplementedError

    def _RAND_INT(self):
        raise NotImplementedError

    @enable_shallow_tracing
    def COS(self):
        w_x = self.pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.cos(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.cos(w_x.floatvalue))
        else:
            raise OperationError
        self.push(w_c)

    def _COS(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.cos(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.cos(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    @enable_shallow_tracing
    def SIN(self):
        w_x = self.pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.sin(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.sin(w_x.floatvalue))
        else:
            raise OperationError
        self.push(w_c)

    def _SIN(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_c = W_FloatObject(math.sin(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_c = W_FloatObject(math.sin(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_c)

    @enable_shallow_tracing
    def SQRT(self):
        w_x = self.pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(math.sqrt(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_x = W_FloatObject(math.sqrt(w_x.floatvalue))
        else:
            raise OperationError
        self.push(w_x)

    def _SQRT(self):
        w_x = self._pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(math.sqrt(w_x.intvalue))
        elif isinstance(w_x, W_FloatObject):
            w_x = W_FloatObject(math.sqrt(w_x.floatvalue))
        else:
            raise OperationError
        self._push(w_x)

    @enable_shallow_tracing
    def INT_TO_FLOAT(self):
        w_x = self.pop()
        if isinstance(w_x, W_IntObject):
            w_x = W_FloatObject(float(w_x.intvalue))
        self.push(w_x)

    def _INT_TO_FLOAT(self):
        w_x = self.pop()
        assert isinstance(w_x, W_IntObject)
        w_x = W_FloatObject(float(w_x.intvalue))
        self.push(w_x)

    @enable_shallow_tracing
    def FLOAT_TO_INT(self):
        w_x = self.pop()
        assert isinstance(w_x, W_FloatObject)
        w_x = W_IntObject(int(w_x.floatvalue))
        self.push(w_x)

    def _FLOAT_TO_INT(self):
        w_x = self.pop()
        assert isinstance(w_x, W_FloatObject)
        w_x = W_IntObject(int(w_x.floatvalue))
        self.push(w_x)

    @enable_shallow_tracing
    def ABS_FLOAT(self):
        w_x = self.pop()
        assert isinstance(w_x, W_FloatObject)
        self.push(W_FloatObject(abs(w_x.floatvalue)))

    def _ABS_FLOAT(self):
        w_x = self._pop()
        assert isinstance(w_x, W_FloatObject)
        self._push(W_FloatObject(abs(w_x.floatvalue)))

    def _interp(self, pc=0):
        "tracing interpreter"
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
                self._CONST_INT(pc, True)
                pc += 1

            elif opcode == CONST_FLOAT:
                self._CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self._CONST_FLOAT(pc, True)
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

                # create a new frame
                frame = self.copy_frame(argnum, pc)
                frame._CALL(self, t, argnum)

            elif opcode == CALL_N:
                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                # create a new frame
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

    @jit.call_assembler
    def interp_CALL_ASSEMBLER(self, pc, entry, bytecode, tstack, dummy):
        if dummy:
            return self.take(0)

        return self.interp(pc, dummy)


    def interp(self, pc=0, dummy=False):
        if dummy:
            return

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
                self.CONST_INT(pc, True)
                pc += 1

            elif opcode == CONST_FLOAT:
                self.CONST_FLOAT(pc)
                pc += 9

            elif opcode == CONST_NEG_FLOAT:
                self.CONST_FLOAT(pc, True)
                pc += 9

            elif opcode == CONST_N:
                self.CONST_N(pc)
                pc += 4

            elif opcode == POP:
                if we_are_jitted():
                    _ = self.POP(dummy=True)
                else:
                    _ = self.POP(dummy=False)

            elif opcode == POP1:
                self.POP1()

            elif opcode == DUP:
                self.DUP()

            elif opcode == DUPN:
                self.DUPN(pc)

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
                self.RAND_INT()

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

                # create a new frame
                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL(self, t, argnum, dummy=True)
                else:
                    entry = t
                    if t < pc:
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=frame)
                    frame.CALL(self, t, argnum, dummy=False)

            elif opcode == CALL_N:

                t = _construct_value(bytecode, pc)
                argnum = ord(bytecode[pc + 4])
                pc += 5

                # create a new frame
                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL(self, t, argnum, dummy=True)
                else:
                    entry = t
                    if t < pc:
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=frame)
                    frame.CALL(self, t, argnum, dummy=False)

            elif opcode == CALL_ASSEMBLER:
                t = ord(bytecode[pc])
                argnum = ord(bytecode[pc + 1])
                pc += 2

                # create a new frame
                frame = self.copy_frame(argnum, pc)

                if we_are_jitted():
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=True)
                else:
                    entry = t
                    if t < pc:
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=frame)
                    frame.CALL_ASSEMBLER(self, t, argnum, bytecode, t_empty(), dummy=False)

            elif opcode == RET:
                argnum = hint(ord(bytecode[pc]), promote=True)
                pc += 1
                if we_are_jitted():
                    if tstack.t_is_empty():
                        w_x = self.POP(dummy=True)
                        jit.emit_ret(w_x)
                        pc = entry
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=entry, tstack=tstack, self=self)
                    else:
                        w_x = self.POP(dummy=True)
                        pc, tstack = tstack.t_pop()
                        jit.emit_ret(w_x)
                else:
                    return self.RET(argnum)

            elif opcode == JUMP:
                t = ord(bytecode[pc])

                # if t < pc:
                #     # pc is incremented just after fetching opcode
                #     if bytecode.counts[pc-1] == TRACE_THRESHOLD:
                #         raise ContinueInTracingJIT(pc-1)
                #     bytecode.counts[pc-1] += 1

                if we_are_jitted():
                    if tstack.t_is_empty():
                        if t < pc:
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                        pc = t
                    else:
                        pc, tstack = tstack.t_pop()

                    if t < pc:
                        jit.emit_jump(t)
                else:
                    if t < pc:
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
                        jit.emit_jump(t)
                else:
                    if t < pc:
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                    pc = t

            elif opcode == JUMP_IF:
                target = ord(bytecode[pc])
                pc += 1

                if we_are_jitted():
                    if self.is_true():
                        tstack = t_push(pc, tstack)
                        pc = target
                    else:
                        tstack = t_push(target, tstack)
                else:
                    if self.is_true():
                        if target < pc:
                            entry = target
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=target, tstack=tstack, self=self)
                        pc = target

            elif opcode == JUMP_IF_N:
                target = _construct_value(bytecode, pc)
                pc += 4

                if we_are_jitted():
                    if self.is_true():
                        tstack = t_push(pc, tstack)
                        pc = target
                    else:
                        tstack = t_push(target, tstack)

                else:
                    if self.is_true():
                        if target < pc:
                            entry = target
                            tier1driver.can_enter_jit(
                                bytecode=bytecode, entry=entry, pc=target, tstack=tstack, self=self)
                        pc = target

            elif opcode == EXIT:
                if we_are_jitted():
                    if tstack.t_is_empty():
                        w_x = self.POP(dummy=True)
                        jit.emit_ret(w_x)
                        pc = entry
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=pc, tstack=tstack, self=self)
                    else:
                        w_x = self.POP(dummy=True)
                        pc, tstack = tstack.t_pop()
                        jit.emit_ret(w_x)
                else:
                    return self.POP()

            elif opcode == PRINT:
                self.PRINT()

            elif opcode == FRAME_RESET:
                old_arity = ord(bytecode[pc])
                local_size = ord(bytecode[pc+1])
                new_arity = ord(bytecode[pc+2])
                pc += 3
                self.FRAME_RESET(old_arity, local_size, new_arity)

            elif opcode == NOP:
                continue

            else:
                assert False, 'Unknown opcode: %s' % bytecodes[opcode]


def run(bytecode, w_arg, debug=False, tier=1):
    frame = Frame(bytecode)
    frame.push(w_arg)
    if tier >= 2:
        w_result = frame._interp()
    else:
        w_result = frame.interp()
    return w_result


def _run(bytecode, w_arg, debug=False, tier=1):
    frame = Frame(bytecode)
    frame.push(w_arg)
    if tier >= 2:
        w_result = frame._interp()
        return w_result
    else:
        pc = 0
        while True:
            try:
                w_result = frame.interp(pc=pc)
                return w_result
            except ContinueInTracingJIT as e:
                print "switching to tracing", e.pc
                pc = e.pc

            try:
                w_result = frame._interp(pc=pc)
                return w_result
            except ContinueInThreadedJIT as e:
                print "swiching to threaded", e.pc
                pc = e.pc
