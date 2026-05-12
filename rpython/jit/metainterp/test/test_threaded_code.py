import math
import sys

import py
import pytest
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
    isconstant, isvirtual, set_param, record_exact_class, not_in_trace)
from rpython.rlib.longlong2float import float2longlong, longlong2float
from rpython.rlib.rarithmetic import ovfcheck, is_valid_int, int_force_ge_zero
from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.jit.tl.threadedcode.traverse_stack import TStack, t_push, t_empty
from rpython.jit.tl.threadedcode.bytecode import (
    assemble,
    bytecodes,
    hasarg,
    NOP,
    CONST_INT,
    CONST_N,
    DUP,
    LT,
    JUMP,
    JUMP_IF,
    SUB,
    EXIT,
    ADD,
)


def _tla_encode_const(n):
    """Single-byte CONST_INT or 32-bit CONST_N, same encoding as ``tla_assembler`` / ``assemble``."""
    if 0 <= n <= 255:
        return [CONST_INT, n]
    return [
        CONST_N,
        (n >> 24) & 0xFF,
        (n >> 16) & 0xFF,
        (n >> 8) & 0xFF,
        n & 0xFF,
    ]


def _tla_opcode_printable(pc, bytecode):
    op = ord(bytecode[pc])
    name = bytecodes[op]
    na = hasarg[op]
    if na == 0:
        return name
    if na == 1:
        return "%s %s" % (name, ord(bytecode[pc + 1]))
    return "%s ..." % (name,)


class _FrameSnapshot(object):
    __slots__ = ('sp', 'stack')

    def __init__(self, size):
        self.sp = 0
        self.stack = [0] * size


def _make_frame_snapshot_helpers(size):
    """Mutable snapshot for save/restore across threaded can_enter_jit boundaries."""
    snap = _FrameSnapshot(size)

    @not_in_trace
    def save_state(frame):
        snap.sp = frame.sp
        for i in range(size):
            snap.stack[i] = frame.stack[i]

    @not_in_trace
    def restore_state(frame):
        for i in range(size):
            frame.stack[i] = snap.stack[i]
        frame.sp = snap.sp

    return save_state, restore_state


class ThreadedMiniLangFrame(object):
    """Operand stack matching this file's bytecode (ints), not ``tla.Frame`` (``W_Object``)."""
    size = 64

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.stack = [0] * self.size
        self.sp = 0

    def push(self, x):
        self.stack[self.sp] = x
        self.sp += 1

    def pop(self):
        self.sp -= 1
        v = self.stack[self.sp]
        self.stack[self.sp] = 0
        return v

    def dup(self):
        self.push(self.stack[self.sp - 1])

    def const_int(self, v):
        self.push(v)

    def add(self):
        b = self.pop()
        a = self.pop()
        self.push(a + b)

    def sub(self):
        b = self.pop()
        a = self.pop()
        self.push(a - b)

    def lt(self):
        b = self.pop()
        a = self.pop()
        self.push(1 if a < b else 0)

    def is_true(self):
        return self.pop() != 0


def _make_threaded_stack_interp(
        myjitdriver, code, save_state, restore_state, emit_jump, emit_ret):
    """Run TLA bytecode from a ``code`` list (``tla_assembler`` / ``assemble`` input)."""

    def interp(x):
        bytecode = assemble(code)
        tstack = t_empty()
        pc = 0
        frame = ThreadedMiniLangFrame(bytecode)
        frame.push(x)
        entry_state = pc, tstack
        while True:
            myjitdriver.jit_merge_point(
                pc=pc, entry_state=entry_state, bytecode=bytecode, tstack=tstack,
                frame=frame)
            op = ord(bytecode[pc])
            pc += 1
            if op == CONST_INT:
                v = ord(bytecode[pc])
                pc += 1
                frame.const_int(v)
            elif op == CONST_N:
                v = (
                    ord(bytecode[pc]) << 24
                    | ord(bytecode[pc + 1]) << 16
                    | ord(bytecode[pc + 2]) << 8
                    | ord(bytecode[pc + 3]))
                pc += 4
                frame.const_int(v)
            elif op == DUP:
                frame.dup()
            elif op == ADD:
                frame.add()
            elif op == SUB:
                frame.sub()
            elif op == LT:
                frame.lt()
            elif op == NOP:
                pass
            elif op == JUMP:
                t = ord(bytecode[pc])
                if we_are_jitted():
                    if tstack.t_is_empty():
                        pc = t
                    else:
                        pc, tstack = tstack.t_pop()
                        pc = emit_jump(pc, t, None)
                else:
                    if t < pc:
                        entry_state = t, tstack
                        save_state(frame)
                        myjitdriver.can_enter_jit(
                            pc=t, entry_state=entry_state, bytecode=bytecode,
                            tstack=tstack, frame=frame)
                    pc = t
            elif op == JUMP_IF:
                t = ord(bytecode[pc])
                if frame.is_true():
                    if we_are_jitted():
                        pc += 1
                        tstack = t_push(pc, tstack)
                    else:
                        if t < pc:
                            entry_state = t, tstack
                            save_state(frame)
                            myjitdriver.can_enter_jit(
                                pc=t, entry_state=entry_state, bytecode=bytecode,
                                tstack=tstack, frame=frame)
                    pc = t
                else:
                    if we_are_jitted():
                        tstack = t_push(t, tstack)
                    pc += 1
            elif op == EXIT:
                if we_are_jitted():
                    if tstack.t_is_empty():
                        v = frame.pop()
                        pc, tstack = entry_state
                        pc = emit_ret(pc, v)
                        restore_state(frame)
                        myjitdriver.can_enter_jit(
                            pc=pc, entry_state=entry_state, bytecode=bytecode,
                            tstack=tstack, frame=frame)
                    else:
                        pc, tstack = tstack.t_pop()
                        v = frame.pop()
                        pc = emit_ret(pc, v)
                else:
                    return frame.pop()
            else:
                raise AssertionError("unsupported opcode")

    interp.oopspec = 'jit.not_in_trace()'
    return interp


class BasicTests:
    @pytest.mark.skip(reason="currently the case that red variables are"
                      "number cannot work correctly")
    def test_minilang_num_1(self):

        @dont_look_inside
        def lt(lhs, rhs):
            if lhs < rhs:
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
            return x > 0

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

        myjitdriver = JitDriver(greens=['pc', 'bytecode', 'tstack'], reds=['x',],
                                get_printable_location=opcode_to_string,
                                threaded_code_gen=True,
                                conditions=['is_true'])
        def interp(x):
            tstack = t_empty()
            pc = 0
            # bytecode = [NOP, JUMP_IF, 5, JUMP, 8, SUB, JUMP, 1, EXIT]
            bytecode = [NOP, SUB, JUMP_IF, 1, EXIT]
            while True:
                myjitdriver.jit_merge_point(pc=pc, bytecode=bytecode, tstack=tstack, x=x)
                op = bytecode[pc]
                pc += 1
                if op == ADD:
                    x = add(x, 1)
                elif op == SUB:
                    x = sub(x, 1)
                elif op == JUMP:
                    t = int(bytecode[pc])
                    if we_are_jitted():
                        if tstack.t_is_empty():
                            pc = t
                        else:
                            pc, tstack = tstack.t_pop()
                            pc = emit_jump(pc, t, None)
                    else:
                        if t < pc:
                            myjitdriver.can_enter_jit(pc=t, bytecode=bytecode, tstack=tstack, x=x)
                        pc = t
                elif op == JUMP_IF:
                    t = int(bytecode[pc])
                    if is_true(x):
                        if we_are_jitted():
                            pc += 1
                            if not t < pc:
                                tstack = t_push(pc, tstack)
                            pc = emit_jump(pc, t, None)
                        else:
                            if t < pc:
                                myjitdriver.can_enter_jit(pc=t, bytecode=bytecode, tstack=tstack, x=x)
                            pc = t
                    else:
                        if we_are_jitted():
                            tstack = t_push(t, tstack)
                        pc += 1
                elif op == EXIT:
                    if we_are_jitted():
                        if tstack.t_is_empty():
                            return x
                        else:
                            pc, tstack = tstack.t_pop()
                            pc = emit_ret(pc, x)
                    else:
                        return x

        interp.oopspec = 'jit.not_in_trace()'
        res = self.meta_interp(interp, [100])

    def test_minilang_stack_1(self):

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x, y):
            return x

        def opcode_to_string(pc, entry_state, bytecode, tstack):
            return "%s: %s, tstack top: %s" % (
                pc, _tla_opcode_printable(pc, bytecode), tstack.pc)

        myjitdriver = JitDriver(greens=['pc', 'entry_state', 'bytecode', 'tstack'], reds=['frame'],
                                get_printable_location=opcode_to_string,
                                threaded_code_gen=True,
                                conditions=['is_true'])

        save_state, restore_state = _make_frame_snapshot_helpers(
            ThreadedMiniLangFrame.size)

        # Same shape as a ``code = [...]`` list fed to ``tla_assembler`` (``assemble``).
        code = [
            NOP, DUP, CONST_INT, 1, LT, JUMP_IF, 9, JUMP, 14, CONST_INT, 1, SUB,
            JUMP, 1, EXIT]
        interp = _make_threaded_stack_interp(
            myjitdriver, code, save_state, restore_state, emit_jump, emit_ret)
        res = self.meta_interp(interp, [100])

    def test_minilang_stack_2(self):

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x, y):
            return x

        def opcode_to_string(pc, entry_state, bytecode, tstack):
            return "%s: %s, tstack top: %s" % (
                pc, _tla_opcode_printable(pc, bytecode), tstack.pc)

        myjitdriver = JitDriver(greens=['pc', 'entry_state', 'bytecode', 'tstack'], reds=['frame'],
                                get_printable_location=opcode_to_string,
                                threaded_code_gen=True,
                                conditions=['is_true'])

        save_state, restore_state = _make_frame_snapshot_helpers(
            ThreadedMiniLangFrame.size)

        code = []
        code.append(DUP)
        code.extend(_tla_encode_const(0))
        code.append(LT)
        code.extend([JUMP_IF, 8])
        code.extend([JUMP, 13])
        code.extend(_tla_encode_const(1))
        code.append(SUB)
        code.extend([JUMP, 0])
        code.extend(_tla_encode_const(1000))
        code.append(SUB)
        code.append(EXIT)
        interp = _make_threaded_stack_interp(
            myjitdriver, code, save_state, restore_state, emit_jump, emit_ret)
        res = self.meta_interp(interp, [100])

    def test_minilang_stack_inline_handler(self):
        """Regression: ``threaded_inline_handler=True`` must compile and run a
        loop with multiple per-bytecode handlers per trace block."""

        @dont_look_inside
        def emit_jump(x, y, z):
            return x

        @dont_look_inside
        def emit_ret(x, y):
            return x

        def opcode_to_string(pc, entry_state, bytecode, tstack):
            return "%s: %s, tstack top: %s" % (
                pc, _tla_opcode_printable(pc, bytecode), tstack.pc)

        myjitdriver = JitDriver(
            greens=['pc', 'entry_state', 'bytecode', 'tstack'], reds=['frame'],
            get_printable_location=opcode_to_string,
            threaded_code_gen=True,
            threaded_inline_handler=True,
            conditions=['is_true'])

        save_state, restore_state = _make_frame_snapshot_helpers(
            ThreadedMiniLangFrame.size)

        code = [
            NOP, DUP, CONST_INT, 1, LT, JUMP_IF, 9, JUMP, 14, CONST_INT, 1, SUB,
            JUMP, 1, EXIT]
        interp = _make_threaded_stack_interp(
            myjitdriver, code, save_state, restore_state, emit_jump, emit_ret)
        res = self.meta_interp(interp, [100])


class TestLLtype(BasicTests, LLJitMixin):
    pass
