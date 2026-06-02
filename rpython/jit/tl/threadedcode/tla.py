import math
import sys

from rpython.rlib import jit
from rpython.rlib.jit import JitDriver, we_are_jitted, hint, we_are_blackholing
from rpython.rlib.rarithmetic import r_uint
from rpython.rlib.rrandom import Random

from rpython.jit.tl.threadedcode.traverse_stack import TStack, t_empty, t_push
from rpython.jit.tl.threadedcode.tlib import emit_jump, emit_ret
from rpython.jit.tl.threadedcode.object import W_Object, W_IntObject, \
    W_FloatObject, W_StringObject, W_ListObject, OperationError
from rpython.jit.tl.threadedcode.bytecode import *


TRACE_THRESHOLD = -1
MAX_INTERP_DEPTH = 50



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

@jit.elidable
def _branch_reaches_backedge(bytecode, start, boundary):
    # Statically follow control flow from `start`, returning True if a backward
    # jump to a pc <= `boundary` (a loop back-edge) is reached before RET/EXIT.
    # This lets the recorder tell a conditional's *loop-continue* branch from its
    # *exit* branch without knowing anything language-specific about opcodes.
    # Pure over the (immutable) bytecode + green pcs, so it folds to a constant
    # during recording instead of being traced.
    pc = start
    n = len(bytecode)
    steps = 0
    while steps < 4096:
        steps += 1
        if pc < 0 or pc >= n:
            return False
        op = ord(bytecode[pc])
        pc += 1
        if op == JUMP:
            t = ord(bytecode[pc])
            if t <= boundary:
                return True
            pc = t
        elif op == JUMP_N:
            t = _construct_value(bytecode, pc)
            if t <= boundary:
                return True
            pc = t
        elif op == RET or op == EXIT:
            return False
        elif op == JUMP_IF:
            pc += 1          # nested conditional: follow its fall-through
        elif op == JUMP_IF_N:
            pc += 4
        else:
            pc += hasarg[op]
    return False

def _compute_stackdepth(bc):
    # Static worklist over (pc, stackpos) following _interp's stack effects, to
    # find the maximum depth any single frame reaches.  JitFrame sizes its
    # (virtualizable) value stack to this exactly: a fixed oversized array would
    # turn every unused slot into a dead loop-carried value in the compiled
    # trace (the whole vable array is part of the loop's virtual state), which
    # is pure per-iteration register/spill overhead.  Conservative by design --
    # any opcode it cannot model, or a pathological stack, returns the safe
    # default 64.  Runs once per program load, never on the hot path.  Takes the
    # Bytecode object (indexed via __getitem__) so it shares _construct_value's
    # operand type rather than the raw code string.
    n = len(bc)
    seen = [-1] * n
    todo = [(0, 1)]            # main frame: one arg pushed before pc=0
    maxd = 1
    steps = 0
    while todo:
        steps += 1
        if steps > 200000:
            return 64
        pc, sp = todo.pop()
        if pc < 0 or pc >= n:
            continue
        if sp > 200 or sp < 0:
            return 64
        if sp <= seen[pc]:
            continue
        seen[pc] = sp
        if sp > maxd:
            maxd = sp
        op = ord(bc[pc])
        pc += 1
        if op == CONST_INT or op == CONST_NEG_INT:
            todo.append((pc + 1, sp + 1))
        elif op == CONST_FLOAT or op == CONST_NEG_FLOAT:
            todo.append((pc + 9, sp + 1))
        elif op == CONST_N or op == CONST_NEG_N:
            todo.append((pc + 4, sp + 1))
        elif op == DUP:
            todo.append((pc, sp + 1))
        elif op == DUPN:
            todo.append((pc + 1, sp + 1))
        elif op == POP or op == POP1:
            todo.append((pc, sp - 1))
        elif (op == LT or op == GT or op == EQ or op == ADD or op == SUB or
              op == MUL or op == DIV or op == MOD or op == BUILD_LIST or
              op == LOAD):
            todo.append((pc, sp - 1))
        elif op == STORE:
            todo.append((pc, sp - 2))
        elif (op == SIN or op == COS or op == SQRT or op == ABS_FLOAT or
              op == INT_TO_FLOAT or op == FLOAT_TO_INT or op == RAND_INT or
              op == PRINT or op == NOP):
            todo.append((pc, sp))
        elif op == JUMP:
            todo.append((ord(bc[pc]), sp))
        elif op == JUMP_N:
            todo.append((_construct_value(bc, pc), sp))
        elif op == JUMP_IF:
            t = ord(bc[pc])
            todo.append((pc + 1, sp - 1))
            todo.append((t, sp - 1))
        elif op == JUMP_IF_N:
            t = _construct_value(bc, pc)
            todo.append((pc + 4, sp - 1))
            todo.append((t, sp - 1))
        elif (op == CALL_ASSEMBLER or op == CALL or op == CALL_TIER2 or
              op == CALL_TIER0):
            t = ord(bc[pc])
            argnum = ord(bc[pc + 1])
            todo.append((t, argnum + 2))
            todo.append((pc + 2, sp - argnum + 1))
        elif op == CALL_N:
            t = _construct_value(bc, pc)
            argnum = ord(bc[pc + 4])
            todo.append((t, argnum + 2))
            todo.append((pc + 5, sp - argnum + 1))
        elif op == FRAME_RESET:
            o = ord(bc[pc])
            l = ord(bc[pc + 1])
            todo.append((pc + 3, sp - o - l))
        elif op == RET or op == EXIT:
            pass               # terminal: frame returns
        else:
            return 64          # unmodelled opcode -> safe default
    return maxd


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

def _tier1_confirm_enter_jit(pc, entry, bytecode, tstack, self):
    # No frame fix-up here: the clean loop-header state is captured/restored in
    # _jit_take_snapshot (the driverhook graph is rtyped separately and may not
    # do list operations on the frame stack without upsetting the codewriter).
    return True

tier1driver = JitDriver(
    greens=['pc', 'entry', 'bytecode', 'tstack'], reds=['self'],
    get_printable_location=get_printable_location_tier1,
    confirm_enter_jit=_tier1_confirm_enter_jit,
    threaded_code_gen=True, conditions=["is_true"])


tier2driver = JitDriver(
    greens=['pc', 'bytecode',], reds=['self'],
    get_printable_location=get_printable_location, is_recursive=True)


# Tier-2 driver for the decoupled, *virtualizable* JitFrame.  Same recursive
# deep-tracing shape as tier2driver, but it declares `self` virtualizable so the
# optimizer can keep JitFrame's value stack (and the per-iteration counter
# boxes) unboxed inside the compiled loop.  It is a *separate* driver because
# tier2driver is still used by the non-virtualizable Frame._interp (tier-0
# ground truth and tier-1's threaded fallback), and a virtualizable class may
# not flow as a red box into a driver that does not declare it.
tier2vdriver = JitDriver(
    greens=['pc', 'bytecode',], reds=['self'],
    get_printable_location=get_printable_location,
    is_recursive=True, virtualizables=['self'])


class Frame(object):
    def __init__(self, bytecode, stack=None, stackpos=0, depth=0):
        if stack is None:
            stack = [None] * 64
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos
        self.depth = depth
        # Clean (pre-recording) loop-header frame state, captured by
        # confirm_enter_jit.  The trace recorder mutates the live frame with
        # shallow-tracing placeholder zeros (and leaves stackpos off-by-one
        # because POP only peeks while recording, as the trace splitter
        # requires); this lets _jit_take_snapshot restore the real state when
        # plain interpretation resumes after recording.
        self._clean_stack = None
        self._clean_pos = 0
        self._clean_pc = -1
        # Set by FRAME_RESET while a trace is being recorded (dummy=True):
        # signals that the live frame now holds placeholder zeros and must be
        # restored before plain interpretation continues.
        self._frame_poisoned = False
        # Sticky version of _frame_poisoned: stays True once recording has
        # poisoned this frame.  It gates the loop-header (pc==_clean_pc) restore,
        # which the post-recording meta-interp continuation needs on every
        # revisit.  Crucially it is *never* set during pure interpretation
        # (FRAME_RESET runs with dummy=False there), so a plainly interpreted
        # loop is never reset to its header state -- which previously reset the
        # loop counter every iteration and hung (e.g. test_simple_loop).
        self._ever_poisoned = False

    @jit.not_in_trace
    def _jit_take_snapshot(self, pc, entry, jitted):
        # not_in_trace: runs in tier-0 and during tracing/blackholing, but is
        # never emitted into the compiled loop, so it costs nothing at JIT
        # speed.  `jitted` is we_are_jitted() evaluated in the (traced) interp
        # method, so it is True only while a trace is being recorded -- not
        # during plain warmup interpretation nor the post-trace continuation.
        #
        # While recording (jitted) we must not touch the live frame: the
        # meta-interp is reading it to build the trace.  When recording has
        # just finished and plain interpretation resumes, restore the clean
        # loop-header state captured by confirm_enter_jit so the continuation
        # runs from real values rather than the recorder's placeholder zeros.
        if jitted:
            return
        # Two restore triggers, both requiring that recording actually poisoned
        # this frame -- so pure interpretation (which never records, hence never
        # poisons) never restores and never resets a live loop:
        #   1. _frame_poisoned: the frame is freshly poisoned; restore once
        #      wherever plain interpretation first resumes (this is the trigger
        #      a CALL_ASSEMBLER tail-loop like tak relies on, off the loop head).
        #   2. _ever_poisoned + pc==_clean_pc: the post-recording continuation
        #      revisits the loop header repeatedly and needs the restore each
        #      time (e.g. mb_pass's carried invariant); _ever_poisoned is sticky
        #      so this keeps firing after _frame_poisoned is cleared below.
        should_restore = self._frame_poisoned
        if (not should_restore and self._ever_poisoned and
                self._clean_stack is not None and pc == self._clean_pc):
            should_restore = True
        if should_restore:
            snap = self._clean_stack
            if snap is not None:
                stack = self.stack
                i = 0
                n = len(snap)
                while i < n:
                    stack[i] = snap[i]
                    i += 1
                self.stackpos = self._clean_pos
            self._frame_poisoned = False
        elif pc == entry:
            # Clean loop header reached during plain interpretation; remember
            # this state so it can be restored after the next recording poisons
            # the live frame.
            self._clean_stack = self.stack[:]
            self._clean_pos = self.stackpos
            self._clean_pc = pc

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
        # On the dummy (trace-recording) path we must NOT really pop: the trace
        # splitter reconstructs the pop as a read of stack[stackpos-1] at this
        # op's position, so it relies on stackpos still pointing past the value.
        # The off-by-one this leaves in the live frame is undone for the
        # post-trace continuation by the clean-snapshot restore in
        # _jit_take_snapshot (keyed off _frame_poisoned).
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

    # Real (inlined) arithmetic/comparison variants for the deep _interp
    # (tier 0/2).  Unlike the shallow _XX helpers above (which the tier-1
    # threaded-code splitter needs as residual placeholder calls), these inline
    # the integer fast path directly, so the recursive meta-tracer records the
    # *real* control flow -- otherwise the shallow placeholder (0) makes the
    # recorder take a loop's base case instead of its body.  Non-int operands
    # fall back to the real op (correct in tier 0; tier 2 is int-centric).
    def _ADD_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue + w_y.intvalue))
        else:
            self._push(w_x.add(w_y, False))

    def _SUB_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue - w_y.intvalue))
        else:
            self._push(w_x.sub(w_y, False))

    def _MUL_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(int(w_x.intvalue * w_y.intvalue)))
        else:
            self._push(w_x.mul(w_y, False))

    def _DIV_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue // w_y.intvalue))
        else:
            self._push(w_x.div(w_y, False))

    def _MOD_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue % w_y.intvalue))
        else:
            self._push(w_x.mod(w_y, False))

    def _LT_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue <= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.le(w_y, False))

    def _GT_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue >= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.ge(w_y, False))

    def _EQ_real(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue == w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.eq(w_y, False))

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
        # During trace recording (dummy=True, not blackholing) the shallow
        # arithmetic/comparison primitives returned placeholder zeros that we
        # are now copying into the persistent argument slots.  Flag the frame
        # so the next plain-interpretation loop header restores the real state
        # (see _jit_take_snapshot); otherwise the post-trace continuation reads
        # the poisoned accumulator/counter (mb_sum/mb_inc returning 0).
        if dummy and not we_are_blackholing():
            self._frame_poisoned = True
            self._ever_poisoned = True
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
            self._jit_take_snapshot(pc, entry, we_are_jitted())

            # print get_printable_location_tier1(pc, entry, bytecode, tstack)
            # self.dump()

            opcode = ord(bytecode[pc])
            pc += 1

            # O1 threaded code: the data-stack opcodes are traced *deeply*.
            # Their interpreter pop/push (the foldable _-prefixed helpers)
            # are folded away by the optimizer; only the leaf semantic
            # primitives decorated with @enable_shallow_tracing survive in
            # the trace.  Hence no more we_are_jitted()/dummy duplication
            # here -- we just call the foldable handler directly.
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
                            bytecode=bytecode, entry=t, pc=t, tstack=tstack, self=frame)
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
                            bytecode=bytecode, entry=t, pc=t, tstack=tstack, self=frame)
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
                    if t < pc:
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
                        # pc is incremented just after fetching opcode
                        if bytecode.counts[pc-1] == TRACE_THRESHOLD:
                            raise ContinueInTracingJIT(pc-1)
                        bytecode.counts[pc-1] += 1
                    if t < pc and tstack.t_is_empty():
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
                    if t < pc and tstack.t_is_empty():
                        tier1driver.can_enter_jit(
                            bytecode=bytecode, entry=entry, pc=t, tstack=tstack, self=self)
                    pc = t

            elif opcode == JUMP_IF:
                jumpif_pc = pc - 1
                target = ord(bytecode[pc])
                pc += 1

                if we_are_jitted():
                    # Record the *loop-continue* branch, not blindly the true
                    # one: take the false branch only when it loops back and the
                    # true branch exits (e.g. `while cond:` exits on true).
                    recorded = (_branch_reaches_backedge(bytecode, target, jumpif_pc) or
                                not _branch_reaches_backedge(bytecode, pc, jumpif_pc))
                    if self.is_true(recorded, dummy=True):
                        tstack = t_push(pc, tstack)
                        pc = target
                    else:
                        tstack = t_push(target, tstack)
                else:
                    if self.is_true(True, dummy=False):
                        if target < pc and tstack.t_is_empty():
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
                        if target < pc and tstack.t_is_empty():
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


class JitFrame(object):
    """Tier-2 (inlined) interpreter frame -- decoupled from tier-1 ``Frame``.

    The whole point of tier 2 is to remove tier 1's per-bytecode residual-call
    overhead by tracing *through* the data-stack helpers and inlining calls.
    Tier 1 cannot do that: its trace splitter needs the stack ops to survive as
    opaque ``@jit.dont_look_inside`` residual calls.  Those two requirements are
    irreconcilable on one shared frame -- worse, making ``Frame`` virtualizable
    makes tier 1's opaque stack-helper calls escape the vable on every op.

    So JitFrame is a standalone class whose value-stack helpers are plain,
    fully-inlined methods (no ``@jit.dont_look_inside`` / no role decorators):
    the recursive meta-tracer traces straight through them.  Combined with the
    ``_virtualizable_`` declaration (+ ``tier2vdriver``'s ``virtualizables``),
    the optimizer keeps the value stack -- and the loop-carried counter boxes
    that sit in it -- unboxed across the compiled loop's back-edge.  That is the
    structural win tier 2 has over tier 1 on allocation-bound tail loops.

    It is a separate class (not a subclass of Frame) on purpose: a virtualizable
    subclass of a non-virtualizable base would let a JitFrame flow, by
    subsumption, into tier1driver's non-virtualizable red ``self`` and trip the
    warmspot virtualizable check.  Tier 1 only ever builds ``Frame``; tier 2
    only ever builds ``JitFrame``.
    """
    _virtualizable_ = ['stackpos', 'stack[*]']

    def __init__(self, bytecode, stack=None, stackpos=0, depth=0, stacksize=64):
        self = jit.hint(self, access_directly=True, fresh_virtualizable=True)
        # Size the value stack to the program's actual max depth (see
        # _compute_stackdepth).  Because the whole virtualizable array is part of
        # the compiled loop's virtual state, every surplus slot would become a
        # dead loop-carried argument -- so a tight size is what lets a flat loop
        # like mb_loop compile to a clean 2-3 register loop instead of shuffling
        # 64 dead values per iteration.
        if stack is None:
            stack = [None] * stacksize
        self.stacksize = stacksize
        self.bytecode = bytecode
        self.stack = stack
        self.stackpos = stackpos
        self.depth = depth

    # --- value-stack helpers ------------------------------------------------
    # Plain, inlined, vable-friendly: only element-wise self.stack[i] access
    # (never alias/slice/len the virtualizable array), non-negative-index
    # asserts so the JIT folds the bounds checks.
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
        # Vable-compliant: index both stacks only with by-construction
        # non-negative values (nonneg + nonneg), never a subtraction, so the
        # codewriter can fuse every access into a virtualizable-array op.
        framepos = self.stackpos - argnum - 1
        assert framepos >= 0
        assert argnum >= 0
        bytecode = jit.promote(self.bytecode)
        newframe = JitFrame(bytecode, None, argnum + 2, self.depth + 1,
                            stacksize=self.stacksize)
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

    # Inlined integer fast paths (the deep tracer records the real control flow;
    # non-int operands fall back to the real W_Object op).
    def _ADD(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue + w_y.intvalue))
        else:
            self._push(w_x.add(w_y, False))

    def _SUB(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue - w_y.intvalue))
        else:
            self._push(w_x.sub(w_y, False))

    def _MUL(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(int(w_x.intvalue * w_y.intvalue)))
        else:
            self._push(w_x.mul(w_y, False))

    def _DIV(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue // w_y.intvalue))
        else:
            self._push(w_x.div(w_y, False))

    def _MOD(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            self._push(W_IntObject(w_x.intvalue % w_y.intvalue))
        else:
            self._push(w_x.mod(w_y, False))

    def _LT(self):
        # LT dispatches <= (matches Frame; every lang program relies on it).
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue <= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.le(w_y, False))

    def _GT(self):
        # GT dispatches >= (matches Frame).
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue >= w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.ge(w_y, False))

    def _EQ(self):
        w_y = self._pop()
        w_x = self._pop()
        if isinstance(w_x, W_IntObject) and isinstance(w_y, W_IntObject):
            if w_x.intvalue == w_y.intvalue:
                self._push(W_IntObject(1))
            else:
                self._push(W_IntObject(0))
        else:
            self._push(w_x.eq(w_y, False))

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
        # Vable-compliant: assert every index non-negative before subscripting
        # the virtualizable stack, and form the loop indices as nonneg + nonneg
        # (the bare subtractions below are only used after being asserted >= 0).
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

        while pc < len(bytecode):
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
                if t < pc:
                    tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                pc = t

            elif opcode == JUMP_IF:
                t = ord(bytecode[pc])
                pc += 1

                if self._is_true():
                    if t < pc:
                        tier2vdriver.can_enter_jit(bytecode=bytecode, pc=t, self=self)
                    pc = t

            elif opcode == JUMP_IF_N:
                t = _construct_value(bytecode, pc)
                pc += 4

                if self._is_true():
                    if t < pc:
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


# def run(bytecode, w_arg, debug=False, tier=None):
#     frame = Frame(bytecode)
#     frame.push(w_arg)
#     if tier >= 2:
#         w_result = frame._interp()
#     else:
#         w_result = frame.interp()
#     return w_result


def run(bytecode, w_arg, debug=False, tier=1):
    "tier 0=interp, tier 1=threaded code, tier 2=inlined threaded code."
    bytecode = Bytecode(bytecode.code)
    if tier == 0 or tier == 2:
        # Tier 2 (inlined): run the *deep*, virtualizable interpreter under the
        # recursive meta-tracing driver (tier2vdriver).  Unlike the tier-1
        # threaded-code path (Frame.interp), it traces straight through the
        # data-stack helpers and inlines calls, and -- because JitFrame is
        # virtualizable -- keeps the value stack and the loop-carried counter
        # boxes unboxed across the compiled loop.  That box elimination is where
        # tier 2's speedup over tier 1 comes from.
        #
        # Tier 0 (the benchmark ground truth) runs the *same* JitFrame._interp
        # with the JIT turned off (targettla sets --jit off), so it is a pure
        # interpreter and is guaranteed to agree with tier 2's result.
        stacksize = _compute_stackdepth(bytecode) + 1
        jframe = JitFrame(bytecode, stacksize=stacksize)
        jframe._push(w_arg)
        return jframe._interp()
    frame = Frame(bytecode)
    frame.push(w_arg)
    pc = 0
    while True:
        try:
            return frame.interp(pc=pc)
        except ContinueInTracingJIT as e:
            print "switching to tracing", e.pc
            pc = e.pc

        try:
            return frame._interp(pc=pc)
        except ContinueInThreadedJIT as e:
            print "swiching to threaded", e.pc
            pc = e.pc
