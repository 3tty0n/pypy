"""Pure, class-independent helpers for the TLA threaded-code interpreter.

Extracted from tla.py: the JIT printable-location callbacks, bytecode value
decoders, the @jit.elidable static control-flow / entry scanners, the value
stack-depth computation and the tier-1 confirm-enter-jit hook -- plus the shared
constants and the tier-switch exceptions.  None of these depend on the Frame /
JitFrame classes, so they live here and are imported by frames.py and tla.py.
"""
import math
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
        elif op == RET:
            pc += 1
        elif op == JUMP_IF:
            pc += 1          # nested conditional: follow its fall-through
        elif op == JUMP_IF_N:
            pc += 4
        else:
            pc += hasarg[op]
    return False


@jit.elidable
def _entry_has_foreign_call_assembler(bytecode, entry):
    pc = entry
    n = len(bytecode)
    while pc < n:
        op = ord(bytecode[pc])
        pc += 1
        if op == CONST_INT or op == CONST_NEG_INT:
            pc += 1
        elif op == CONST_FLOAT or op == CONST_NEG_FLOAT:
            pc += 9
        elif op == CONST_N:
            pc += 4
        elif op == DUPN:
            pc += 1
        elif op == CALL_ASSEMBLER:
            t = ord(bytecode[pc])
            if t != entry:
                return True
            pc += 2
        elif op == CALL_N:
            t = _construct_value(bytecode, pc)
            if t != entry:
                return True
            pc += 5
        elif op == CALL or op == CALL_TIER2 or op == CALL_TIER0:
            pc += 2
        elif op == JUMP or op == JUMP_IF:
            pc += 1
        elif op == JUMP_N or op == JUMP_IF_N:
            pc += 4
        elif op == RET:
            pc += 1
        elif op == FRAME_RESET:
            pc += 3
    return False

@jit.elidable
def _entry_has_wide_call_assembler(bytecode, entry):
    if not we_are_translated_to_c():
        return False
    pc = entry
    n = len(bytecode)
    while pc < n:
        op = ord(bytecode[pc])
        pc += 1
        if op == CONST_INT or op == CONST_NEG_INT:
            pc += 1
        elif op == CONST_FLOAT or op == CONST_NEG_FLOAT:
            pc += 9
        elif op == CONST_N:
            pc += 4
        elif op == DUPN:
            pc += 1
        elif op == CALL_ASSEMBLER:
            t = ord(bytecode[pc])
            argnum = ord(bytecode[pc + 1])
            if t == entry and argnum > 1:
                return True
            pc += 2
        elif op == CALL_N:
            t = _construct_value(bytecode, pc)
            argnum = ord(bytecode[pc + 4])
            if t == entry and argnum > 1:
                return True
            pc += 5
        elif op == CALL or op == CALL_TIER2 or op == CALL_TIER0:
            pc += 2
        elif op == JUMP or op == JUMP_IF:
            pc += 1
        elif op == JUMP_N or op == JUMP_IF_N:
            pc += 4
        elif op == RET or op == EXIT:
            return False
        elif op == FRAME_RESET:
            pc += 3
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

@jit.elidable
def _entry_has_array_op(bytecode, entry):
    n = len(bytecode)
    seen = [False] * n
    todo = [entry]
    while todo:
        pc = todo.pop()
        if pc < 0 or pc >= n or seen[pc]:
            continue
        seen[pc] = True
        op = ord(bytecode[pc])
        pc += 1
        if op == LOAD or op == STORE or op == BUILD_LIST:
            return True
        if op == RET or op == EXIT:
            continue
        if op == JUMP:
            todo.append(ord(bytecode[pc]))
        elif op == JUMP_N:
            todo.append(_construct_value(bytecode, pc))
        elif op == JUMP_IF:
            todo.append(ord(bytecode[pc]))
            todo.append(pc + 1)
        elif op == JUMP_IF_N:
            todo.append(_construct_value(bytecode, pc))
            todo.append(pc + 4)
        else:
            todo.append(pc + hasarg[op])
    return False

@jit.elidable
def _entry_has_assembler_call(bytecode, entry):
    # Tier 1 pays one residual call per shallow opcode.  That tradeoff is useful
    # for threaded recursive entries, where CALL_ASSEMBLER stitches nested traces
    # together, but plain in-frame tail loops (mb_sum) are faster and safer in
    # the interpreter.
    n = len(bytecode)
    seen = [False] * n
    todo = [entry]
    while todo:
        pc = todo.pop()
        if pc < 0 or pc >= n or seen[pc]:
            continue
        seen[pc] = True
        op = ord(bytecode[pc])
        pc += 1
        if op == CALL_ASSEMBLER or op == CALL_N:
            return True
        if op == RET or op == EXIT:
            continue
        if op == JUMP:
            todo.append(ord(bytecode[pc]))
        elif op == JUMP_N:
            todo.append(_construct_value(bytecode, pc))
        elif op == JUMP_IF:
            todo.append(ord(bytecode[pc]))
            todo.append(pc + 1)
        elif op == JUMP_IF_N:
            todo.append(_construct_value(bytecode, pc))
            todo.append(pc + 4)
        else:
            todo.append(pc + hasarg[op])
    return False

def _tier1_confirm_enter_jit(pc, entry, bytecode, tstack, self):
    # No frame fix-up here: the clean loop-header state is captured/restored in
    # _jit_take_snapshot (the driverhook graph is rtyped separately and may not
    # do list operations on the frame stack without upsetting the codewriter).
    return (not _entry_has_array_op(bytecode, entry) and
            _entry_has_assembler_call(bytecode, entry))

@jit.elidable
def _tier1_use_frame_inliner_for_plain_loops(bytecode):
    if _entry_has_array_op(bytecode, 0):
        return False
    pc = 0
    n = len(bytecode)
    while pc < n:
        op = ord(bytecode[pc])
        pc += 1
        if op == LOAD or op == STORE or op == BUILD_LIST:
            return False
        if op == CALL_ASSEMBLER:
            t = ord(bytecode[pc])
            if _entry_has_array_op(bytecode, t):
                return False
            if _entry_has_assembler_call(bytecode, t):
                return False
            pc += 2
        elif op == CALL_N:
            t = _construct_value(bytecode, pc)
            if _entry_has_array_op(bytecode, t):
                return False
            if _entry_has_assembler_call(bytecode, t):
                return False
            pc += 5
        else:
            pc += hasarg[op]
    return True
