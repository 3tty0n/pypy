"""
Binding-time analysis (BTA) against Python bytecode.

Lightweight forward dataflow that classifies, per bytecode offset:
  - stack-slot binding times
  - local-variable binding times

Binding-time lattice (flat):
    BT_STATIC  <  BT_DYNAMIC
    BT_UNKNOWN is the conservative top.

The analysis is intentionally simple (single-pass, no fixpoint for
back-edges) so it can run in RPython at trace-entry time.

Integration point (pyjitpl.py):
  - At trace entry, MetaInterp extracts pycode from the greenkey,
    runs analyse_pycode(), and stashes BTAInfo.
  - debug_merge_point updates the "current Python bytecode offset".
  - execute_and_record checks BTA before recording: if the current
    Python opcode is classified as producing a static result, and the
    jitcode op being recorded is pure, we constant-fold and skip
    recording entirely.
"""

# ---------------------------------------------------------------------------
# Binding-time lattice
# ---------------------------------------------------------------------------

BT_STATIC  = 0
BT_DYNAMIC = 1
BT_UNKNOWN = 2


def bt_meet(a, b):
    if a == BT_STATIC and b == BT_STATIC:
        return BT_STATIC
    if a == BT_UNKNOWN:
        return b
    if b == BT_UNKNOWN:
        return a
    return BT_DYNAMIC


def bt_is_static(bt):
    return bt == BT_STATIC


# ---------------------------------------------------------------------------
# Opcode constants (Python 2.7 subset used by PyPy)
# We hardcode the common ones to avoid importing opcode at runtime in
# translated builds.
# ---------------------------------------------------------------------------

HAVE_ARGUMENT = 90

# Stack / locals
LOAD_CONST    = 100
LOAD_FAST     = 124
STORE_FAST    = 125
LOAD_NAME     = 101
STORE_NAME    = 90
LOAD_GLOBAL   = 116
STORE_GLOBAL  = 97
LOAD_DEREF    = 136
STORE_DEREF   = 137
LOAD_CLOSURE  = 135

# Stack manipulation
POP_TOP       = 1
DUP_TOP       = 4
ROT_TWO       = 2
ROT_THREE     = 3

# Arithmetic / bitwise (binary)
BINARY_ADD        = 23
BINARY_SUBTRACT   = 24
BINARY_MULTIPLY   = 20
BINARY_DIVIDE     = 21
BINARY_FLOOR_DIVIDE = 26
BINARY_TRUE_DIVIDE  = 27
BINARY_MODULO     = 22
BINARY_POWER      = 19
BINARY_LSHIFT     = 62
BINARY_RSHIFT     = 63
BINARY_AND        = 64
BINARY_XOR        = 65
BINARY_OR         = 66

# Unary
UNARY_POSITIVE    = 10
UNARY_NEGATIVE    = 11
UNARY_NOT         = 12
UNARY_INVERT      = 15

# Compare
COMPARE_OP        = 107

# Subscript
BINARY_SUBSCR     = 25
STORE_SUBSCR      = 60
DELETE_SUBSCR     = 61

# Jumps
JUMP_ABSOLUTE         = 113
JUMP_FORWARD          = 110
JUMP_IF_FALSE_OR_POP  = 111
JUMP_IF_TRUE_OR_POP   = 112
POP_JUMP_IF_FALSE     = 114
POP_JUMP_IF_TRUE      = 115
FOR_ITER              = 93
BREAK_LOOP            = 80
CONTINUE_LOOP         = 119
SETUP_LOOP            = 120
SETUP_EXCEPT          = 121
SETUP_FINALLY         = 122

# Calls
CALL_FUNCTION       = 131
CALL_FUNCTION_VAR   = 140
CALL_FUNCTION_KW    = 141
CALL_FUNCTION_VAR_KW = 142

# Returns / misc
RETURN_VALUE    = 83
PRINT_EXPR      = 70
PRINT_ITEM      = 71
PRINT_ITEM_TO   = 72
PRINT_NEWLINE   = 73
PRINT_NEWLINE_TO = 74
POP_BLOCK       = 87
END_FINALLY     = 88
WITH_CLEANUP    = 81
RAISE_VARARGS   = 130
BUILD_TUPLE     = 102
BUILD_LIST      = 103
BUILD_SET       = 104
BUILD_MAP       = 105
BUILD_SLICE     = 133
LIST_APPEND     = 94
SET_ADD         = 146
MAP_ADD         = 147
UNPACK_SEQUENCE = 92
EXTENDED_ARG    = 143


# Opcodes that are *always* static in their effect (no runtime dependence).
# LOAD_CONST is the canonical example.
_STATIC_OPCODES = (LOAD_CONST,)

# Pure binary ops: static if both inputs are static.
_PURE_BINOP_OPCODES = (
    BINARY_ADD, BINARY_SUBTRACT, BINARY_MULTIPLY,
    BINARY_FLOOR_DIVIDE, BINARY_TRUE_DIVIDE, BINARY_MODULO,
    BINARY_POWER,
    BINARY_LSHIFT, BINARY_RSHIFT,
    BINARY_AND, BINARY_XOR, BINARY_OR,
)

# Pure unary ops: static if input is static.
_PURE_UNARY_OPCODES = (
    UNARY_POSITIVE, UNARY_NEGATIVE, UNARY_NOT, UNARY_INVERT,
)

# Opcodes that produce dynamic outputs regardless of inputs.
_ALWAYS_DYNAMIC_OPCODES = (
    CALL_FUNCTION, CALL_FUNCTION_VAR, CALL_FUNCTION_KW,
    CALL_FUNCTION_VAR_KW,
    BINARY_SUBSCR,
    LOAD_NAME, LOAD_GLOBAL, LOAD_DEREF,
    BUILD_MAP,
)

# Opcodes with no stack push (or push of a known constant).
_NO_RESULT_OPCODES = (
    POP_TOP, STORE_FAST, STORE_NAME, STORE_GLOBAL, STORE_DEREF,
    STORE_SUBSCR, DELETE_SUBSCR,
    PRINT_EXPR, PRINT_ITEM, PRINT_ITEM_TO, PRINT_NEWLINE, PRINT_NEWLINE_TO,
    POP_BLOCK, END_FINALLY, WITH_CLEANUP, BREAK_LOOP,
    JUMP_ABSOLUTE, JUMP_FORWARD, JUMP_IF_FALSE_OR_POP, JUMP_IF_TRUE_OR_POP,
    POP_JUMP_IF_FALSE, POP_JUMP_IF_TRUE, FOR_ITER,
    CONTINUE_LOOP, SETUP_LOOP, SETUP_EXCEPT, SETUP_FINALLY,
    RETURN_VALUE, RAISE_VARARGS,
)


# ---------------------------------------------------------------------------
# Stack effect (simplified, for common opcodes)
# ---------------------------------------------------------------------------

def _stack_effect(opcode, oparg):
    if opcode in _STATIC_OPCODES:
        return 1
    if opcode in _PURE_BINOP_OPCODES:
        return -1   # pop 2, push 1
    if opcode in _PURE_UNARY_OPCODES:
        return 0    # pop 1, push 1
    if opcode == BINARY_SUBSCR:
        return -1   # pop 2, push 1
    if opcode == COMPARE_OP:
        return -1
    if opcode == LOAD_FAST or opcode == LOAD_NAME or opcode == LOAD_GLOBAL or opcode == LOAD_DEREF:
        return 1
    if opcode == STORE_FAST or opcode == STORE_NAME or opcode == STORE_GLOBAL or opcode == STORE_DEREF:
        return -1
    if opcode == POP_TOP:
        return -1
    if opcode == DUP_TOP:
        return 1
    if opcode == ROT_TWO or opcode == ROT_THREE:
        return 0
    if opcode == CALL_FUNCTION:
        return -(oparg & 0xFF) - ((oparg >> 8) & 0xFF)
    if opcode == CALL_FUNCTION_VAR or opcode == CALL_FUNCTION_KW:
        return -(oparg & 0xFF) - ((oparg >> 8) & 0xFF) - 1
    if opcode == CALL_FUNCTION_VAR_KW:
        return -(oparg & 0xFF) - ((oparg >> 8) & 0xFF) - 2
    if opcode == BUILD_TUPLE or opcode == BUILD_LIST or opcode == BUILD_SET:
        return 1 - oparg
    if opcode == BUILD_MAP:
        return 1
    if opcode == BUILD_SLICE:
        return 1 - oparg
    if opcode == UNPACK_SEQUENCE:
        return oparg - 1
    if opcode == LIST_APPEND:
        return -1
    if opcode == SET_ADD or opcode == MAP_ADD:
        return -1
    if opcode == RETURN_VALUE:
        return -1
    if opcode == FOR_ITER:
        return 1   # push iterator result, or jump
    if opcode == JUMP_IF_FALSE_OR_POP or opcode == JUMP_IF_TRUE_OR_POP:
        return -1
    if opcode == POP_JUMP_IF_FALSE or opcode == POP_JUMP_IF_TRUE:
        return -1
    # Conservative default
    return 0


# ---------------------------------------------------------------------------
# BTA info container
# ---------------------------------------------------------------------------

class BTAInfo(object):
    """Result of binding-time analysis for one PyCode.

    Attributes (all flat lists indexed by bytecode offset):
        offset_bt     - binding time of the value produced by this opcode
                        (for stack-manipulating ops: the value pushed)
        stack_top_bt  - binding time of the stack top *after* this opcode
        local_bt      - flat list of binding times for local variables
                        (only updated by STORE_FAST)
    """

    def __init__(self, co_code, num_locals):
        self.co_code = co_code
        self.num_offsets = len(co_code)
        self.num_locals = num_locals
        # per-offset state
        self.offset_bt = [BT_UNKNOWN] * self.num_offsets
        self.stack_top_bt = [BT_UNKNOWN] * self.num_offsets
        # local variable state at each offset (flattened: offset * nlocals + idx)
        self.local_bt = [BT_UNKNOWN] * (self.num_offsets * num_locals)

    def _local_idx(self, offset, local_index):
        return offset * self.num_locals + local_index

    def get_local_bt(self, offset, local_index):
        if local_index < 0 or local_index >= self.num_locals:
            return BT_DYNAMIC
        return self.local_bt[self._local_idx(offset, local_index)]

    def set_local_bt(self, offset, local_index, bt):
        if 0 <= local_index < self.num_locals:
            self.local_bt[self._local_idx(offset, local_index)] = bt

    def get_offset_bt(self, offset):
        if 0 <= offset < self.num_offsets:
            return self.offset_bt[offset]
        return BT_UNKNOWN

    def is_result_static(self, offset):
        return self.get_offset_bt(offset) == BT_STATIC

    def can_skip_pure_op(self, offset, opnum_name):
        """Heuristic: can we skip recording a pure jitcode op that
        corresponds to the Python opcode at `offset`?"""
        if offset < 0 or offset >= self.num_offsets:
            return False
        return self.offset_bt[offset] == BT_STATIC


# ---------------------------------------------------------------------------
# Bytecode decoder
# ---------------------------------------------------------------------------

def _decode_oparg(co_code, offset):
    """Return (opcode, oparg, next_offset)."""
    opcode = ord(co_code[offset])
    offset += 1
    if opcode >= HAVE_ARGUMENT:
        lo = ord(co_code[offset])
        hi = ord(co_code[offset + 1])
        oparg = (hi << 8) | lo
        offset += 2
        # handle EXTENDED_ARG
        while opcode == EXTENDED_ARG:
            opcode = ord(co_code[offset])
            if opcode < HAVE_ARGUMENT:
                break
            lo = ord(co_code[offset + 1])
            hi = ord(co_code[offset + 2])
            offset += 3
            oparg = (oparg * 65536) | (hi << 8) | lo
    else:
        oparg = 0
    return opcode, oparg, offset


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_pycode_data(co_code, co_nlocals, co_argcount, co_name):
    """Run a lightweight forward BTA on raw Python bytecode data.

    Returns a BTAInfo.  The analysis is intentionally conservative:
    it does a single linear pass and treats all merge points as
    dynamic unless both incoming edges agree on STATIC.
    """
    num_locals = co_nlocals
    info = BTAInfo(co_code, num_locals)

    # Current dataflow state
    stack = []          # list of binding times, top at end
    # Conservative: all locals start as DYNAMIC unless proven static.
    locals_state = [BT_DYNAMIC] * num_locals

    # Initialize argument locals as DYNAMIC (conservative).
    for i in range(min(num_locals, co_argcount)):
        locals_state[i] = BT_DYNAMIC

    offset = 0
    while offset < len(co_code):
        opcode, oparg, next_offset = _decode_oparg(co_code, offset)

        # Compute stack effect and output binding time for this opcode
        out_bt = BT_UNKNOWN

        if opcode in _STATIC_OPCODES:
            out_bt = BT_STATIC
            stack.append(out_bt)

        elif opcode in _PURE_BINOP_OPCODES:
            if len(stack) >= 2:
                a = stack.pop()
                b = stack.pop()
                out_bt = bt_meet(a, b)
            else:
                out_bt = BT_DYNAMIC
            stack.append(out_bt)

        elif opcode in _PURE_UNARY_OPCODES:
            if len(stack) >= 1:
                a = stack.pop()
                out_bt = a
            else:
                out_bt = BT_DYNAMIC
            stack.append(out_bt)

        elif opcode == LOAD_FAST:
            out_bt = locals_state[oparg] if oparg < num_locals else BT_DYNAMIC
            stack.append(out_bt)

        elif opcode == STORE_FAST:
            if len(stack) >= 1:
                val_bt = stack.pop()
            else:
                val_bt = BT_DYNAMIC
            if oparg < num_locals:
                locals_state[oparg] = val_bt
            out_bt = BT_UNKNOWN  # no push

        elif opcode == DUP_TOP:
            if len(stack) >= 1:
                out_bt = stack[-1]
            else:
                out_bt = BT_DYNAMIC
            stack.append(out_bt)

        elif opcode == POP_TOP:
            if len(stack) >= 1:
                stack.pop()
            out_bt = BT_UNKNOWN

        elif opcode == ROT_TWO:
            if len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            out_bt = BT_UNKNOWN

        elif opcode == ROT_THREE:
            if len(stack) >= 3:
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                stack.append(a)
                stack.append(c)
                stack.append(b)
            out_bt = BT_UNKNOWN

        elif opcode in _ALWAYS_DYNAMIC_OPCODES:
            se = _stack_effect(opcode, oparg)
            if se < 0:
                for _ in range(-se):
                    if stack:
                        stack.pop()
            elif se > 0:
                for _ in range(se):
                    stack.append(BT_DYNAMIC)
            out_bt = BT_DYNAMIC

        elif opcode in _NO_RESULT_OPCODES:
            se = _stack_effect(opcode, oparg)
            if se < 0:
                for _ in range(-se):
                    if stack:
                        stack.pop()
            elif se > 0:
                for _ in range(se):
                    stack.append(BT_UNKNOWN)
            out_bt = BT_UNKNOWN

        else:
            # Unknown opcode: conservative
            se = _stack_effect(opcode, oparg)
            if se < 0:
                for _ in range(-se):
                    if stack:
                        stack.pop()
            elif se > 0:
                for _ in range(se):
                    stack.append(BT_DYNAMIC)
            out_bt = BT_DYNAMIC

        # Record result
        info.offset_bt[offset] = out_bt
        info.stack_top_bt[offset] = stack[-1] if stack else BT_UNKNOWN
        for i in range(num_locals):
            info.set_local_bt(offset, i, locals_state[i])

        offset = next_offset

    return info


def analyse_pycode(pycode):
    """Backward-compatible wrapper that extracts raw fields from a PyCode
    object and delegates to analyse_pycode_data."""
    return analyse_pycode_data(
        pycode.co_code,
        pycode.co_nlocals,
        pycode.co_argcount,
        pycode.co_name,
    )


# ---------------------------------------------------------------------------
# Convenience: is a pycode worth analysing?
# ---------------------------------------------------------------------------

def should_analyse_pycode(pycode):
    """Quick filter: only analyse user code (not internal RPython code)."""
    # Heuristic: if the code name starts with an underscore or is
    # a standard internal name, skip.
    name = pycode.co_name
    if name.startswith('<'):
        return False
    if name.startswith('__'):
        return False
    return True


# ---------------------------------------------------------------------------
# Module-level cache (RPython-safe: keyed by co_code string)
# ---------------------------------------------------------------------------

_bta_cache = {}

def _cached_analyse_pycode_data(co_code, co_nlocals, co_argcount, co_name):
    """Wrapper around analyse_pycode_data with module-level cache."""
    cached = _bta_cache.get(co_code, None)
    if cached is not None:
        return cached
    info = analyse_pycode_data(co_code, co_nlocals, co_argcount, co_name)
    _bta_cache[co_code] = info
    return info
