bytecodes = []
hasarg = []

def define_op(name, has_arg=False):
    globals()[name] = len(bytecodes)
    bytecodes.append(name)
    hasarg.append(has_arg)

_bytecodes_has_args = [
    ('NOP', 0),
    ('CONST_INT', 1),
    ('CONST_NEG_INT', 1),
    ('CONST_FLOAT', 9),
    ('CONST_NEG_FLOAT', 9),
    ('CONST_N', 4),
    ('CONST_NEG_N', 4),
    ('DUP', 0),
    ('DUPN', 1),
    ('POP', 0),
    ('POP1', 0),
    ('LT', 0),
    ('GT', 0),
    ('EQ', 0),
    ('ADD', 0),
    ('SUB', 0),
    ('MUL', 0),
    ('DIV', 0),
    ('MOD', 0),
    ('EXIT', 0),
    ('JUMP', 1),
    ('JUMP_N', 4),
    ('JUMP_IF', 1),
    ('JUMP_IF_N', 4),
    ('CALL', 2),
    ('CALL_N', 5),
    ('CALL_ASSEMBLER', 2),
    ('CALL_TIER2', 2),
    ('CALL_TIER0', 2),
    ('RET', 1),
    ('NEWSTR', 1),
    ('FRAME_RESET', 3),
    ('PRINT', 0),
    ('LOAD', 0),
    ('STORE', 0),
    ('BUILD_LIST', 0),
    ('RAND_INT', 4),
    ('FLOAT_TO_INT', 0),
    ('INT_TO_FLOAT', 0),
    ('ABS_FLOAT', 1),
    ('SIN', 0),
    ('COS', 0),
    ('SQRT', 0)
]

for bytecode, has_arg in _bytecodes_has_args:
    define_op(bytecode, has_arg)

class CompilerContext(object):
    def __init__(self):
        self.data = []
        self.stack = []
        self.names_to_numbers = {}
        self.functions = {}

    def register_function(self, pos, f):
        self.functions[pos] = f

    def register_constant(self, val):
        self.stack.append(val)
        return len(self.stack) - 1

    def register_assignment(self, var, val):
        self.names_to_numbers[var] = val
        self.stack.append(val)
        return len(self.stack) - 1

    def emit(self, bc, arg=0):
        raise NotImplementedError

    def create_bytecode(self):
        raise NotImplementedError

class Bytecode(object):
    # `code` is write-once (the compiled program).  `poly` is quasi-immutable:
    # the tier-4 dispatch reads poly[site] every op, so folding it to a constant
    # (instead of a per-op getarrayitem + guard off a mutable array that the
    # in-loop stores prevent hoisting) removes the entire hybrid trace overhead.
    # A changed decision replaces the whole array (see _t4_set_poly), which
    # invalidates the affected traces.  The profiling counters (counts/seen/
    # cnt_a/cnt_b) stay plain mutable -- they are only touched in the interpreted
    # warmup, never in a compiled trace.
    _immutable_fields_ = ['code', 'poly?[*]']

    def __init__(self, code):
        self.code = code
        self.counts = [0] * len(code)
        self.seen = [0] * len(code)
        self.poly = [0] * len(code)
        # Frequency-aware (tier-4 adaptive) operand-type profile: per arithmetic
        # site, cnt_a counts int/int executions and cnt_b counts every other
        # operand-type signature.  The inline-vs-residual decision is driven by
        # the *minority* fraction min(cnt_a, cnt_b) / (cnt_a + cnt_b) rather than
        # by a single "saw two types" event, so a dominant-type-with-rare-off-type
        # site stays inlined (fast) instead of being residualised forever.
        self.cnt_a = [0] * len(code)
        self.cnt_b = [0] * len(code)
        # DRR (tier-4 deopt-rate re-decision, ratio>1 only): per arithmetic site
        # bails = off-type guard-bail replays under the blackhole, inl_runs = its
        # inlined (interpreted) executions, redecided = one-shot latch.  Plain
        # mutable and touched only off the compiled trace, like cnt_a/cnt_b, so
        # they are NOT in _immutable_fields_.
        self.bails = [0] * len(code)
        self.inl_runs = [0] * len(code)
        self.redecided = [0] * len(code)
        # Tier-4 adaptive compilation state.  0 means "profiling/baseline";
        # otherwise it is the concrete compiler tier selected for this bytecode.
        self.adaptive_invocations = 0
        self.adaptive_tier = 0
        # off-trace only; not in _immutable_fields_
        self.reopt_retry = 0
        self.reopt_baseline = 0

    def __len__(self):
        return len(self.code)

    def __getitem__(self, i):
        return self.code[i]

    def __setitem__(self, i, v):
        self.code[i] = v

    def dump(self):
        lines = []
        i = 0
        while i < len(self.code):
            c = ord(self.code[i])
            name, arg_num = _bytecodes_has_args[c]
            op_str = name
            if arg_num:
                arg_str = ""
                for j in range(arg_num):
                    arg = ord(self.code[j + 1])
                    arg_str = arg_str + ", " + str(arg)
                op_str = op_str + arg_str
                i += arg_num
            lines.append(op_str + ",")
            i += 1

        return '\n'.join(lines)

def assemble(mylist):
    return ''.join([chr(x) for x in mylist])


def compile(file_name):
    # see ../tlopcode.py
    pass
