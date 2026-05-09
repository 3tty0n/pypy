"""
RPython-compatible bytecode compiler for the mini-language in ``grammar.txt``.

Use ``compile_program`` from translated code. For CPython tests you can combine
the EBNF ``parser.parse`` with ``compile_program``, or use ``tl_rparse`` +
``tl_pipeline``.
"""
from rpython.jit.tl.threadedcode import bytecode as bc
from rpython.jit.tl.threadedcode.tl_ast import (
    Program,
    ConstInt,
    ConstFloat,
    Variable,
    BinOp,
    LetIn,
    Function,
    FunApp,
    ArrayMake,
    ArrayLoad,
    ArrayStore,
    If,
    While,
)

NO_FUNC_ADDR = -1


class CompileError(Exception):
    """Raised when the program is ill-formed for compilation."""


def _copy_bindings(bindings):
    out = {}
    for k in bindings:
        out[k] = bindings[k]
    return out


class _Context(object):
    def __init__(self):
        self.code = []
        self.stack_top = 0
        self.bindings = {}
        self.functions = {}

    def here(self):
        return len(self.code)

    def emit_opc(self, opcode):
        self.code.append(opcode & 0xFF)

    def emit_opc_u8(self, opcode, arg):
        self.code.append(opcode & 0xFF)
        self.code.append(arg & 0xFF)

    def emit_opc_be32(self, opcode, n):
        self.code.append(opcode & 0xFF)
        self.code.append((n >> 24) & 0xFF)
        self.code.append((n >> 16) & 0xFF)
        self.code.append((n >> 8) & 0xFF)
        self.code.append(n & 0xFF)

    def emit_const_int(self, n):
        if n < 0:
            mag = -n
            if mag <= 255:
                self.emit_opc_u8(bc.CONST_NEG_INT, mag)
            else:
                raise CompileError()
        else:
            if n <= 255:
                self.emit_opc_u8(bc.CONST_INT, n)
            else:
                self.emit_opc_be32(bc.CONST_N, n)
        self.stack_top += 1

    def emit_const_float(self, x):
        raise CompileError()

    def emit_dupn_for_slot(self, slot_index):
        n = self.stack_top - 1 - slot_index
        if n < 0:
            raise CompileError()
        if n > 255:
            raise CompileError()
        self.emit_opc_u8(bc.DUPN, n)
        self.stack_top += 1

    def bind_var(self, name, slot_index):
        self.bindings[name] = slot_index

    def forget_var(self, name):
        newd = {}
        for k in self.bindings:
            if k != name:
                newd[k] = self.bindings[k]
        self.bindings = newd

    def lookup_slot(self, name):
        if name not in self.bindings:
            raise CompileError()
        return self.bindings[name]

    def note_pop1(self):
        self.stack_top -= 1

    def note_binop(self):
        self.stack_top -= 1

    def emit_jump_placeholder_u8(self):
        self.emit_opc_u8(bc.JUMP, 0)
        return len(self.code) - 1

    def patch_be32_arg(self, pos, n):
        self.code[pos] = (n >> 24) & 0xFF
        self.code[pos + 1] = (n >> 16) & 0xFF
        self.code[pos + 2] = (n >> 8) & 0xFF
        self.code[pos + 3] = n & 0xFF

    def emit_jump_if_n_placeholder(self):
        self.emit_opc_be32(bc.JUMP_IF_N, 0)
        return len(self.code) - 4

    def emit_jump_n_placeholder(self):
        self.emit_opc_be32(bc.JUMP_N, 0)
        return len(self.code) - 4

    def emit_jump_n(self, target_pc):
        self.emit_opc_be32(bc.JUMP_N, target_pc)

    def patch_u8(self, pos, target):
        if target < 0:
            raise CompileError()
        if target > 255:
            raise CompileError()
        self.code[pos] = target & 0xFF


def compile_expr(node, ctx):
    if isinstance(node, ConstInt):
        ctx.emit_const_int(node.intval)
    elif isinstance(node, ConstFloat):
        ctx.emit_const_float(node.floatval)
    elif isinstance(node, Variable):
        ctx.emit_dupn_for_slot(ctx.lookup_slot(node.val))
    elif isinstance(node, BinOp):
        compile_expr(node.left, ctx)
        compile_expr(node.right, ctx)
        op = node.op
        if op == '+':
            ctx.emit_opc(bc.ADD)
        elif op == '-':
            ctx.emit_opc(bc.SUB)
        elif op == '<':
            ctx.emit_opc(bc.LT)
        elif op == '==':
            ctx.emit_opc(bc.EQ)
        elif op == '*':
            ctx.emit_opc(bc.MUL)
        elif op == '%':
            ctx.emit_opc(bc.MOD)
        elif op == '>':
            ctx.emit_opc(bc.GT)
        else:
            raise CompileError()
        ctx.note_binop()
    elif isinstance(node, If):
        depth_before = ctx.stack_top
        compile_expr(node.condition, ctx)
        ctx.stack_top -= 1
        pos_jif = ctx.emit_jump_if_n_placeholder()
        ctx.stack_top = depth_before
        compile_expr(node.else_expr, ctx)
        pos_skip_then = ctx.emit_jump_n_placeholder()
        then_pc = ctx.here()
        ctx.patch_be32_arg(pos_jif, then_pc)
        ctx.stack_top = depth_before
        compile_expr(node.then_expr, ctx)
        end_pc = ctx.here()
        ctx.patch_be32_arg(pos_skip_then, end_pc)
        ctx.stack_top = depth_before + 1
    elif isinstance(node, While):
        depth_before = ctx.stack_top
        top_pc = ctx.here()
        compile_expr(node.cond_expr, ctx)
        ctx.stack_top -= 1
        pos_jif_body = ctx.emit_jump_if_n_placeholder()
        pos_j_exit = ctx.emit_jump_n_placeholder()
        body_pc = ctx.here()
        ctx.patch_be32_arg(pos_jif_body, body_pc)
        ctx.stack_top = depth_before
        compile_expr(node.body_expr, ctx)
        ctx.stack_top = depth_before
        ctx.emit_jump_n(top_pc)
        exit_pc = ctx.here()
        ctx.patch_be32_arg(pos_j_exit, exit_pc)
        ctx.emit_const_int(0)
        ctx.stack_top = depth_before + 1
    elif isinstance(node, LetIn):
        compile_expr(node.rhs, ctx)
        slot = ctx.stack_top - 1
        ctx.bind_var(node.name, slot)
        compile_expr(node.body, ctx)
        ctx.emit_opc(bc.POP1)
        ctx.note_pop1()
        ctx.forget_var(node.name)
    elif isinstance(node, ArrayMake):
        compile_expr(node.init_expr, ctx)
        compile_expr(node.size_expr, ctx)
        ctx.emit_opc(bc.BUILD_LIST)
        ctx.stack_top -= 1
    elif isinstance(node, ArrayLoad):
        compile_expr(node.array_expr, ctx)
        compile_expr(node.index_expr, ctx)
        ctx.emit_opc(bc.LOAD)
        ctx.stack_top -= 1
    elif isinstance(node, ArrayStore):
        compile_expr(node.value_expr, ctx)
        compile_expr(node.array_expr, ctx)
        compile_expr(node.index_expr, ctx)
        ctx.emit_opc(bc.STORE)
        ctx.stack_top -= 2
    elif isinstance(node, FunApp):
        callee = node.callee
        if not isinstance(callee, Variable):
            raise CompileError()
        if callee.val not in ctx.functions:
            raise CompileError()
        target_arity = ctx.functions[callee.val]
        target = target_arity[0]
        def_arity = target_arity[1]
        if target == NO_FUNC_ADDR:
            raise CompileError()
        arity = len(node.args)
        if arity == 0:
            raise CompileError()
        if def_arity != arity:
            raise CompileError()
        pre = ctx.stack_top
        j = 0
        while j < arity:
            compile_expr(node.args[j], ctx)
            j += 1
        ctx.emit_opc_u8(bc.DUPN, arity - 1)
        ctx.stack_top += 1
        ctx.emit_opc_u8(bc.CALL, target)
        ctx.emit_opc(arity & 0xFF)
        ctx.stack_top = pre + 1
    else:
        raise CompileError()


def compile_program(prog):
    """Return a list of byte values suitable for ``bytecode.assemble``."""
    if not isinstance(prog, Program):
        raise CompileError()
    ctx = _Context()
    funcs = []
    main = []
    i = 0
    while i < len(prog.exprs):
        e = prog.exprs[i]
        if isinstance(e, Function):
            funcs.append(e)
        else:
            main.append(e)
        i += 1

    j = 0
    while j < len(funcs):
        f = funcs[j]
        ctx.functions[f.funcname] = (NO_FUNC_ADDR, len(f.args))
        j += 1

    if len(funcs) > 0:
        skip_jmp_arg_pos = ctx.emit_jump_placeholder_u8()
        k = 0
        while k < len(funcs):
            f = funcs[k]
            start = ctx.here()
            ctx.functions[f.funcname] = (start, len(f.args))
            saved_bindings = _copy_bindings(ctx.bindings)
            saved_top = ctx.stack_top
            ctx.bindings = {}
            arity = len(f.args)
            ctx.stack_top = arity + 2
            idx = 0
            while idx < arity:
                ctx.bind_var(f.args[idx], idx)
                idx += 1
            compile_expr(f.body, ctx)
            ctx.emit_opc_u8(bc.RET, arity & 0xFF)
            ctx.bindings = saved_bindings
            ctx.stack_top = saved_top
            k += 1
        main_start = ctx.here()
        ctx.patch_u8(skip_jmp_arg_pos, main_start)

    mi = 0
    while mi < len(main):
        compile_expr(main[mi], ctx)
        if mi + 1 < len(main):
            ctx.emit_opc(bc.POP1)
            ctx.note_pop1()
        mi += 1
    ctx.emit_opc(bc.EXIT)
    return ctx.code
