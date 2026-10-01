import py

from rpython.jit.codewriter.genextension import (
    GenExtension, _int_as_str, _float_as_str)
from rpython.rlib.rarithmetic import ovfcheck
from rpython.rtyper.lltypesystem import lltype
from rpython.jit.metainterp.blackhole import (
    LeaveFrame, BlackholeInterpreter, plain_int, signedord)


class RunModeUnsupported(Exception):
    pass


def _todo(name):
    def emit(self):
        raise NotImplementedError("run mode: %s not implemented" % name)
    return emit


class RunModeGenerator(GenExtension):

    def generate_run(self):
        """Build jit_run(): a dispatch loop with one block per pc, registers
        kept in locals (loaded on entry, written ones stored back in a
        finally)"""
        self._scan_pcs()
        self.used_regs = set()       # (kind, index) loaded into locals
        self.written_regs = set()    # (kind, index) stored back on exit
        self.code = []
        prefix = ""
        for pc in sorted(self.pc_to_insn):
            self._select_pc(pc)
            meth = getattr(self, "emit_run_" + self.name, None)
            try:
                if meth is None:
                    lines = self.emit_run_default()
                else:
                    lines = meth()
            except RunModeUnsupported:
                lines = self.emit_run_fallback()
            self.code.append("%sif pc == %s: # %s" % (prefix, pc, self.name))
            for line in lines:
                self.code.append("    " + line)
            prefix = "el"
        self.code.append("else:")
        self.code.append("    return pc")
        source = ["def jit_run(bh): # %s" % self.jitcode.name,
                  "    pc = bh.position"]
        for kind, index in sorted(self.used_regs | self.written_regs):
            source.append("    %s%d = bh.registers_%s[%d]" %
                          (kind, index, kind, index))
        indent = 4
        if self.written_regs:
            source.append("    try:")
            indent = 8
        source.append(" " * indent + "while 1:")
        for line in self.code:
            source.append(" " * (indent + 4) + line)
        if self.written_regs:
            source.append("    finally:")
            for kind, index in sorted(self.written_regs):
                source.append("        bh.registers_%s[%d] = %s%d" %
                              (kind, index, kind, index))
        self.jitcode._genext_run_source = "\n".join(source)
        d = {"plain_int": plain_int, "ovfcheck": ovfcheck, "LeaveFrame": LeaveFrame}
        d.update(self.globals)
        exec py.code.Source(self.jitcode._genext_run_source).compile() in d
        return d["jit_run"]

    def _select_pc(self, pc):
        self._reset_insn()
        self.insn = self.pc_to_insn[pc]
        self.pc = pc
        instruction = self.insns[ord(self.jitcode.code[pc])]
        self.name, self.argcodes = instruction.split("/")

    def _scan_pcs(self):
        from rpython.jit.codewriter.flatten import Label
        for index, insn in enumerate(self.ssarepr.insns):
            if isinstance(insn[0], Label) or insn[0] == '---':
                continue
            pc = self.ssarepr._insns_pos[index]
            self.pc_to_insn[pc] = insn
            self.pc_to_index[pc] = index
            if index == len(self.ssarepr.insns) - 1:
                self.pc_to_nextpc[pc] = len(self.jitcode.code)
            else:
                self.pc_to_nextpc[pc] = self.ssarepr._insns_pos[index + 1]

    def _bhimpl(self):
        func = getattr(BlackholeInterpreter, "bhimpl_" + self.name, None)
        if func is None:
            raise RunModeUnsupported(self.name)
        return func.im_func

    def _run_arg(self, argcode, position):
        code = self.jitcode.code
        if argcode == 'c':
            return str(signedord(code[position]))
        if argcode in ('i', 'r', 'f'):
            return self._run_reg(argcode, ord(code[position]))
        raise RunModeUnsupported(self.name)

    def _run_reg(self, kind, index):
        """Source for reading a register: a literal for a jitcode constant
        (these sit at index >= num_regs_X), else its local"""
        jc = self.jitcode
        if kind == 'i':
            if index >= jc.num_regs_i():
                value = jc.constants_i[index - jc.num_regs_i()]
                return _int_as_str(value, lltype.typeOf(value),
                                   self._add_global)
        elif kind == 'r':
            if index >= jc.num_regs_r():
                return self._add_global(jc.constants_r[index - jc.num_regs_r()])
        elif kind == 'f':
            if index >= jc.num_regs_f():
                value = jc.constants_f[index - jc.num_regs_f()]
                return _float_as_str(value, lltype.typeOf(value),
                                     self._add_global)
        self.used_regs.add((kind, index))
        return "%s%d" % (kind, index)

    def _run_list(self, argcode, position):
        code = self.jitcode.code
        length = ord(code[position])
        regs = [self._run_reg(argcode.lower(), ord(code[position + 1 + i]))
                for i in range(length)]
        return "[%s]" % ", ".join(regs), position + 1 + length

    def _run_portal_args(self):
        position = self.pc + 1
        args = [self._run_arg(self.argcodes[0], position)]
        position += 1
        for argcode in self.argcodes[1:7]:
            assert argcode in 'IRF'
            text, position = self._run_list(argcode, position)
            args.append(text)
        return args, position

    def _run_dest(self, kind, index):
        """Source for writing a register; written locals are stored back"""
        self.written_regs.add((kind, index))
        return "%s%d" % (kind, index)

    def emit_run_default(self):
        func = self._bhimpl()
        code = self.jitcode.code
        position = self.pc + 1
        next_argcode = 0
        args = []
        for argtype in func.argtypes:
            if argtype not in ('i', 'r', 'f'):
                raise RunModeUnsupported(self.name)
            args.append(self._run_arg(self.argcodes[next_argcode], position))
            next_argcode += 1
            position += 1
        restype = func.resulttype
        if restype not in (None, 'i', 'r', 'f'):
            raise RunModeUnsupported(self.name)
        call = "%s(%s)" % (self._add_global(func), ", ".join(args))
        if restype is None:
            lines = [call]
        else:
            if restype == 'i':
                call = "plain_int(%s)" % call
            dest = self._run_dest(restype, ord(code[position]))
            lines = ["%s = %s" % (dest, call)]
        lines.append("pc = %s" % self.pc_to_nextpc[self.pc])
        lines.append("continue")
        return lines

    def emit_run_goto(self):
        return ["pc = %s" % self._decode_label(self.pc + 1), "continue"]

    def _emit_branch(self, nargs, cond):
        position = self.pc + 1
        if self.argcodes[nargs] != 'L':
            raise RunModeUnsupported(self.name)
        args = [self._run_arg(self.argcodes[i], position + i)
                for i in range(nargs)]
        return ["if %s:" % cond(*args),
                "    pc = %s" % self.pc_to_nextpc[self.pc],
                "else:",
                "    pc = %s" % self._decode_label(position + nargs),
                "continue"]

    def emit_run_goto_if_not(self):
        return self._emit_branch(1, lambda a: a)

    def emit_run_int_return(self):
        value = self._run_arg(self.argcodes[0], self.pc + 1)
        return ["bh.tmpreg_i = %s" % value,
                "bh._return_type = 'i'",
                "return -1"]

    def emit_run_goto_if_not_int_lt(self):
        return self._emit_branch(2, lambda a, b: "%s < %s" % (a, b))

    def emit_run_goto_if_not_int_eq(self):
        return self._emit_branch(2, lambda a, b: "%s == %s" % (a, b))

    def emit_run_goto_if_not_int_gt(self):
        return self._emit_branch(2, lambda a, b: "%s > %s" % (a, b))

    def emit_run_goto_if_not_int_is_true(self):
        return self._emit_branch(1, lambda a: a)

    def emit_run_goto_if_not_int_is_zero(self):
        return self._emit_branch(1, lambda a: "%s == 0" % a)

    def emit_run_int_add_jump_if_ovf(self):
        position = self.pc + 1
        if self.argcodes[0] != 'L':
            raise RunModeUnsupported(self.name)
        target = self._decode_label(position)
        a = self._run_arg(self.argcodes[1], position + 2)
        b = self._run_arg(self.argcodes[2], position + 3)
        result = self._run_dest('i', ord(self.jitcode.code[position + 4]))
        return ["try:",
                "    %s = ovfcheck(%s + %s)" % (result, a, b),
                "except OverflowError:",
                "    pc = %s" % target,
                "    continue",
                "pc = %s" % self.pc_to_nextpc[self.pc],
                "continue"]

    def emit_run_strlen(self):
        position = self.pc + 1
        string = self._run_arg(self.argcodes[0], position)
        result = self._run_dest('i', ord(self.jitcode.code[position + 1]))
        return ["%s = bh.cpu.bh_strlen(%s)" % (result, string),
                "pc = %s" % self.pc_to_nextpc[self.pc],
                "continue"]

    def emit_run_goto_if_not_ptr_iszero(self):
        return self._emit_branch(1, lambda a: "not %s" % a)

    def emit_run_goto_if_not_ptr_nonzero(self):
        return self._emit_branch(1, lambda a: a)

    def emit_run_ref_return(self):
        value = self._run_arg(self.argcodes[0], self.pc + 1)
        return ["bh.tmpreg_r = %s" % value,
                "bh._return_type = 'r'",
                "return -1"]

    def emit_run_void_return(self):
        return ["bh._return_type = 'v'",
                "return -1"]


    emit_run_fallback = _todo("fallback")

    # TODO: How to implement exceptions?
    emit_run_raise = _todo("raise")
    emit_run_reraise = _todo("reraise")
    emit_run_catch_exception = _todo("catch_exception")
    emit_run_goto_if_exception_mismatch = _todo("goto_if_exception_mismatch")

    # TODO: How to implement switch?
    emit_run_switch = _todo("switch")

    emit_run_getfield_gc_i = _todo("getfield_gc_i")
    emit_run_getfield_gc_i_pure = _todo("getfield_gc_i_pure")
    emit_run_getfield_gc_r = _todo("getfield_gc_r")
    emit_run_getfield_gc_r_pure = _todo("getfield_gc_r_pure")
    emit_run_getfield_raw_i = _todo("getfield_raw_i")
    emit_run_getfield_vable_i = _todo("getfield_vable_i")
    emit_run_getfield_vable_r = _todo("getfield_vable_r")
    emit_run_setfield_gc = _todo("setfield_gc")
    emit_run_setfield_gc_i = _todo("setfield_gc_i")
    emit_run_setfield_gc_r = _todo("setfield_gc_r")
    emit_run_setfield_vable_i = _todo("setfield_vable_i")
    emit_run_getarrayitem_gc_i = _todo("getarrayitem_gc_i")
    emit_run_getarrayitem_gc_i_pure = _todo("getarrayitem_gc_i_pure")
    emit_run_getarrayitem_gc_r = _todo("getarrayitem_gc_r")
    emit_run_getarrayitem_gc_r_pure = _todo("getarrayitem_gc_r_pure")
    emit_run_getarrayitem_vable_r = _todo("getarrayitem_vable_r")
    emit_run_setarrayitem_gc = _todo("setarrayitem_gc")
    emit_run_setarrayitem_gc_r = _todo("setarrayitem_gc_r")
    emit_run_setarrayitem_vable_r = _todo("setarrayitem_vable_r")
    emit_run_new = _todo("new")
    emit_run_new_array_clear = _todo("new_array_clear")
    emit_run_new_with_vtable = _todo("new_with_vtable")
    emit_run_newstr = _todo("newstr")
    emit_run_strgetitem = _todo("strgetitem")
    emit_run_strsetitem = _todo("strsetitem")
    emit_run_copystrcontent = _todo("copystrcontent")

    emit_run_residual_call_r_i = _todo("residual_call_r_i")
    emit_run_residual_call_r_r = _todo("residual_call_r_r")
    emit_run_residual_call_ir_i = _todo("residual_call_ir_i")

    def _make_emit_recursive_call(restype):
        def emit(self):
            args, position = self._run_portal_args()
            nextpc = self.pc_to_nextpc[self.pc]
            # Naive approach: call blackhole implementations
            call = "bh.bhimpl_recursive_call_%s(%s)" % (
                restype, ", ".join(args))
            lines = ["bh.position = %d" % nextpc]
            if restype == 'v':
                lines.append(call)
            else:
                if restype == 'i':
                    call = "plain_int(%s)" % call
                result = self._run_dest(restype, ord(self.jitcode.code[position]))
                lines.append("%s = %s" % (result, call))
            lines += ["pc = %d" % nextpc, "continue"]
            return lines
        return emit

    emit_run_recursive_call_i = _make_emit_recursive_call("i")
    emit_run_recursive_call_f = _make_emit_recursive_call("f")
    emit_run_recursive_call_r = _make_emit_recursive_call("r")
    emit_run_recursive_call_v = _make_emit_recursive_call("v")

    def _make_emit_inline_call(argtypes, restype):
        def emit(self):
            args, position = self._run_portal_args()
            nextpc = self.pc_to_nextpc[self.pc]
            call = "bh.bhimpl_inline_call_%s_%s(%s)" % (
                argtypes, restype, ", ".join(args))

            lines = ["bh.position = %d" % nextpc]
            if restype == 'v':
                lines.append(call)
            else:
                if restype == 'i':
                    call = "plain_int(%s)" % call
                result = self._run_dest(restype, ord(self.jitcode.code[position]))
                lines.append("%s = %s" % (result, call))
            lines += ["pc = %d" % nextpc, "continue"]
            return lines
        return emit

    emit_run_inline_call_r_i = _make_emit_inline_call("r", "i")
    emit_run_inline_call_r_r = _make_emit_inline_call("r", "r")
    emit_run_inline_call_r_v = _make_emit_inline_call("r", "v")

    emit_run_inline_call_ir_i = _make_emit_inline_call("ir", "i")
    emit_run_inline_call_ir_r = _make_emit_inline_call("ir", "r")
    emit_run_inline_call_ir_v = _make_emit_inline_call("ir", "v")

    emit_run_guard_class = _todo("guard_class")
    emit_run_guard_nonnull = _todo("guard_nonnull")

    def emit_run_live(self):
        return ["pc = %d" % self.pc_to_nextpc[self.pc]]

    def _emit_run_portal_point(name):
        def emit(self):
            args, _ = self._run_portal_args()
            return ["bh.position = %d" % self.pc_to_nextpc[self.pc],
                    "try:",
                    "    bh.bhimpl_%s(%s)" % (name, ", ".join(args)),
                    "except LeaveFrame:",
                    "    return -1"]
        return emit

    emit_run_jit_merge_point = _emit_run_portal_point("jit_merge_point")
    emit_run_pe_bailout_point = _todo("pe_bailout_point")


def generate_run_function(genext):
    gen = RunModeGenerator(genext.assembler, genext.ssarepr, genext.jitcode)
    return gen.generate_run()
