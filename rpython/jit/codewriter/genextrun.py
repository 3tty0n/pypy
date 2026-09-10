import py

from rpython.jit.codewriter.genextension import GenExtension
from rpython.rlib.rarithmetic import ovfcheck
from rpython.jit.metainterp.blackhole import (
    BlackholeInterpreter, plain_int, signedord)


class RunModeUnsupported(Exception):
    pass


def _todo(name):
    def emit(self):
        raise NotImplementedError("run mode: %s not implemented" % name)
    return emit


class RunModeGenerator(GenExtension):
    pass

    def generate_run(self):
        self._scan_pcs()
        self.code = []
        prefix = ""
        for pc in sorted(self.pc_to_insn):
            self._reset_insn()
            self.insn = self.pc_to_insn[pc]
            self.pc = pc
            instruction = self.insns[ord(self.jitcode.code[pc])]
            self.name, self.argcodes = instruction.split("/")
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
                  "    pc = bh.position",
                  "    while 1:"]
        for line in self.code:
            source.append(" " * 8 + line)
        self.jitcode._genext_run_source = "\n".join(source)
        d = {"plain_int": plain_int, "ovfcheck": ovfcheck}
        d.update(self.globals)
        exec py.code.Source(self.jitcode._genext_run_source).compile() in d
        return d["jit_run"]

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
            return "bh.registers_%s[%s]" % (argcode, ord(code[position]))
        raise RunModeUnsupported(self.name)

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
            lines = ["bh.registers_%s[%s] = %s" %
                     (restype, ord(code[position]), call)]
        lines.append("pc = %s" % self.pc_to_nextpc[self.pc])
        lines.append("continue")
        return lines

    def emit_run_goto(self):
        return ["pc = %s" % self._decode_label(self.pc + 1), "continue"]

    def emit_run_goto_if_not(self):
        position = self.pc + 1
        cond = self._run_arg(self.argcodes[0], position)
        position += 1
        if self.argcodes[1] != 'L':
            raise RunModeUnsupported(self.name)
        return ["if not %s:" % cond,
                "    pc = %s" % self._decode_label(position),
                "else:",
                "    pc = %s" % self.pc_to_nextpc[self.pc],
                "continue"]

    def emit_run_int_return(self):
        value = self._run_arg(self.argcodes[0], self.pc + 1)
        return ["bh.tmpreg_i = %s" % value,
                "bh._return_type = 'i'",
                "return -1"]

    def emit_run_goto_if_not_int_lt(self):
        position = self.pc + 1
        a = self._run_arg(self.argcodes[0], position)
        b = self._run_arg(self.argcodes[1], position + 1)
        if self.argcodes[2] != 'L':
            raise RunModeUnsupported(self.name)
        return ["if %s < %s:" % (a, b),
                "    pc = %s" % self.pc_to_nextpc[self.pc],
                "else:",
                "    pc = %s" % self._decode_label(position + 2),
                "continue"]

    def emit_run_int_add_jump_if_ovf(self):
        position = self.pc + 1
        if self.argcodes[0] != 'L':
            raise RunModeUnsupported(self.name)
        target = self._decode_label(position)
        a = self._run_arg(self.argcodes[1], position + 2)
        b = self._run_arg(self.argcodes[2], position + 3)
        result = ord(self.jitcode.code[position + 4])
        return ["try:",
                "    bh.registers_i[%s] = ovfcheck(%s + %s)" % (result, a, b),
                "except OverflowError:",
                "    pc = %s" % target,
                "    continue",
                "pc = %s" % self.pc_to_nextpc[self.pc],
                "continue"]

    def emit_run_strlen(self):
        position = self.pc + 1
        string = self._run_arg(self.argcodes[0], position)
        result = ord(self.jitcode.code[position + 1])
        return ["bh.registers_i[%s] = bh.cpu.bh_strlen(%s)" % (result, string),
                "pc = %s" % self.pc_to_nextpc[self.pc],
                "continue"]

    emit_run_fallback = _todo("fallback")

    emit_run_catch_exception = _todo("catch_exception")
    emit_run_goto_if_exception_mismatch = _todo("goto_if_exception_mismatch")
    emit_run_goto_if_not_int_eq = _todo("goto_if_not_int_eq")
    emit_run_goto_if_not_int_gt = _todo("goto_if_not_int_gt")
    emit_run_goto_if_not_int_is_true = _todo("goto_if_not_int_is_true")
    emit_run_goto_if_not_int_is_zero = _todo("goto_if_not_int_is_zero")
    emit_run_goto_if_not_ptr_iszero = _todo("goto_if_not_ptr_iszero")
    emit_run_goto_if_not_ptr_nonzero = _todo("goto_if_not_ptr_nonzero")
    emit_run_switch = _todo("switch")

    emit_run_ref_return = _todo("ref_return")
    emit_run_void_return = _todo("void_return")
    emit_run_raise = _todo("raise")
    emit_run_reraise = _todo("reraise")

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
    emit_run_recursive_call_i = _todo("recursive_call_i")
    emit_run_inline_call_r_i = _todo("inline_call_r_i")
    emit_run_inline_call_r_r = _todo("inline_call_r_r")
    emit_run_inline_call_r_v = _todo("inline_call_r_v")
    emit_run_inline_call_ir_i = _todo("inline_call_ir_i")
    emit_run_inline_call_ir_r = _todo("inline_call_ir_r")
    emit_run_inline_call_ir_v = _todo("inline_call_ir_v")

    emit_run_guard_class = _todo("guard_class")
    emit_run_guard_nonnull = _todo("guard_nonnull")

    emit_run_live = _todo("-live-")
    emit_run_jit_merge_point = _todo("jit_merge_point")
    emit_run_pe_bailout_point = _todo("pe_bailout_point")


def generate_run_function(genext):
    gen = RunModeGenerator(genext.assembler, genext.ssarepr, genext.jitcode)
    return gen.generate_run()
