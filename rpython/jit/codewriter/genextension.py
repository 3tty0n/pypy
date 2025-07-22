import py
import re
from rpython.jit.metainterp.history import (Const, ConstInt, ConstPtr,
    ConstFloat, CONST_NULL, getkind, AbstractDescr)
from rpython.jit.metainterp import support
from rpython.flowspace.model import Constant
from rpython.jit.codewriter.flatten import Register, TLabel, Label
from rpython.jit.codewriter.jitcode import SwitchDictDescr
from rpython.rtyper.lltypesystem import lltype, llmemory, rstr
from rpython.rtyper.rclass import OBJECTPTR
from rpython.rlib import objectmodel

class GenExtension(object):
    def __init__(self, assembler, ssarepr, jitcode):
        self.assembler = assembler
        self.insns = [None] * len(assembler.insns)
        for insn, index in assembler.insns.iteritems():
            self.insns[index] = insn
        self.ssarepr = ssarepr
        self.jitcode = jitcode
        self.precode = []
        self.pc_to_insn = {}
        self.pc_to_nextpc = {}
        self.pc_to_index = {}
        self.code = []
        self.globals = {}
        self._reset_insn()

    def _reset_insn(self):
        # the following attributes are set for each instruction emitted
        self.name = None
        self.methodname = None
        self.argcodes = None
        self.insn = None
        self.args = None
        self.args_as_objects = None
        self.returncode = None
        self.returnindex = None

    def generate(self):
        from rpython.jit.codewriter.flatten import Label
        from rpython.jit.codewriter.jitcode import JitCode
        from rpython.jit.metainterp.pyjitpl import ChangeFrame
        for index, insn in enumerate(self.ssarepr.insns):
            self._reset_insn()
            if isinstance(insn[0], Label) or insn[0] == '---':
                continue
            pc = self.ssarepr._insns_pos[index]
            self.pc_to_insn[pc] = insn
            if index == len(self.ssarepr.insns) - 1:
                nextpc = len(self.jitcode.code)
            else:
                nextpc = self.ssarepr._insns_pos[index + 1]
            self.pc_to_nextpc[pc] = nextpc
            self.pc_to_index[pc] = index
        self.work_list = WorkList(self.pc_to_insn, self.assembler.label_positions, self.pc_to_nextpc, self.globals)
        code_per_pc = {}
        for startpc in self.assembler.startpoints:
            spec = self.work_list.specialize_pc(frozenset([]), startpc)
        size = len(code_per_pc)
        while len(code_per_pc) != len(self.work_list.specialize_instruction):
            for key, spec in self.work_list.specialize_instruction.items():
                code_per_pc[spec.spec_pc] = spec.make_code(), spec
        assert not self.code
        for pc, (code, spec) in code_per_pc.iteritems():
            if code is None:
                self.code = []
                if spec.constant_registers:
                    spec._emit_sync_registers(self.code)
                    self.code.append("pc = %s" % spec.orig_pc)
                    self.code.append("continue")
                else:
                    self._make_code(self.pc_to_index[spec.orig_pc], spec.insn)
                code_per_pc[pc] = (str(py.code.Source("\n".join(self.code)).deindent()), spec)
        self.code = []
        allconsts = set()
        for pc, (code, spec) in code_per_pc.iteritems():
            allconsts.update(spec.constant_registers)
            self.code.append("if pc == %s: # %s %s" % (pc, spec.insn, spec.constant_registers))
            #self.code.append("    import pdb;pdb.set_trace()")
            self.code.append("    self.pc = %s" % (self.pc_to_nextpc[spec.orig_pc], ))
            for line in str(py.code.Source(code).indent('    ')).splitlines():
                self.code.append(line)
        self.code.append("assert 0 # unreachable")
        allcode = []
        allconsts = sorted(["%s%s" % (val.kind[0], val.index) for val in allconsts])
        self.precode.append("def jit_shortcut(self): # %s" % self.jitcode.name)
        self.precode.append("    pc = self.pc")
        for name in allconsts:
            assert name[0] in 'ir'
            if name[0] == 'i':
                default = '0xcafedead'
            else:
                default = 'lltype.nullptr(llmemory.GCREF.TO)'
            self.precode.append("    %s = %s" % (name, default))
        prefix = ""
        for pc in self.assembler.startpoints:
            self.precode.append("    %sif pc == %s: pc = %s" % (prefix, pc, pc))
            prefix = "el"
        self.precode.append("    else: assert 0, 'unreachable'")
        self.precode.append("    while 1:")
        allcode.extend(self.precode)
        for line in self.code:
            allcode.append(" " * 8 + line)
        self.jitcode._genext_source = "\n".join(allcode)
        d = {"ConstInt": ConstInt, "ConstPtr": ConstPtr, "ConstFloat": ConstFloat, "JitCode": JitCode, "ChangeFrame": ChangeFrame,
             "lltype": lltype, "rstr": rstr, 'llmemory': llmemory, 'OBJECTPTR': OBJECTPTR, 'support': support}
        d.update(self.globals)
        source = py.code.Source(self.jitcode._genext_source)
        exec source.compile() in d
        print "_____"
        print self.jitcode.dump()
        print "_____"
        print self.jitcode._genext_source
        self.jitcode.genext_function = d['jit_shortcut']
        self.jitcode.genext_function.__name__ += "_" + self.jitcode.name

    def _make_code(self, index, insn):
            self._reset_insn()
            assert not (isinstance(insn[0], Label) or insn[0] == '---')
            self.insn = insn
            pc = self.ssarepr._insns_pos[index]
            nextpc = self.pc_to_nextpc[pc]
            instruction = self.insns[ord(self.jitcode.code[pc])]
            self.name, self.argcodes = instruction.split("/")
            self.methodname = 'opimpl_' + self.name
            lines, needed_orgpc, needed_label = self._parse_args(index, pc, nextpc)
            for line in lines:
                self.code.append("    " + line)
            meth = getattr(self, "emit_" + self.name, self.emit_default)
            lines = meth()
            for line in lines:
                self.code.append("    " + line)
            pcs = self.next_possible_pcs(insn, needed_label, nextpc)
            if len(pcs) == 0:
                self.code.append("    assert 0 # unreachable")
                return
            elif len(pcs) == 1:
                next_insn = self.pc_to_insn[pcs[0]]
                goto_target = self._find_actual_jump_target_chain(next_insn, pcs[0])
                self.code.append("    pc = %s" % goto_target)
            else:
                self.code.append("    pc = self.pc")
                # do the trick
                prefix = ''
                for pc in pcs:
                    next_insn = self.pc_to_insn[pc]
                    goto_target = self._find_actual_jump_target(next_insn, pc)
                    self.code.append("    %sif pc == %s: pc = %s" % (prefix, pc, goto_target))
                    prefix = "el"
                self.code.append("    else:")
                self.code.append("        assert 0 # unreachable")
            self.code.append("    continue")

    def _add_global(self, obj):
        name = "glob%s" % len(self.globals)
        self.globals[name] = obj
        return name

    def _decode_label(self, position):
        code = self.jitcode.code
        needed_label = ord(code[position]) | (ord(code[position+1])<<8)
        return needed_label

    def _find_actual_jump_target(self, next_insn, targetpc):
        if next_insn[0] == 'goto':
            return self._decode_label(targetpc+1)
        elif next_insn[0] == '-live-':
            return self.pc_to_nextpc[targetpc]
        else:
            # otherwise, just return pc
            return targetpc

    def _find_actual_jump_target_chain(self, next_insn, targetpc):
        insn = next_insn[0]
        while True:
            if insn == 'goto':
                targetpc = self._decode_label(targetpc+1)
            elif insn == '-live-':
                targetpc = self.pc_to_nextpc[targetpc]
            else:
                break
            insn = self.pc_to_insn[targetpc][0]
        return targetpc

    def _parse_args(self, index, pc, nextpc):
        from rpython.jit.metainterp.pyjitpl import MIFrame
        from rpython.jit.metainterp.blackhole import signedord
        lines = []

        unboundmethod = getattr(MIFrame, self.methodname).im_func
        argtypes = unboundmethod.argtypes

        # collect arguments, this is a 'timeshifted' version of the code in
        # pyjitpl._get_opimpl_method
        args = []
        args_as_objects = []
        next_argcode = 0
        code = self.jitcode.code
        orgpc = pc
        position = pc
        position += 1
        needed_orgpc = False
        needed_label = None
        for argtype in argtypes:
            arg_as_object = None
            if argtype == "box":     # a box, of whatever type
                argcode = self.argcodes[next_argcode]
                next_argcode = next_argcode + 1
                if argcode == 'i':
                    value = "self.registers_i[%s]" % (ord(code[position]), )
                elif argcode == 'c':
                    value = "ConstInt(%s)" % signedord(code[position])
                elif argcode == 'r':
                    value = "self.registers_r[%s]" % (ord(code[position]), )
                elif argcode == 'f':
                    value = "self.registers_f[%s]" % (ord(code[position]), )
                else:
                    raise AssertionError("bad argcode")
                position += 1
            elif argtype == "descr" or argtype == "jitcode":
                assert self.argcodes[next_argcode] == 'd'
                next_argcode = next_argcode + 1
                index = ord(code[position]) | (ord(code[position+1])<<8)
                arg_as_object = self.assembler.descrs[index]
                value = self._add_global(arg_as_object)
                if argtype == "jitcode":
                    self.code.append("    assert isinstance(%s, JitCode)" % value)
                position += 2
            elif argtype == "label":
                assert self.argcodes[next_argcode] == 'L'
                next_argcode = next_argcode + 1
                assert needed_label is None # only one label per instruction
                needed_label = self._decode_label(position)
                value = str(needed_label)
                position += 2
            elif argtype == "boxes":     # a list of boxes of some type
                length = ord(code[position])
                value = [None] * length
                self.prepare_list_of_boxes(value, 0, position,
                                           self.argcodes[next_argcode])
                next_argcode = next_argcode + 1
                position += 1 + length
                value = '[' + ",".join(value) + "]"
            elif argtype == "boxes2":     # two lists of boxes merged into one
                length1 = ord(code[position])
                position2 = position + 1 + length1
                length2 = ord(code[position2])
                value = [None] * (length1 + length2)
                self.prepare_list_of_boxes(value, 0, position,
                                           self.argcodes[next_argcode])
                self.prepare_list_of_boxes(value, length1, position2,
                                           self.argcodes[next_argcode + 1])
                next_argcode = next_argcode + 2
                position = position2 + 1 + length2
                value = '[' + ",".join(value) + "]"
            elif argtype == "boxes3":    # three lists of boxes merged into one
                length1 = ord(code[position])
                position2 = position + 1 + length1
                length2 = ord(code[position2])
                position3 = position2 + 1 + length2
                length3 = ord(code[position3])
                value = [None] * (length1 + length2 + length3)
                self.prepare_list_of_boxes(value, 0, position,
                                           self.argcodes[next_argcode])
                self.prepare_list_of_boxes(value, length1, position2,
                                           self.argcodes[next_argcode + 1])
                self.prepare_list_of_boxes(value, length1 + length2, position3,
                                           self.argcodes[next_argcode + 2])
                next_argcode = next_argcode + 3
                position = position3 + 1 + length3
                value = '[' + ",".join(value) + "]"
            elif argtype == "newframe" or argtype == "newframe2" or argtype == "newframe3":
                assert argtypes == (argtype, )
                # this and the next two are basically equivalent to
                # jitcode boxes/boxes2/boxes3
                # instead of allocating the list of boxes, just put everything
                # into the correct position of a new MIFrame

                # first get the jitcode
                assert self.argcodes[next_argcode] == 'd'
                next_argcode = next_argcode + 1
                index = ord(code[position]) | (ord(code[position+1])<<8)
                value = argname = "arg%s" % position
                jitcode = self._add_global(self.assembler.descrs[index])
                lines.append("assert isinstance(%s, JitCode)" % jitcode)
                position += 2
                # make a new frame
                lines.append("%s = self.metainterp.newframe(%s)" % (argname, jitcode))
                lines.append("%s.pc = 0" % (argname, ))

                # generate code to put boxes into the right places
                length = ord(code[position])
                self.fill_registers(lines, argname, length, position + 1,
                                    self.argcodes[next_argcode])
                next_argcode = next_argcode + 1
                position += 1 + length
                if argtype != "newframe": # 2/3 lists of boxes
                    length = ord(code[position])
                    self.fill_registers(lines, argname, length, position + 1,
                                        self.argcodes[next_argcode])
                    next_argcode = next_argcode + 1
                    position += 1 + length
                if argtype == "newframe3": # 3 lists of boxes
                    length = ord(code[position])
                    self.fill_registers(lines, argname, length, position + 1,
                                        self.argcodes[next_argcode])
                    next_argcode = next_argcode + 1
                    position += 1 + length
            elif argtype == "orgpc":
                value = str(orgpc)
                needed_orgpc = True
            elif argtype == "int":
                argcode = self.argcodes[next_argcode]
                next_argcode = next_argcode + 1
                if argcode == 'i':
                    pos = ord(code[position])
                    num_regs_i = self.jitcode.num_regs_i()
                    value = "self.registers_i[%s].getint()" % (pos, )
                    if pos >= num_regs_i:
                        intval = self.jitcode.constants_i[pos - num_regs_i]
                        if isinstance(intval, int):
                            value = str(intval)
                elif argcode == 'c':
                    value = str(signedord(code[position]))
                else:
                    raise AssertionError("bad argcode")
                position += 1
            elif argtype == "jitcode_position":
                value = str(position)
            else:
                raise AssertionError("bad argtype: %r" % (argtype,))
            args.append(value)
            args_as_objects.append(arg_as_object)
        num_return_args = len(self.argcodes) - next_argcode
        assert num_return_args == 0 or num_return_args == 2
        if num_return_args:
            returncode = self.argcodes[next_argcode + 1]
            resindex = ord(code[position])
        else:
            returncode = 'v'
            resindex = -1
        self.args = args
        self.args_as_objects = args_as_objects
        self.returncode = returncode
        self.resindex = resindex
        return lines, needed_orgpc, needed_label

    def emit_newframe_function(self):
        return ["self._result_argcode = %r" % (self.returncode, ), "return # change frame"]
    emit_inline_call_r_i = emit_newframe_function
    emit_inline_call_r_r = emit_newframe_function
    emit_inline_call_r_v = emit_newframe_function
    emit_inline_call_ir_i = emit_newframe_function
    emit_inline_call_ir_r = emit_newframe_function
    emit_inline_call_ir_v = emit_newframe_function
    emit_inline_call_irf_i = emit_newframe_function
    emit_inline_call_irf_r = emit_newframe_function
    emit_inline_call_irf_f = emit_newframe_function
    emit_inline_call_irf_v = emit_newframe_function

    def emit_default(self):
        lines = []
        strargs = ", ".join(self.args)
        if self.returncode != 'v':
            # Save the type of the resulting box.  This is needed if there is
            # a get_list_of_active_boxes().  See comments there.
            lines.append("self._result_argcode = %r" % (self.returncode, ))
            if self.returncode == "i":
                prefix = "self.registers_i[%s] = " % self.resindex
            elif self.returncode == "r":
                prefix = "self.registers_r[%s] = " % self.resindex
            elif self.returncode == "f":
                prefix = "self.registers_f[%s] = " % self.resindex
            else:
                assert 0
        else:
            lines.append("self._result_argcode = 'v'")
            prefix = ''

        lines.append("%sself.%s(%s)" % (prefix, self.methodname, strargs))
        return lines

    def emit_return(self):
        lines = []
        lines.append("try:")
        lines.append("    self.%s(%s)" % (self.methodname, self.args[0]))
        lines.append("except ChangeFrame: return")
        return lines

    emit_int_return = emit_return
    emit_ref_return = emit_return
    emit_float_return = emit_return

    def prepare_list_of_boxes(self, outvalue, startindex, position, argcode):
        assert argcode in 'IRF'
        code = self.jitcode.code
        length = ord(code[position])
        position += 1
        for i in range(length):
            index = ord(code[position+i])
            if   argcode == 'I': reg = "self.registers_i[%s]" % index
            elif argcode == 'R': reg = "self.registers_r[%s]" % index
            elif argcode == 'F': reg = "self.registers_f[%s]" % index
            else: raise AssertionError(argcode)
            outvalue[startindex+i] = reg

    def fill_registers(self, lines, argname, length, position, argcode):
        assert argcode in 'IRF'
        code = self.jitcode.code
        for i in range(length):
            index = ord(code[position+i])
            if   argcode == 'I':
                lines.append("%s.registers_i[%s] = self.registers_i[%s]" % (argname, i, index))
            elif argcode == 'R':
                lines.append("%s.registers_r[%s] = self.registers_r[%s]" % (argname, i, index))
            elif argcode == 'F':
                lines.append("%s.registers_f[%s] = self.registers_f[%s]" % (argname, i, index))
            else:
                raise AssertionError(argcode)

    def next_possible_pcs(self, insn, needed_label, nextpc):
        if insn[0] == "goto":
            return [needed_label]
        if needed_label is not None:
            return [nextpc, needed_label]
        if insn[0].endswith("return"):
            return []
        if insn[0].endswith("raise"):
            return []
        if insn[0] == "switch":
            return insn[2].dict.values() + [nextpc]
        else:
            return [nextpc]


class WorkList(object):

    OFFSET = 100

    def __init__(self, pc_to_insn=None, label_to_pc=None, pc_to_nextpc=None, globals=None):
        self.max_used_pc = 0
        if pc_to_insn is None:
            pc_to_insn = dict()
        if len(pc_to_insn) > 0:
            self.max_used_pc = max(pc_to_insn)
        if pc_to_nextpc is None:
            pc_to_nextpc = {}
        self.pc_to_nextpc = pc_to_nextpc
        self.orig_pc_to_insn = pc_to_insn
        self.specialize_instruction = dict() # (pc, insn, constant?registers) =? Specializer
        self.todo = []
        self.free_pc = self.max_used_pc + self.OFFSET
        self.label_to_pc = {}
        if label_to_pc is not None:
            self.label_to_pc.update(label_to_pc)
        if globals is not None:
            self.globals = globals
        else:
            self.globals = {}

    def _make_spec(self, insn, constant_registers, orig_pc):
        assert self.orig_pc_to_insn[orig_pc] == insn
        constant_registers = frozenset(val for val in constant_registers if not isinstance(val, Constant))
        constant_registers = self._remove_dead_const_registers(insn, constant_registers, orig_pc)
        key = (orig_pc, insn, frozenset(constant_registers))
        if key in self.specialize_instruction:
            return self.specialize_instruction[key]
        else:
            if not constant_registers:
                spec_pc = orig_pc
            else:
                spec_pc = self.payout_new_free_pc()
            spec = self.specialize_instruction[key] = Specializer(
                insn, constant_registers, orig_pc, spec_pc, self)
            self.todo.append(spec)
            return spec

    def _remove_dead_const_registers(self, insn, constant_registers, orig_pc):
        if insn[0] == '-live-':
            constant_registers = frozenset([var for var in constant_registers if var in insn])
        return constant_registers

    def _shortcut_live_and_goto(self, insn, constant_registers, orig_pc):
        while insn[0] in ('-live-', 'goto'):
            if insn[0] == '-live-':
                constant_registers = frozenset([var for var in constant_registers if var in insn])
                orig_pc = self.pc_to_nextpc[orig_pc]
            else:
                assert insn[0] == 'goto'
                orig_pc = self.label_to_pc[insn[1].name]
            insn = self.orig_pc_to_insn[orig_pc]
        return insn, constant_registers, orig_pc

    def payout_new_free_pc(self):
        free_pc = self.free_pc
        self.free_pc += 1
        return free_pc

    def specialize_insn(self, insn, constant_registers, orig_pc, label=None):
        return self._make_spec(insn, constant_registers, orig_pc)

    def specialize_pc(self, constant_registers, orig_pc):
        return self._make_spec(self.orig_pc_to_insn[orig_pc], constant_registers, orig_pc)


class Specializer(object):
    def __init__(self, insn, constant_registers, orig_pc, spec_pc, work_list):
        self.insn = insn
        self.constant_registers = constant_registers
        self.orig_pc = orig_pc
        self.spec_pc = spec_pc
        if not constant_registers: # not specialized
            assert orig_pc == spec_pc
        self.work_list = work_list

        self.name = self.insn[0]
        self.methodname = "opimpl_" + self.name
        self.resindex = len(self.insn) - 1 if '->' in self.insn else None
        self.tempvarindex = 0

    def _reset_specializer(self):
        self.name = None
        self.methoname = None
        self.resindex = None
        self.tempvarindex = 0

    def _add_global(self, obj):
        name = "glob%s" % len(self.work_list.globals)
        self.work_list.globals[name] = obj
        return name

    def _get_args(self):
        if self.resindex:
            return self.insn[1:-2]
        else:
            return self.insn[1:]

    def get_pc(self):
        return self.spec_pc

    def get_target_pc(self, label):
        return self.work_list.label_to_pc[label.name]

    def is_constant(self, arg):
        return arg in self.constant_registers

    def make_code(self):
        args = self._get_args()
        try:
            if not self._check_all_constant_args(args):
                return self._make_code_unspecialized()
            return self._make_code_specialized()
        except Unsupported:
            return None

    def _is_label(self, arg):
        return isinstance(arg, Label) or isinstance(arg, TLabel)

    def _check_all_constant_args(self, args):
        for arg in args:
            if (
                    arg not in self.constant_registers and
                    not isinstance(arg, Constant) and
                    not self._is_label(arg) and
                    not isinstance(arg, AbstractDescr)
            ):
                return False
        return True

    def _make_code_specialized(self):
        meth = getattr(self, "emit_specialized_" + self.name.strip('-'), None)
        if meth is not None:
            return '\n'.join(meth())
        return None

    def _make_code_unspecialized(self):
        meth = getattr(self, "emit_unspecialized_" + self.name.strip('-'), None)
        if meth is not None:
            return '\n'.join(meth())
        return None

    def get_next_constant_registers(self):
        if not self.resindex:
            return self.constant_registers

        args = self._get_args()
        if not self._check_all_constant_args(args):
            return self.constant_registers - {self.insn[self.resindex]}
        return self.constant_registers.union({self.insn[self.resindex]})

    def _get_new_temp_variable(self):
        i = self.tempvarindex
        self.tempvarindex += 1
        return "v%d" %i

    def emit_specialized_int_add(self):
        return self._emit_specialized_int_binary("+")

    def emit_specialized_int_mul(self):
        return self._emit_specialized_int_binary("*")

    def emit_specialized_int_or(self):
        return self._emit_specialized_int_binary("|")

    def emit_specialized_int_sub(self):
        return self._emit_specialized_int_binary("-")

    def _emit_specialized_int_binary(self, op):
        args = self._get_args()
        assert len(args) == 2
        arg0, arg1 = args[0], args[1]
        result = self.insn[self.resindex]
        lines = ["i%s = %s %s %s" % (result.index, self._get_as_unboxed(arg0),
                                     op, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def _emit_jump(self, lines, target_pc=-1, constant_registers=None, indent=''):
        if target_pc == -1:
            target_pc = self.work_list.pc_to_nextpc[self.orig_pc]
        if constant_registers is None:
            constant_registers = self.get_next_constant_registers()
        insn = self.work_list.orig_pc_to_insn[target_pc]
        insn, constant_registers, target_pc = self.work_list._shortcut_live_and_goto(
                insn, constant_registers, target_pc)

        spec_next = self.work_list.specialize_pc(
                constant_registers, target_pc)
        lines.append("%spc = %s" % (indent, spec_next.spec_pc))
        lines.append(indent + "continue")

    def emit_specialized_strgetitem(self):
        args = self._get_args()
        assert len(args) == 2
        arg0, arg1 = args[0], args[1]
        result = self.insn[self.resindex]
        lines = ["i%s = ord(lltype.cast_opaque_ptr(lltype.Ptr(rstr.STR), r%d).chars[%s])" % (
            result.index, arg0.index, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def emit_specialized_int_guard_value(self):
        lines = ['# guard_value, argument is already constant']
        self._emit_jump(lines)
        return lines
    emit_specialized_ref_guard_value = emit_specialized_int_guard_value

    def emit_specialized_guard_class(self):
        lines = ['# guard_class, argument is already constant']
        arg, = self._get_args()
        res = self.insn[self.resindex]
        lines.append('i%s = support.ptr2int(lltype.cast_opaque_ptr(OBJECTPTR, r%s).typeptr)' % (res.index, arg.index))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines

    def emit_specialized_getfield_raw_i(self):
        if self.insn[2].is_always_pure():
            lines = []
            arg, descr = self._get_args()
            res = self.insn[self.resindex]
            PTRTYPE, name = _get_ptrtype_fieldname_from_fielddescr(descr)
            resultcast = _find_result_cast(PTRTYPE, name)
            lines.append('%s = %sllmemory.cast_adr_to_ptr(support.int2adr(i%s), %s).%s)' % (self._get_as_unboxed(res), resultcast, arg.index, self._add_global(PTRTYPE), name))
            self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
            return lines
        raise Unsupported

    def emit_specialized_getfield_gc_i_pure(self):
        if self.insn[2].is_always_pure():
            lines = []
            arg, descr = self._get_args()
            res = self.insn[self.resindex]
            PTRTYPE, name = _get_ptrtype_fieldname_from_fielddescr(descr)
            resultcast = _find_result_cast(PTRTYPE, name)
            lines.append('%s = %slltype.cast_opaque_ptr(%s, %s).%s)' % (
                self._get_as_unboxed(res), resultcast, self._add_global(PTRTYPE), self._get_as_unboxed(arg), name))
            self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
            return lines
        raise Unsupported
    emit_specialized_getfield_gc_r_pure = emit_specialized_getfield_gc_i_pure

    def emit_specialized_int_copy(self):
        arg0, = self._get_args()
        res = self.insn[self.resindex]
        lines = ["%s = %s" % (self._get_as_unboxed(res), self._get_as_unboxed(arg0))]
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines
    emit_specialized_ref_copy = emit_specialized_int_copy

    def emit_specialized_int_between(self):
        arg0, arg1, arg2 = self._get_args()
        lines = []
        tempvar = self._get_new_temp_variable()
        result = self.insn[self.resindex]
        lines.append('i%s = %s <= %s < %s' % (
            result.index,
            self._get_as_unboxed(arg0),
            self._get_as_unboxed(arg1),
            self._get_as_unboxed(arg2)))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({result}))
        return lines

    def emit_specialized_instance_ptr_eq(self):
        arg0, arg1 = self._get_args()
        lines = []
        result = self.insn[self.resindex]
        lines.append('i%s = %s is %s' % (
            result.index,
            self._get_as_unboxed(arg0),
            self._get_as_unboxed(arg1),
        ))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({result}))
        return lines

    def emit_specialized_goto(self):
        label, = self._get_args()
        label_pc = self.get_target_pc(label)
        lines = []
        self._emit_jump(lines, label_pc)
        return lines

    def emit_specialized_goto_if_not_absolute(self, name, symbol_fmt):
        if symbol_fmt == '':
            symbol_fmt == '%s'
        elif '%s' not in symbol_fmt:
            assert 0, "expected a valid format string for symbol_fmt"
        lines = []
        arg, label = self._get_args()
        unboxed_arg = self._get_as_unboxed(arg)
        operation = symbol_fmt % (unboxed_arg, )
        lines.append("cond = %s" % (operation,))
        lines.append("if not cond:")
        label_pc = self.get_target_pc(label)
        target_spec = self.work_list.specialize_pc(self.constant_registers, label_pc)
        lines.append("    pc = %d" % (target_spec.spec_pc,))
        lines.append("    continue")
        self._emit_jump(lines)
        return lines

    def emit_specialized_goto_if_not_int_is_true(self):
        return self.emit_specialized_goto_if_not_absolute('int_is_true', '%s != 0')

    def emit_specialized_goto_if_not_int_is_zero(self):
        return self.emit_specialized_goto_if_not_absolute('int_is_zero', '%s == 0')

    def emit_specialized_goto_if_not_ptr_nonzero(self):
        return self.emit_specialized_goto_if_not_absolute('ptr_nonzero', '%s')

    def emit_specialized_goto_if_not_ptr_zero(self):
        return self.emit_specialized_goto_if_not_absolute('ptr_zero', 'not %s')

    def emit_specialized_goto_if_not(self):
        return self.emit_specialized_goto_if_not_absolute('', '%s')

    def emit_specialized_goto_if_not_int_comparison(self, name, symbol):
        lines = []
        arg0, arg1, label = self._get_args()
        lines.append("cond = %s %s %s" % (self._get_as_unboxed(arg0), symbol, self._get_as_unboxed(arg1)))
        lines.append("if not cond:")
        label_pc = self.get_target_pc(label)
        target_spec = self.work_list.specialize_pc(self.constant_registers, label_pc)
        lines.append("    pc = %d" % (target_spec.spec_pc,))
        lines.append("    continue")
        self._emit_jump(lines)
        return lines

    def emit_specialized_goto_if_not_int_lt(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_lt', '<')

    def emit_specialized_goto_if_not_int_gt(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_gt', '>')

    def emit_specialized_goto_if_not_int_ge(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_ge', '>=')

    def emit_specialized_goto_if_not_int_le(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_le', '<=')

    def emit_specialized_goto_if_not_int_ne(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_ne', '!=')

    def emit_specialized_goto_if_not_int_eq(self):
        return self.emit_specialized_goto_if_not_int_comparison('int_eq', '==')

    def emit_specialized_switch(self):
        lines = []
        arg = self.insn[1]
        descr = self.insn[2]
        switchdict = descr.dict

        prefix = ''
        for val in switchdict:
            lines.append('%sif %s%d == %d:' % (prefix, self._get_type_prefix(arg), arg.index, val))
            target_pc = switchdict[val]
            self._emit_jump(lines, target_pc=target_pc, indent='    ')
            prefix = 'el'
        self._emit_jump(lines)
        return lines

    def emit_specialized_unreachable(self):
        return ["assert 0, 'unreachable'"]
    emit_unspecialized_unreachable = emit_specialized_unreachable

    def emit_specialized_int_return(self):
        return self.emit_unspecialized_int_return()

    def _get_type_prefix(self, arg):
        if isinstance(arg, Constant) or isinstance(arg, Register):
            # TODO: this logic also works for the 'else' case. probably.
            if isinstance(arg, Constant):
                kind = getkind(arg.concretetype)
            else:
                kind = arg.kind
            assert kind in ('int', 'ref')
            return kind[0]
        else:
            m = re.search('%([i,r,f])[0-9]+', str(arg))
            assert m is not None, "ensure regex match"
            return m.group(1)

    def _get_as_unboxed(self, arg):
        if isinstance(arg, Constant):
            kind = getkind(arg.concretetype)
            if kind == 'int':
                TYPE = arg.concretetype
                if isinstance(TYPE, lltype.Ptr):
                    assert TYPE.TO._gckind == 'raw'
                    return "support.ptr2int(%s)" % (self._add_global(arg.value), )
                val = lltype.cast_primitive(lltype.Signed, arg.value)
                if not isinstance(val, int):
                    return self._add_global(arg.value)
                return str(val)
            raise Unsupported
        else:
            t = self._get_type_prefix(arg)
            return "%s%s" % (t, arg.index)

    def _get_as_box(self, arg):
        if isinstance(arg, Constant):
            kind = getkind(arg.concretetype)
            if kind == 'int':
                TYPE = arg.concretetype
                if isinstance(TYPE, lltype.Ptr):
                    assert TYPE.TO._gckind == 'raw'
                    return "ConstInt(support.ptr2int(%s))" % (self._add_global(arg.value), )
                val = lltype.cast_primitive(lltype.Signed, arg.value)
                if not isinstance(val, int):
                    return "ConstInt(%s)" % self._add_global(arg.value)
                return "ConstInt(%d)" % val
            elif kind == 'ref':
                return "ConstPtr(%d)" % arg.value
            else:
                assert False
        elif arg in self.constant_registers:
            if arg.kind == 'int':
                return "ConstInt(i%d)" % arg.index
            elif arg.kind == 'ref':
                return "ConstPtr(r%d)" % arg.index
            else:
                assert False
        else:
            t = self._get_type_prefix(arg)
            return "r%s%d" % (t, arg.index)

    def _emit_unbox_by_type(self, arg, lines, indent=''):
        t = self._get_type_prefix(arg)
        line = ''
        if t == 'i':
            line = "i%d = ri%d.getint()" % (arg.index, arg.index,)
        elif t == 'r':
            line = "r%d = rr%d.getref_base()" % (arg.index, arg.index,)
        elif t == 'f':
            line = "f%d = rf%d.getfloat()" % (arg.index, arg.index,)
        else:
            assert False, "%s is unsupported type" % (arg)
        lines.append(indent + line)

    def _emit_box_by_type(self, arg, lines, indent=''):
        t = self._get_type_prefix(arg)
        line = ''
        if t == 'i':
            line = "ri%d = self.registers_i[%d]" % (arg.index, arg.index)
        elif t == 'r':
            line = "rr%d = self.registers_r[%d]" % (arg.index, arg.index)
        elif t == 'f':
            line = "rf%d = self.registers_f[%d]" % (arg.index, arg.index)
        else:
            assert False, "%s is unsupported type" % (arg)
        lines.append(indent + line)

    def _emit_assignment_return_const_check(self, arg, lines):
        if isinstance(arg, Constant):
            return None
        if arg in self.constant_registers:
            return None
        t = self._get_type_prefix(arg)
        if t in 'irf':
            lines.append("r%s%d = self.registers_%s[%d]" % (t, arg.index, t, arg.index))
        else:
            assert False, "%s is unsupported type" % (arg)
        if t == 'i':
            cls = 'ConstInt'
        elif t == 'r':
            cls = 'ConstPtr'
        else:
            cls = 'ConstFloat'
        return "isinstance(r%s%s, %s)" % (t, arg.index, cls)

    def _emit_unary_if(self, arg, lines):
        check = self._emit_assignment_return_const_check(arg, lines)
        assert check is not None
        lines.append("if %s:" % (check, ))
        self._emit_unbox_by_type(arg, lines, '    ')

    def _emit_binary_if(self, arg0, arg1, lines):
        check0 = self._emit_assignment_return_const_check(arg0, lines)
        check1 = self._emit_assignment_return_const_check(arg1, lines)
        assert check0 is not None or check1 is not None
        if check0 is None:
            cond = check1
        elif check1 is None:
            cond = check0
        else:
            cond = "%s and %s" % (check0, check1)
        lines.append("if %s:" % (cond, ))
        if check0 is not None:
            self._emit_unbox_by_type(arg0, lines, '    ')
        if check1 is not None:
            self._emit_unbox_by_type(arg1, lines, '    ')

    def _emit_n_ary_if(self, args, lines):
        args_and_checks = []
        at_least_one_not_none = False
        for arg in args:
            check = self._emit_assignment_return_const_check(arg, lines)
            if check is not None:
                at_least_one_not_none = True
            args_and_checks.append((arg, check))
        assert at_least_one_not_none
        condition = ' and '.join([ac[1] for ac in args_and_checks if ac[1] is not None])
        lines.append('if %s:' % condition)
        for arg, check in args_and_checks:
            if check is not None:
                self._emit_unbox_by_type(arg, lines, '    ')

    def _emit_unspecialized_binary(self):
        lines = []
        args = self._get_args()
        assert len(args) == 2
        arg0, arg1 = args[0], args[1]
        result = self.insn[self.resindex]
        self._emit_binary_if(arg0, arg1, lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0, arg1}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc()))
        lines.append("    continue")
        lines.append("else:")
        lines.append("    self.registers_i[%d] = self.%s(%s, %s)" % (
            result.index, self.methodname,
            self._get_as_box(arg0), self._get_as_box(arg1)))
        self._emit_jump(lines)
        return lines

    emit_unspecialized_int_add = _emit_unspecialized_binary
    emit_unspecialized_int_sub = _emit_unspecialized_binary
    emit_unspecialized_int_mul = _emit_unspecialized_binary
    emit_unspecialized_int_or = _emit_unspecialized_binary

    def emit_unspecialized_strgetitem(self):
        lines = []
        arg0, arg1 = self.insn[1], self.insn[2]
        result = self.insn[self.resindex]
        self._emit_binary_if(arg0, arg1, lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0, arg1}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc()))
        lines.append("    continue")
        lines.append("else:")
        lines.append("    self.registers_i[%d] = self.opimpl_strgetitem(%s, %s)" % (
            result.index, self._get_as_box(arg0), self._get_as_box(arg1)))
        self._emit_jump(lines)
        return lines

    def emit_unspecialized_guard_value(self):
        lines = []
        arg0 = self.insn[1]

        cond = self._emit_assignment_return_const_check(arg0, lines)
        assert cond is not None
        lines.append('if %s:' % cond)
        self._emit_unbox_by_type(arg0, lines, indent='    ')
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0}), self.orig_pc)
        lines.append('    pc = %d' % specializer.get_pc())
        lines.append('    continue')

        self._emit_sync_registers(lines)
        lines.append('self.opimpl_%s(%s, %d)' % (self.insn[0], self._get_as_box(arg0), self.orig_pc))
        self._emit_box_by_type(arg0, lines)
        self._emit_unbox_by_type(arg0, lines)
        self._emit_jump(lines, constant_registers=self.constant_registers.union({arg0}))
        return lines

    emit_unspecialized_int_guard_value = emit_unspecialized_guard_value
    emit_unspecialized_ref_guard_value = emit_unspecialized_guard_value

    def emit_unspecialized_guard_class(self):
        arg0 = self.insn[1]
        res = self.insn[self.resindex]
        lines = []
        self._emit_box_by_type(arg0, lines)
        box = self._get_as_box(arg0)
        lines.append('if self.metainterp.heapcache.is_class_known(%s):' % box)

        lines.append('    i%d = self.cls_of_box(%s).getint()' % (res.index, box, ))
        specializer = self.work_list.specialize_pc(
            self.constant_registers.union({res}), self.work_list.pc_to_nextpc[self.orig_pc])
        lines.append('    pc = %d' % specializer.get_pc())
        lines.append('    continue')

        self._emit_sync_registers(lines)
        lines.append('i%s = self.opimpl_%s(%s, %d).getint()' % (res.index, self.insn[0], self._get_as_box(arg0), self.orig_pc))
        lines.append('pc = %d' % specializer.get_pc())
        lines.append('continue')
        return lines

    def emit_unspecialized_int_copy(self):
        arg0, = self._get_args()
        res = self.insn[self.resindex]
        lines = []
        cond = self._emit_assignment_return_const_check(arg0, lines)
        assert cond is not None
        lines.append("self.registers_%s[%s] = %s" % (res.kind[0], res.index, self._get_as_box(arg0)))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_ref_copy = emit_unspecialized_int_copy

    def emit_unspecialized_int_between(self):
        args = self._get_args()
        res = self.insn[self.resindex]
        lines = []
        # try to figure out every register is constant
        self._emit_n_ary_if(args, lines)
        # if all registers are constant, let the control to the specialized path
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union(set(args)), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc(), ))
        lines.append("    continue")
        result = self.insn[self.resindex]
        lines.append("self.registers_i[%s] = self.opimpl_int_between(%s, %s, %s)" % (
            result.index,
            self._get_as_box(args[0]), self._get_as_box(args[1]), self._get_as_box(args[2])
        ))
        self._emit_jump(lines)
        return lines

    def emit_unspecialized_instance_ptr_eq(self):
        args = self._get_args()
        res = self.insn[self.resindex]
        lines = []
        # try to figure out every register is constant
        self._emit_n_ary_if(args, lines)
        # if all registers are constant, let the control to the specialized path
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union(set(args)), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc(), ))
        lines.append("    continue")
        result = self.insn[self.resindex]
        lines.append("self.registers_i[%s] = self.opimpl_instance_ptr_eq(%s, %s)" % (
            result.index, self._get_as_box(args[0]), self._get_as_box(args[1])
        ))
        self._emit_jump(lines)
        return lines


    def emit_unspecialized_goto_if_not_absolute(self, name):
        lines = []
        _, arg0, arg1 = self.insn # argument, label

        target_pc = self.get_target_pc(arg1)
        self._emit_unary_if(arg0, lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc(), ))
        lines.append("    continue")
        self._emit_sync_registers(lines)
        if name:
            name = "_" + name
        lines.append("self.opimpl_goto_if_not%s(%s, %s, %s)" % \
            (name, self._get_as_box(arg0), target_pc, self.orig_pc))
        lines.append("pc = self.pc")
        lines.append("if pc == %s:" % (target_pc,))
        specializer = self.work_list.specialize_pc(
            self.constant_registers, target_pc)
        lines.append("    pc = %s" % (specializer.spec_pc,))
        lines.append("else:")
        next_pc = self.work_list.pc_to_nextpc[self.orig_pc]
        specializer = self.work_list.specialize_pc(
            self.constant_registers, next_pc)
        lines.append("    assert self.pc == %s" % (specializer.orig_pc,))
        lines.append("    pc = %s" % (specializer.spec_pc,))
        lines.append("continue")
        return lines

    def emit_unspecialized_goto_if_not_int_is_true(self):
        return self.emit_unspecialized_goto_if_not_absolute("int_is_true")

    def emit_unspecialized_goto_if_not_int_is_zero(self):
        return self.emit_unspecialized_goto_if_not_absolute("int_is_zero")

    def emit_unspecialized_goto_if_not_ptr_nonzero(self):
        return self.emit_unspecialized_goto_if_not_absolute("ptr_nonzero")

    def emit_unspecialized_goto_if_not_ptr_zero(self):
        return self.emit_unspecialized_goto_if_not_absolute("ptr_zero")

    def emit_unspecialized_goto_if_not(self):
        return self.emit_unspecialized_goto_if_not_absolute("")

    def emit_unspecialized_goto_if_not_comparison(self, name, symbol):
        lines = []
        _, arg0, arg1, arg2 = self.insn # left, right, label

        target_pc = self.get_target_pc(arg2)
        self._emit_binary_if(arg0, arg1, lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0, arg1}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc(), ))
        lines.append("    continue")
        lines.append("condbox = self.opimpl_%s(%s, %s)" % (name, self._get_as_box(arg0), self._get_as_box(arg1)))
        self._emit_sync_registers(lines)
        lines.append("self.opimpl_goto_if_not(condbox, %d, %d)" % (target_pc, self.orig_pc))
        lines.append("pc = self.pc")
        lines.append("if pc == %s:" % (target_pc,))
        specializer = self.work_list.specialize_pc(
            self.constant_registers, target_pc)
        lines.append("    pc = %s" % (specializer.spec_pc,))
        lines.append("else:")
        next_pc = self.work_list.pc_to_nextpc[self.orig_pc]
        specializer = self.work_list.specialize_pc(
            self.constant_registers, next_pc)
        lines.append("    assert self.pc == %s" % (specializer.orig_pc,))
        lines.append("    pc = %s" % (specializer.spec_pc,))
        lines.append("continue")
        return lines

    def emit_unspecialized_goto_if_not_int_lt(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_lt", "<")

    def emit_unspecialized_goto_if_not_int_gt(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_gt", ">")

    def emit_unspecialized_goto_if_not_int_le(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_le", "<=")

    def emit_unspecialized_goto_if_not_int_ge(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_ge", ">=")

    def emit_unspecialized_goto_if_not_int_ne(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_ne", "!=")

    def emit_unspecialized_goto_if_not_int_eq(self):
        return self.emit_unspecialized_goto_if_not_comparison("int_eq", "==")

    def emit_unspecialized_switch(self):
        lines = []
        arg0, descr = self._get_args()
        name_descr = self._add_global(descr) # add descr to global

        cond = self._emit_assignment_return_const_check(arg0, lines)
        assert cond is not None
        arg0_var = self._get_as_box(arg0)
        lines.append('if %s:' % (cond, ))
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0}), self.orig_pc)
        self._emit_unbox_by_type(arg0, lines, indent='    ')
        lines.append('    pc = %d' % specializer.get_pc())
        lines.append('    continue')
        self._emit_sync_registers(lines)
        lines.append("self.opimpl_switch(%s, %s, %d)" % (arg0_var, name_descr, self.orig_pc))
        lines.append("pc = self.pc")
        # do the trick
        prefix = ''
        for pc in sorted(descr.dict.values()) + [self.work_list.pc_to_nextpc[self.orig_pc]]:
            specializer = self.work_list.specialize_pc(
                self.constant_registers, pc)
            lines.append("%sif pc == %s: pc = %s" % (prefix, pc, specializer.spec_pc))
            prefix = "el"
        lines.append("else: assert 0")
        lines.append("continue")
        return lines

    def emit_unspecialized_return(self):
        lines = []
        value, = self._get_args()
        if not isinstance(value, Constant):
            self._emit_box_by_type(value, lines)
        lines.append("try:")
        lines.append("    self.%s(%s)" % (self.methodname, self._get_as_box(value)))
        lines.append("except ChangeFrame: return")
        return lines
    emit_unspecialized_int_return = emit_unspecialized_return

    def emit_unspecialized_live(self):
        lines = []
        self._emit_jump(lines)
        return lines
    emit_specialized_live = emit_unspecialized_live

    def _emit_sync_registers(self, lines):
        # we need to sync the registers from the unboxed values to e.g. allow a guard to be created
        if not self.constant_registers:
            return
        func, args = _make_register_syncer(self.constant_registers)
        funcname = self._add_global(func)
        lines.append("%s(self, %s) # %s" % (funcname, ", ".join(args), func.func_name))


class Unsupported(Exception):
    pass

def _get_ptrtype_fieldname_from_fielddescr(descr):
    if hasattr(descr, 'S'): # llgraph backend
        return lltype.Ptr(descr.S), descr.fieldname
    return lltype.Ptr(descr.offset.TYPE), descr.offset.fldname

def _find_result_cast(T, field):
    RES = getattr(T.TO, field)
    kind = getkind(RES)
    if kind == 'int':
        if RES == lltype.Signed:
            return '('
        if isinstance(RES, lltype.Primitive):
            return 'lltype.cast_primitive(lltype.Signed, '
        if isinstance(RES, lltype.Ptr):
            assert RES.TO._gckind == 'raw'
            return 'support.ptr2int('
    raise Unsupported

def _make_register_syncer(constant_registers, cache={}):
    key = constant_registers
    if constant_registers in cache:
        return cache[constant_registers]
    constant_registers = sorted(constant_registers, key=lambda reg: (reg.kind, reg.index))
    args = [reg.kind[0] + str(reg.index) for reg in constant_registers]
    name = "jit_sync_regs_" + "_".join(args)
    lines = ["def %s(self, %s):" % (name, ", ".join(args))]
    for reg in constant_registers:
        if reg.kind == 'int':
            val = "ConstInt(i%d)" % reg.index
        elif reg.kind == 'ref':
            val = "ConstPtr(r%d)" % reg.index
        else:
            assert 0
        lines.append('    self.registers_%s[%d] = %s' % (reg.kind[0], reg.index, val))
    source = py.code.Source("\n".join(lines))
    d = {"ConstInt": ConstInt, "ConstPtr": ConstPtr}
    exec source.compile() in d
    res = objectmodel.dont_inline(d[name])
    cache[key] = res, args
    return res, args
