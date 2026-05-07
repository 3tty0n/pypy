import py
import re
import collections
from rpython.jit.metainterp.history import (Const, ConstInt, ConstPtr,
    ConstFloat, CONST_NULL, getkind, AbstractDescr)
from rpython.jit.metainterp import support
from rpython.flowspace.model import Constant
from rpython.jit.codewriter.flatten import (
    Register, TLabel, Label, ListOfKind, IndirectCallTargets)
from rpython.jit.codewriter.jitcode import SwitchDictDescr
from rpython.rtyper.lltypesystem import lltype, llmemory, rstr
from rpython.rtyper.rclass import OBJECTPTR
from rpython.rlib import objectmodel
from rpython.rlib.rarithmetic import r_uint
from rpython.jit.metainterp.support import int_signext
from rpython.jit.codewriter.genextprof import (
    SLOWPATH_PROFILE_ENABLED, get_profiler, classify_opcode,
    REASON_NO_UNSPEC_METHOD, REASON_NO_SPEC_METHOD,
    REASON_UNSUPPORTED_SPEC, REASON_UNSUPPORTED_UNSPEC,
    REASON_SPEC_RETURNED_NONE, REASON_UNSPEC_RETURNED_NONE,
    REASON_IS_CALL, REASON_IS_GUARD, REASON_IS_MEMORY_OP,
    REASON_HAS_LABEL_ARG, REASON_IS_LIVE_OP, REASON_NEWFRAME,
    REASON_FAST_PATH
)

HEAPCACHE_SKIP_OPS = frozenset([
    # Integer binary operations
    'int_add', 'int_sub', 'int_mul', 'int_floordiv', 'int_mod',
    'int_and', 'int_or', 'int_xor', 'int_rshift', 'int_lshift',
    'uint_rshift', 'uint_mul_high',
    # Integer comparisons
    'int_lt', 'int_le', 'int_eq', 'int_ne', 'int_gt', 'int_ge',
    'uint_lt', 'uint_le', 'uint_gt', 'uint_ge',
    # Integer unary operations
    'int_neg', 'int_invert', 'int_is_true', 'int_is_zero',
    'int_force_ge_zero', 'int_signext',
    # Float binary operations
    'float_add', 'float_sub', 'float_mul', 'float_truediv',
    # Float comparisons
    'float_lt', 'float_le', 'float_eq', 'float_ne', 'float_gt', 'float_ge',
    # Float unary operations
    'float_neg', 'float_abs',
    # Cast operations (int <-> float)
    'cast_float_to_int', 'cast_int_to_float',
    'cast_float_to_singlefloat', 'cast_singlefloat_to_float',
])

def can_skip_heapcache(opname):
    return opname in HEAPCACHE_SKIP_OPS

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
        for insn in self.ssarepr.insns:
            if isinstance(insn[0], Label) or insn[0] == '---':
                continue
            opname = insn[0]
            if (
                    opname in ('raise', 'reraise') or
                    opname.startswith('inline_call_') or
                    opname.startswith('getarrayitem_vable_') or
                    opname.startswith('setarrayitem_vable_') or
                    opname.startswith('getfield_vable_') or
                    opname.startswith('setfield_vable_')
                ):
                self.jitcode.genext_function = None
                return
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

        # starting points are pc==0, or the instructions after a -live-, or the
        # -live- after a call, or the target of catch_exception calls
        starting_points = {0}
        last_was_live = False
        for pc in sorted(self.assembler.startpoints):
            if last_was_live:
                starting_points.add(pc)
            insn = self.pc_to_insn[pc]
            if insn[0] == 'catch_exception':
                starting_points.add(self.assembler.label_positions[insn[1].name])
            nextpc = self.pc_to_nextpc[pc]
            if nextpc in self.pc_to_insn:
                next_insn = self.pc_to_insn[nextpc]
                if ('call' in insn[0] or 'jit_merge_point' in insn[0]) and next_insn[0] == '-live-':
                    starting_points.add(nextpc)
            last_was_live = insn[0] == '-live-'

        self.work_list = WorkList(self.pc_to_insn, self.assembler.label_positions, self.pc_to_nextpc, self.globals)
        for startpc in self.assembler.startpoints:
            spec = self.work_list.specialize_pc(frozenset(), startpc)
        code_and_spec_per_pc = self.work_list.make_code()
        assert not self.code
        for pc, (code, spec) in code_and_spec_per_pc.iteritems():
            if code is None:
                self.code = []
                if spec.constant_registers:
                    spec._emit_sync_registers(self.code)
                    self.code.append("pc = %s" % spec.orig_pc)
                    self.code.append("continue")
                else:
                    self._make_code(self.pc_to_index[spec.orig_pc], spec.insn)
                code_and_spec_per_pc[pc] = (str(py.code.Source("\n".join(self.code)).deindent()), spec)
        self.code = []
        allconsts = set()
        for pc, (code, spec) in code_and_spec_per_pc.iteritems():
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
            assert name[0] in 'irf'
            if name[0] == 'i':
                default = '0xcafedead'
            elif name[0] == 'r':
                default = 'lltype.nullptr(llmemory.GCREF.TO)'
            else: # float
                default = '0.0'
            self.precode.append("    %s = %s" % (name, default))
        prefix = ""
        for pc in sorted(starting_points):
            self.precode.append("    %sif pc == %s: pc = %s" % (prefix, pc, pc))
            prefix = "el"
        self.precode.append("    else: assert 0, 'unreachable'")
        self.precode.append("    while 1:")
        allcode.extend(self.precode)
        for line in self.code:
            allcode.append(" " * 8 + line)
        self.jitcode._genext_source = "\n".join(allcode)
        # Import rop for opnum constants used in type-specialized recording
        from rpython.jit.metainterp.resoperation import rop
        d = {"ConstInt": ConstInt, "ConstPtr": ConstPtr, "ConstFloat": ConstFloat, "JitCode": JitCode, "ChangeFrame": ChangeFrame,
             "lltype": lltype, "rstr": rstr, 'llmemory': llmemory, 'OBJECTPTR': OBJECTPTR, 'support': support,
             'rop': rop, 'r_uint': r_uint, 'int_signext': int_signext}
        d.update(self.globals)
        source = py.code.Source(self.jitcode._genext_source)
        exec source.compile() in d
        self.jitcode.genext_function = d['jit_shortcut']
        self.jitcode.genext_function.__name__ += "_" + self.jitcode.name
        self._classify_pure_arithmetic()
        self._generate_compile_function()

    def _classify_pure_arithmetic(self):
        """Check if this jitcode is pure arithmetic (no heap, no calls)."""
        from rpython.jit.codewriter.flatten import Label as FLabel
        CONTROL_OPS = frozenset([
            '-live-', 'goto', 'int_return', 'float_return', 'void_return',
            'int_copy', 'float_copy',
        ])
        for insn in self.ssarepr.insns:
            if isinstance(insn[0], FLabel) or insn[0] == '---':
                continue
            opname = insn[0]
            if opname in HEAPCACHE_SKIP_OPS:
                continue
            if opname in CONTROL_OPS:
                continue
            if opname.startswith('goto_if_not_'):
                continue
            # Any non-pure operation: calls, guards, heap ops
            self.jitcode.genext_is_pure_arithmetic = False
            return
        self.jitcode.genext_is_pure_arithmetic = True

    def _generate_compile_function(self):
        if not self.jitcode.genext_is_pure_arithmetic:
            return

        def compile_shortcut(assembler, inputargs, operations):
            from rpython.jit.backend.x86.regloc import (
                RegLoc, ImmedLoc, FrameLoc, eax, ecx, xmm0, xmm1,
                r10, r11, xmm2, xmm3)
            from rpython.jit.backend.x86.arch import (
                JITFRAME_FIXED_SIZE, WORD, IS_X86_64)
            from rpython.jit.metainterp.resoperation import rop
            from rpython.jit.backend.llsupport.assembler import GuardToken
            from rpython.jit.backend.llsupport.gcmap import allocate_gcmap
            from rpython.rlib.rarithmetic import intmask
            from rpython.rlib.longlong2float import float2longlong
            from rpython.jit.metainterp.history import ConstInt, ConstFloat

            frame_map = {}
            frame_pos = [0]
            last_use = {}
            for i, op in enumerate(operations):
                for j in range(op.numargs()):
                    a = op.getarg(j)
                    if a is not None and not isinstance(a, ConstInt) and \
                            not isinstance(a, ConstFloat):
                        last_use[a] = i
                # guard failargs are read by the recovery stub, not op.numargs()
                if rop.is_guard(op.getopnum()):
                    failargs = op.getfailargs() or []
                    for fa in failargs:
                        if fa is None:
                            continue
                        if isinstance(fa, ConstInt) or isinstance(fa, ConstFloat):
                            continue
                        last_use[fa] = i
            cached_int = [None]
            cached_xmm = [None]
            if IS_X86_64:
                int_pool = [r11, r10]
                xmm_pool = [xmm3, xmm2]
            else:
                int_pool = []
                xmm_pool = []
            box_loc = {}
            current_op_index = [0]

            def _new_frame_slot(box):
                pos = frame_pos[0]
                frame_pos[0] = pos + 1
                ebp_offset = (pos + JITFRAME_FIXED_SIZE) * WORD
                loc = FrameLoc(pos, ebp_offset, box.type)
                frame_map[box] = loc
                return loc

            def _get_frame_loc(box):
                if box not in frame_map:
                    return _new_frame_slot(box)
                return frame_map[box]

            base_ofs = assembler.cpu.get_baseofs_of_frame_field()
            initial_locs = []
            for box in inputargs:
                loc = _get_frame_loc(box)
                initial_locs.append(loc.value - base_ofs)
            if assembler.current_clt is not None:
                assembler.current_clt._ll_initial_locs = initial_locs

            def _loc(box):
                if box is None:
                    return None
                if isinstance(box, ConstInt):
                    return ImmedLoc(box.getint())
                if isinstance(box, ConstFloat):
                    return ImmedLoc(intmask(float2longlong(
                        box.getfloatstorage())), is_float=True)
                if cached_int[0] is box:
                    return eax
                if cached_xmm[0] is box:
                    return xmm0
                if box in box_loc:
                    return box_loc[box]
                return _get_frame_loc(box)

            def _store_box_to_home(box, src_reg, is_float):
                if box in box_loc:
                    home = box_loc[box]
                else:
                    if is_float:
                        if xmm_pool:
                            home = xmm_pool.pop()
                        else:
                            home = _get_frame_loc(box)
                    else:
                        if int_pool:
                            home = int_pool.pop()
                        else:
                            home = _get_frame_loc(box)
                    box_loc[box] = home
                if home is src_reg:
                    return
                if is_float:
                    assembler.mc.MOVSD(home, src_reg)
                else:
                    assembler.mc.MOV(home, src_reg)

            def _spill_cached_int():
                box = cached_int[0]
                cached_int[0] = None
                if box is None:
                    return
                if last_use.get(box, -1) < current_op_index[0]:
                    return
                _store_box_to_home(box, eax, is_float=False)

            def _spill_cached_xmm():
                box = cached_xmm[0]
                cached_xmm[0] = None
                if box is None:
                    return
                if last_use.get(box, -1) < current_op_index[0]:
                    return
                _store_box_to_home(box, xmm0, is_float=True)

            def _load_int(box, reg=eax):
                if reg is eax and cached_int[0] is box:
                    return eax
                if reg is eax:
                    _spill_cached_int()
                loc = _loc(box)
                if loc is not reg:
                    assembler.mc.MOV(reg, loc)
                return reg

            def _load_float(box, reg=xmm0):
                if reg is xmm0 and cached_xmm[0] is box:
                    return xmm0
                if reg is xmm0:
                    _spill_cached_xmm()
                loc = _loc(box)
                if loc is not reg:
                    assembler.mc.MOVSD(reg, loc)
                return reg

            def _publish_int_result(op, op_index):
                box = op.result
                if box is None:
                    cached_int[0] = None
                    return
                lu = last_use.get(box, -1)
                if lu <= op_index:
                    cached_int[0] = None
                    return
                if lu == op_index + 1:
                    cached_int[0] = box
                    return
                cached_int[0] = None
                _store_box_to_home(box, eax, is_float=False)

            def _publish_float_result(op, op_index):
                box = op.result
                if box is None:
                    cached_xmm[0] = None
                    return
                lu = last_use.get(box, -1)
                if lu <= op_index:
                    cached_xmm[0] = None
                    return
                if lu == op_index + 1:
                    cached_xmm[0] = box
                    return
                cached_xmm[0] = None
                _store_box_to_home(box, xmm0, is_float=True)

            def _release_dead_pool_regs(op_index):
                dead = []
                for box, home in box_loc.iteritems():
                    if last_use.get(box, -1) > op_index:
                        continue
                    if not isinstance(home, RegLoc):
                        continue
                    if home.is_xmm:
                        xmm_pool.append(home)
                    else:
                        int_pool.append(home)
                    dead.append(box)
                for box in dead:
                    del box_loc[box]

            def _store_int(resloc, reg=eax):
                if resloc is not reg:
                    assembler.mc.MOV(resloc, reg)

            def _store_float(resloc, reg=xmm0):
                if resloc is not reg:
                    assembler.mc.MOVSD(resloc, reg)

            def _flush_caches():
                _spill_cached_int()
                _spill_cached_xmm()

            def _emit_overflow_guard(guard_op):
                _flush_caches()
                faildescr = guard_op.getdescr()
                failargs = guard_op.getfailargs() or []
                fail_locs = []
                for fa in failargs:
                    if fa is None:
                        fail_locs.append(None)
                    else:
                        fail_locs.append(_loc(fa))
                frame_depth = frame_pos[0] + JITFRAME_FIXED_SIZE
                token = assembler.implement_guard_recovery(
                    guard_op.getopnum(), faildescr, failargs,
                    fail_locs, frame_depth)
                assembler.implement_guard(token)

            for op_index, op in enumerate(operations):
                current_op_index[0] = op_index
                opnum = op.getopnum()
                if opnum == rop.LABEL:
                    mc = assembler.mc
                    pos = mc.get_relative_pos()
                    target = (pos + 15) & ~15
                    for _ in range(target - pos):
                        mc.writechar('\x90')
                    op.getdescr()._ll_loop_code = mc.get_relative_pos()
                    assembler.target_tokens_currently_compiling[
                        op.getdescr()] = None
                    assembler.label()
                elif opnum == rop.JUMP:
                    _flush_caches()
                    label_op = operations[0]
                    assert label_op.getopnum() == rop.LABEL
                    label_args = label_op.getarglist()
                    jump_args = op.getarglist()
                    assert len(label_args) == len(jump_args)
                    for i in range(len(jump_args)):
                        src = _loc(jump_args[i])
                        dst = _loc(label_args[i])
                        if src is not dst:
                            if label_args[i].type == 'f':
                                assembler.mc.MOVSD(dst, src)
                            else:
                                assembler.mc.MOV(dst, src)
                    assembler.closing_jump(op.getdescr())
                elif (opnum == rop.GUARD_NO_OVERFLOW or
                        opnum == rop.GUARD_OVERFLOW):
                    _emit_overflow_guard(op)
                elif opnum == rop.INT_ADD:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_add(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_SUB:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_sub(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_MUL:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOV(ecx, arg1)
                        arg1 = ecx
                    assembler.genop_int_mul(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_ADD_OVF:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_add_ovf(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_SUB_OVF:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_sub_ovf(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_MUL_OVF:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOV(ecx, arg1)
                        arg1 = ecx
                    assembler.genop_int_mul_ovf(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_AND:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_and(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_OR:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_or(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_XOR:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_xor(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_LSHIFT:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOV(ecx, arg1)
                        arg1 = ecx
                    assembler.mc.SHL(arg0, arg1)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_RSHIFT:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOV(ecx, arg1)
                        arg1 = ecx
                    assembler.mc.SAR(arg0, arg1)
                    _publish_int_result(op, op_index)
                elif opnum == rop.UINT_RSHIFT:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOV(ecx, arg1)
                        arg1 = ecx
                    assembler.mc.SHR(arg0, arg1)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_NEG:
                    arg0 = _load_int(op.getarg(0))
                    assembler.genop_int_neg(op, [arg0], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_INVERT:
                    arg0 = _load_int(op.getarg(0))
                    assembler.genop_int_invert(op, [arg0], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_FORCE_GE_ZERO:
                    arg0 = _load_int(op.getarg(0))
                    assembler.genop_int_force_ge_zero(op, [arg0], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.INT_SIGNEXT:
                    arg0 = _load_int(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    assembler.genop_int_signext(op, [arg0, arg1], arg0)
                    _publish_int_result(op, op_index)
                elif opnum == rop.FLOAT_ADD:
                    arg0 = _load_float(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOVSD(xmm1, arg1)
                        arg1 = xmm1
                    elif isinstance(arg1, ImmedLoc):
                        raise CannotCompileGenExt(
                            "float constant in binary op")
                    assembler.genop_float_add(op, [arg0, arg1], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.FLOAT_SUB:
                    arg0 = _load_float(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOVSD(xmm1, arg1)
                        arg1 = xmm1
                    elif isinstance(arg1, ImmedLoc):
                        raise CannotCompileGenExt(
                            "float constant in binary op")
                    assembler.genop_float_sub(op, [arg0, arg1], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.FLOAT_MUL:
                    arg0 = _load_float(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOVSD(xmm1, arg1)
                        arg1 = xmm1
                    elif isinstance(arg1, ImmedLoc):
                        raise CannotCompileGenExt(
                            "float constant in binary op")
                    assembler.genop_float_mul(op, [arg0, arg1], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.FLOAT_TRUEDIV:
                    arg0 = _load_float(op.getarg(0))
                    arg1 = _loc(op.getarg(1))
                    if isinstance(arg1, FrameLoc):
                        assembler.mc.MOVSD(xmm1, arg1)
                        arg1 = xmm1
                    elif isinstance(arg1, ImmedLoc):
                        raise CannotCompileGenExt(
                            "float constant in binary op")
                    assembler.genop_float_truediv(op, [arg0, arg1], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.FLOAT_NEG:
                    arg0 = _load_float(op.getarg(0))
                    assembler.genop_float_neg(op, [arg0], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.FLOAT_ABS:
                    arg0 = _load_float(op.getarg(0))
                    assembler.genop_float_abs(op, [arg0], arg0)
                    _publish_float_result(op, op_index)
                elif opnum == rop.CAST_FLOAT_TO_INT:
                    arg0 = _load_float(op.getarg(0))
                    _spill_cached_int()
                    assembler.genop_cast_float_to_int(op, [arg0], eax)
                    _publish_int_result(op, op_index)
                elif opnum == rop.CAST_INT_TO_FLOAT:
                    arg0 = _load_int(op.getarg(0))
                    _spill_cached_xmm()
                    assembler.genop_cast_int_to_float(op, [arg0], xmm0)
                    _publish_float_result(op, op_index)
                else:
                    raise CannotCompileGenExt(
                        "unsupported op: %s" % op.getopname())
                _release_dead_pool_regs(op_index)

            return frame_pos[0]

        self.jitcode.genext_compile_function = compile_shortcut

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
                    value = self._read_reg_i(ord(code[position]))
                elif argcode == 'c':
                    value = "ConstInt(%s)" % signedord(code[position])
                elif argcode == 'r':
                    value = self._read_reg_r(ord(code[position]))
                elif argcode == 'f':
                    value = self._read_reg_f(ord(code[position]))
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
                        else:
                            import pdb;pdb.set_trace()
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
            if   argcode == 'I': reg = self._read_reg_i(index)
            elif argcode == 'R': reg = self._read_reg_r(index)
            elif argcode == 'F': reg = self._read_reg_f(index)
            else: raise AssertionError(argcode)
            outvalue[startindex+i] = reg

    def _read_reg_i(self, pos):
        if pos >= self.jitcode.num_regs_i():
            const_pos = pos - self.jitcode.num_regs_i()
            value = self.jitcode.constants_i[const_pos]
            return "ConstInt(%s)" % (_int_as_str(value, lltype.typeOf(value), self._add_global), )
        else:
            return "self.registers_i[%s]" % (pos, )

    def _read_reg_r(self, pos):
        if pos >= self.jitcode.num_regs_r():
            const_pos = pos - self.jitcode.num_regs_r()
            value = self.jitcode.constants_r[const_pos]
            return "ConstPtr(%s)" % (self._add_global(value), )
        else:
            return "self.registers_r[%s]" % (pos, )

    def _read_reg_f(self, pos):
        if pos >= self.jitcode.num_regs_f():
            const_pos = pos - self.jitcode.num_regs_f()
            value = self.jitcode.constants_f[const_pos]
            return "ConstFloat(%s)" % (self._add_global(value), )
        else:
            return "self.registers_f[%s]" % (pos, )

    def fill_registers(self, lines, argname, length, position, argcode):
        assert argcode in 'IRF'
        code = self.jitcode.code
        for i in range(length):
            index = ord(code[position+i])
            if   argcode == 'I':
                lines.append("%s.registers_i[%s] = %s" % (argname, i, self._read_reg_i(index)))
            elif argcode == 'R':
                lines.append("%s.registers_r[%s] = %s" % (argname, i, self._read_reg_r(index)))
            elif argcode == 'F':
                lines.append("%s.registers_f[%s] = %s" % (argname, i, self._read_reg_f(index)))
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
        self.todo = collections.deque()
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

    def make_code(self):
        code_and_spec_per_pc = {}
        while self.todo:
            spec = self.todo.popleft()
            if SLOWPATH_PROFILE_ENABLED:
                code = spec.make_code_with_profiling()
            else:
                code = spec.make_code()
            code_and_spec_per_pc[spec.spec_pc] = code, spec
        return code_and_spec_per_pc


def _int_as_str(value, TYPE, add_global):
    if isinstance(TYPE, lltype.Ptr):
        assert TYPE.TO._gckind == 'raw'
        return "support.ptr2int(%s)" % (add_global(value), )
    val = lltype.cast_primitive(lltype.Signed, value)
    if not isinstance(val, int):
        return add_global(value)
    return str(val)


def _float_as_str(value, TYPE, add_global):
    val = lltype.cast_primitive(TYPE, value)
    if isinstance(val, float):
        return add_global(value)
    return str(val)


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

    def __repr__(self):
        return "<Specializer %s %s %s>" % (self.name, self.orig_pc, self.constant_registers)

    def _reset_specializer(self):
        self.name = None
        self.methodname = None
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

    def _get_args_and_res(self):
        assert self.resindex is not None
        return self.insn[1:-2] + (self.insn[-1], )

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

    def make_code_with_profiling(self):
        profiler = get_profiler()
        opname = self.name.strip('-')
        args = self._get_args()

        category = classify_opcode(self.name)
        if category == REASON_IS_LIVE_OP:
            return None

        has_spec = hasattr(self, "emit_specialized_" + opname)
        has_unspec = hasattr(self, "emit_unspecialized_" + opname)
        profiler.record_method_info(opname, has_spec, has_unspec)

        try:
            if not self._check_all_constant_args(args):
                # Non-constant args path
                if not has_unspec:
                    # Determine more specific reason
                    if category:
                        profiler.record_codegen(opname, category)
                    else:
                        profiler.record_codegen(opname, REASON_NO_UNSPEC_METHOD)
                    return None
                result = self._make_code_unspecialized()
                if result is None:
                    profiler.record_codegen(opname, REASON_UNSPEC_RETURNED_NONE)
                else:
                    profiler.record_codegen(opname, REASON_FAST_PATH)
                return result
            else:
                if has_spec:
                    try:
                        result = self._make_code_specialized()
                        if result is None:
                            profiler.record_codegen(opname, REASON_SPEC_RETURNED_NONE)
                        else:
                            profiler.record_codegen(opname, REASON_FAST_PATH)
                        return result
                    except Unsupported:
                        profiler.record_codegen(opname, REASON_UNSUPPORTED_SPEC)
                        return None
                else:
                    if category:
                        profiler.record_codegen(opname, category)
                    else:
                        profiler.record_codegen(opname, REASON_NO_SPEC_METHOD)
                    return None
        except Unsupported:
            if self._check_all_constant_args(args):
                profiler.record_codegen(opname, REASON_UNSUPPORTED_SPEC)
            else:
                profiler.record_codegen(opname, REASON_UNSUPPORTED_UNSPEC)
            return None

    def _is_label(self, arg):
        return isinstance(arg, Label) or isinstance(arg, TLabel)

    def _check_all_constant_args(self, args):
        todo_check = list(args)
        for arg in todo_check:
            if isinstance(arg, ListOfKind):
                todo_check.extend(arg.content)
                continue
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

    def emit_specialized_int_and(self):
        return self._emit_specialized_int_binary("&")

    def emit_specialized_int_sub(self):
        return self._emit_specialized_int_binary("-")

    def emit_specialized_int_rshift(self):
        return self._emit_specialized_int_binary(">>")

    def emit_specialized_int_lshift(self):
        return self._emit_specialized_int_binary("<<")

    def emit_specialized_int_ge(self):
        return self._emit_specialized_int_binary(">=")

    def emit_specialized_int_gt(self):
        return self._emit_specialized_int_binary(">")

    def emit_specialized_int_le(self):
        return self._emit_specialized_int_binary("<=")

    def emit_specialized_int_lt(self):
        return self._emit_specialized_int_binary("<")

    def emit_specialized_int_eq(self):
        return self._emit_specialized_int_binary("==")

    def emit_specialized_int_ne(self):
        return self._emit_specialized_int_binary("!=")

    def emit_specialized_int_xor(self):
        return self._emit_specialized_int_binary("^")

    def emit_specialized_int_mod(self):
        return self._emit_specialized_int_binary("%")

    def emit_specialized_int_floordiv(self):
        return self._emit_specialized_int_binary("//")

    def emit_specialized_float_add(self):
        return self._emit_specialized_float_binary("+")

    def emit_specialized_float_mul(self):
        return self._emit_specialized_float_binary("*")

    def emit_specialized_float_sub(self):
        return self._emit_specialized_float_binary("-")

    def emit_specialized_float_truediv(self):
        return self._emit_specialized_float_binary("/")

    def emit_specialized_float_ge(self):
        return self._emit_specialized_float_comparison(">=")

    def emit_specialized_float_gt(self):
        return self._emit_specialized_float_comparison(">")

    def emit_specialized_float_le(self):
        return self._emit_specialized_float_comparison("<=")

    def emit_specialized_float_lt(self):
        return self._emit_specialized_float_comparison("<")

    def emit_specialized_float_eq(self):
        return self._emit_specialized_float_comparison("==")

    def emit_specialized_float_ne(self):
        return self._emit_specialized_float_comparison("!=")

    def _emit_specialized_int_binary(self, op):
        arg0, arg1, result = self._get_args_and_res()
        lines = ["i%s = %s %s %s" % (result.index, self._get_as_unboxed(arg0),
                                     op, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def _emit_specialized_float_binary(self, op):
        arg0, arg1, result = self._get_args_and_res()
        lines = ["f%s = %s %s %s" % (result.index, self._get_as_unboxed(arg0),
                                     op, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def _emit_specialized_float_comparison(self, op):
        arg0, arg1, result = self._get_args_and_res()
        lines = ["i%s = int(%s %s %s)" % (result.index, self._get_as_unboxed(arg0),
                                          op, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def emit_specialized_int_invert(self):
        arg0, result = self._get_args_and_res()
        lines = ["i%s = ~%s" % (result.index, self._get_as_unboxed(arg0))]
        self._emit_jump(lines)
        return lines

    def _emit_runtime_const_promotion(self, args, lines):
        """Emit runtime checks for ConstInt/ConstPtr/ConstFloat values.

        If all non-constant args are actually constant boxes at runtime,
        unbox them and jump to the specialized state.  Returns True if
        promotion code was emitted.
        """
        nonconst_args = []
        for arg in args:
            if isinstance(arg, Constant) or arg in self.constant_registers:
                continue
            nonconst_args.append(arg)
        if not nonconst_args:
            return False
        checks = []
        unboxes = []
        for arg in nonconst_args:
            t = self._get_type_prefix(arg)
            if t == 'i':
                checks.append("isinstance(ri%d, ConstInt)" % arg.index)
                unboxes.append("i%d = ri%d.getint()" % (arg.index, arg.index))
            elif t == 'f':
                checks.append("isinstance(rf%d, ConstFloat)" % arg.index)
                unboxes.append("f%d = rf%d.getfloat()" % (arg.index, arg.index))
            elif t == 'r':
                checks.append("isinstance(rr%d, ConstPtr)" % arg.index)
                unboxes.append("r%d = rr%d.getref_base()" % (arg.index, arg.index))
        for arg in nonconst_args:
            t = self._get_type_prefix(arg)
            lines.append("r%s%d = self.registers_%s[%d]" % (t, arg.index, t, arg.index))
        lines.append("if %s:" % " and ".join(checks))
        for line in unboxes:
            lines.append("    %s" % line)
        next_consts = self.constant_registers.union(set(nonconst_args))
        self._emit_jump(lines, constant_registers=next_consts, indent='    ')
        return True

    def _emit_unspecialized_int_binary_fast(self, rop_name, py_op):
        """Generate fast-path code for integer binary ops with non-constant args."""
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = _v0 %s _v1" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record2_int(rop.%s, %s, %s, _res)" % (
            indent, rop_name, box0, box1))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_int_comparison_fast(self, rop_name, py_op):
        """Generate fast-path code for integer comparison ops."""
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = int(_v0 %s _v1)" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record2_int(rop.%s, %s, %s, _res)" % (
            indent, rop_name, box0, box1))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_float_binary_fast(self, rop_name, py_op):
        """Generate fast-path code for float binary ops."""
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = _v0 %s _v1" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record2_float(rop.%s, %s, %s, _res)" % (
            indent, rop_name, box0, box1))
        lines.append("%sself.registers_f[%d] = _op" % (indent, result.index))
        lines.append("%sf%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_float_comparison_fast(self, rop_name, py_op):
        """Generate fast-path code for float comparison ops (returns int)."""
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = int(_v0 %s _v1)" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record2_int(rop.%s, %s, %s, _res)" % (
            indent, rop_name, box0, box1))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _get_as_unboxed_after_sync(self, arg):
        """Get unboxed value after sync - handles constant registers properly."""
        if isinstance(arg, Constant):
            return self._get_as_unboxed(arg)
        if arg in self.constant_registers:
            return self._get_as_unboxed(arg)
        t = self._get_type_prefix(arg)
        if t == 'i':
            return "self.registers_i[%d].getint()" % arg.index
        elif t == 'f':
            return "self.registers_f[%d].getfloatstorage()" % arg.index
        else:
            return "self.registers_r[%d].getref_base()" % arg.index

    def emit_unspecialized_int_add(self):
        return self._emit_unspecialized_int_binary_fast("INT_ADD", "+")

    def emit_unspecialized_int_sub(self):
        return self._emit_unspecialized_int_binary_fast("INT_SUB", "-")

    def emit_unspecialized_int_mul(self):
        return self._emit_unspecialized_int_binary_fast("INT_MUL", "*")

    def emit_unspecialized_int_and(self):
        return self._emit_unspecialized_int_binary_fast("INT_AND", "&")

    def emit_unspecialized_int_or(self):
        return self._emit_unspecialized_int_binary_fast("INT_OR", "|")

    def emit_unspecialized_int_xor(self):
        return self._emit_unspecialized_int_binary_fast("INT_XOR", "^")

    def emit_unspecialized_int_lshift(self):
        return self._emit_unspecialized_int_binary_fast("INT_LSHIFT", "<<")

    def emit_unspecialized_int_rshift(self):
        return self._emit_unspecialized_int_binary_fast("INT_RSHIFT", ">>")

    def emit_unspecialized_int_lt(self):
        return self._emit_unspecialized_int_comparison_fast("INT_LT", "<")

    def emit_unspecialized_int_le(self):
        return self._emit_unspecialized_int_comparison_fast("INT_LE", "<=")

    def emit_unspecialized_int_eq(self):
        return self._emit_unspecialized_int_comparison_fast("INT_EQ", "==")

    def emit_unspecialized_int_ne(self):
        return self._emit_unspecialized_int_comparison_fast("INT_NE", "!=")

    def emit_unspecialized_int_gt(self):
        return self._emit_unspecialized_int_comparison_fast("INT_GT", ">")

    def emit_unspecialized_int_ge(self):
        return self._emit_unspecialized_int_comparison_fast("INT_GE", ">=")

    def emit_unspecialized_float_add(self):
        return self._emit_unspecialized_float_binary_fast("FLOAT_ADD", "+")

    def emit_unspecialized_float_sub(self):
        return self._emit_unspecialized_float_binary_fast("FLOAT_SUB", "-")

    def emit_unspecialized_float_mul(self):
        return self._emit_unspecialized_float_binary_fast("FLOAT_MUL", "*")

    def emit_unspecialized_float_truediv(self):
        return self._emit_unspecialized_float_binary_fast("FLOAT_TRUEDIV", "/")

    def emit_unspecialized_float_lt(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_LT", "<")

    def emit_unspecialized_float_le(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_LE", "<=")

    def emit_unspecialized_float_eq(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_EQ", "==")

    def emit_unspecialized_float_ne(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_NE", "!=")

    def emit_unspecialized_float_gt(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_GT", ">")

    def emit_unspecialized_float_ge(self):
        return self._emit_unspecialized_float_comparison_fast("FLOAT_GE", ">=")

    def emit_unspecialized_uint_rshift(self):
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = int(r_uint(_v0) >> r_uint(_v1))" % indent)
        lines.append("%s_op = self.metainterp.history.record2_int(rop.UINT_RSHIFT, %s, %s, _res)" % (
            indent, box0, box1))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def emit_unspecialized_int_force_ge_zero(self):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = _v0 if _v0 >= 0 else 0" % indent)
        lines.append("%s_op = self.metainterp.history.record1_int(rop.INT_FORCE_GE_ZERO, %s, _res)" % (indent, box0))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def emit_unspecialized_int_signext(self):
        arg0, arg1, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0, arg1], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_v1 = %s" % (indent, self._get_as_unboxed_nosync(arg1)))
        lines.append("%s_res = int_signext(_v0, _v1)" % indent)
        lines.append("%s_op = self.metainterp.history.record2_int(rop.INT_SIGNEXT, %s, %s, _res)" % (
            indent, box0, box1))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def emit_unspecialized_cast_float_to_int(self):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = int(_v0)" % indent)
        lines.append("%s_op = self.metainterp.history.record1_int(rop.CAST_FLOAT_TO_INT, %s, _res)" % (indent, box0))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def emit_unspecialized_cast_int_to_float(self):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = float(_v0)" % indent)
        lines.append("%s_op = self.metainterp.history.record1_float(rop.CAST_INT_TO_FLOAT, %s, _res)" % (indent, box0))
        lines.append("%sself.registers_f[%d] = _op" % (indent, result.index))
        lines.append("%sf%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_int_unary_fast(self, rop_name, py_op):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = %s_v0" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record1_int(rop.%s, %s, _res)" % (
            indent, rop_name, box0))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_float_unary_fast(self, rop_name, py_op):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = %s_v0" % (indent, py_op))
        lines.append("%s_op = self.metainterp.history.record1_float(rop.%s, %s, _res)" % (
            indent, rop_name, box0))
        lines.append("%sself.registers_f[%d] = _op" % (indent, result.index))
        lines.append("%sf%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_int_is_true_fast(self):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = int(bool(_v0))" % indent)
        lines.append("%s_op = self.metainterp.history.record1_int(rop.INT_IS_TRUE, %s, _res)" % (indent, box0))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_int_is_zero_fast(self):
        arg0, result = self._get_args_and_res()
        lines = []
        promoted = self._emit_runtime_const_promotion([arg0], lines)
        if promoted:
            lines.append("else:")
            indent = "    "
        else:
            indent = ""
        box0 = self._get_as_box_nosync(arg0)
        lines.append("%s_v0 = %s" % (indent, self._get_as_unboxed_nosync(arg0)))
        lines.append("%s_res = int(_v0 == 0)" % indent)
        lines.append("%s_op = self.metainterp.history.record1_int(rop.INT_IS_ZERO, %s, _res)" % (indent, box0))
        lines.append("%sself.registers_i[%d] = _op" % (indent, result.index))
        lines.append("%si%d = _res" % (indent, result.index))
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
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
        arg0, arg1 = self._get_args()
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

    def emit_specialized_int_is_true(self):
        arg, = self._get_args()
        result = self.insn[self.resindex]
        lines = ["i%s = int(bool(%s))" % (
            result.index,
            self._get_as_unboxed(arg)
        )]
        self._emit_jump(lines)
        return lines

    def emit_specialized_int_is_zero(self):
        arg, = self._get_args()
        result = self.insn[self.resindex]
        # Constant folding: if arg is a constant, evaluate at compile time
        val = self._get_constant_int_value(arg)
        if val is not None:
            lines = ["i%s = %d" % (result.index, int(val == 0))]
            self._emit_jump(lines)
            return lines
        lines = ["i%s = int(%s == 0)" % (
            result.index,
            self._get_as_unboxed(arg)
        )]
        self._emit_jump(lines)
        return lines

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
            resultcast = _find_field_result_cast(PTRTYPE, name)
            lines.append('%s = %sllmemory.cast_adr_to_ptr(support.int2adr(i%s), %s).%s)' % (self._get_as_unboxed(res), resultcast, arg.index, self._add_global(PTRTYPE), name))
            self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
            return lines
        raise Unsupported
    emit_specialized_getfield_raw_r = emit_specialized_getfield_raw_i
    emit_specialized_getfield_raw_f = emit_specialized_getfield_raw_i

    def emit_specialized_getfield_gc_i_pure(self):
        if self.insn[2].is_always_pure():
            lines = []
            arg, descr = self._get_args()
            res = self.insn[self.resindex]
            PTRTYPE, name = _get_ptrtype_fieldname_from_fielddescr(descr)
            resultcast = _find_field_result_cast(PTRTYPE, name)
            lines.append('%s = %slltype.cast_opaque_ptr(%s, %s).%s)' % (
                self._get_as_unboxed(res), resultcast, self._add_global(PTRTYPE), self._get_as_unboxed(arg), name))
            self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
            return lines
        raise Unsupported
    emit_specialized_getfield_gc_r_pure = emit_specialized_getfield_gc_i_pure
    emit_specialized_getfield_gc_f_pure = emit_specialized_getfield_gc_i_pure

    def emit_specialized_getarrayitem_gc_i_pure(self):
        lines = []
        arg, index, descr, res = self._get_args_and_res()
        TYPE, ITEM = _get_ptrtype_itemtype_from_arraydescr(descr)
        resultcast = _find_result_cast(ITEM)
        lines.append('%s = %slltype.cast_opaque_ptr(%s, %s)[%s])' % (
            self._get_as_unboxed(res), resultcast, self._add_global(TYPE), self._get_as_unboxed(arg),
            self._get_as_unboxed(index)))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines
    emit_specialized_getarrayitem_gc_r_pure = emit_specialized_getarrayitem_gc_i_pure
    emit_specialized_getarrayitem_gc_f_pure = emit_specialized_getarrayitem_gc_i_pure

    def emit_specialized_getarrayitem_raw_i_pure(self):
        lines = []
        arg, index, descr, res = self._get_args_and_res()
        TYPE, ITEM = _get_ptrtype_itemtype_from_arraydescr(descr)
        resultcast = _find_result_cast(ITEM)
        lines.append('%s = %sllmemory.cast_adr_to_ptr(support.int2adr(i%s), %s)[%s])' % (
            self._get_as_unboxed(res), resultcast, arg.index, self._add_global(TYPE),
            self._get_as_unboxed(index)))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines
    emit_specialized_getarrayitem_raw_r_pure = emit_specialized_getarrayitem_raw_i_pure
    emit_specialized_getarrayitem_raw_f_pure = emit_specialized_getarrayitem_raw_i_pure

    def emit_specialized_getfield_gc_i(self):
        arg, descr = self._get_args()
        res = self.insn[self.resindex]
        boxvar = self._get_new_temp_variable()
        descrglob = self._add_global(descr)
        lines = []
        lines.append("%s = self.%s(%s, %s)" % (
            boxvar, self.methodname, self._get_as_box(arg), descrglob))
        lines.append("%s = %s.%s()" % (
            self._get_as_unboxed(res), boxvar, _get_primval_by_kind(res.kind)))
        lines.append("self.registers_%s[%s] = %s" % (
            res.kind[0], res.index, boxvar,))
        next_consts = self.constant_registers - {res}
        self._emit_jump(lines, constant_registers=next_consts)
        return lines
    emit_specialized_getfield_gc_r = emit_specialized_getfield_gc_i
    emit_specialized_getfield_gc_f = emit_specialized_getfield_gc_i

    def emit_specialized_setfield_gc_i(self):
        arg0, arg1, descr = self._get_args()
        descrglob = self._add_global(descr)
        lines = []
        self._emit_sync_registers(lines)
        lines.append("self.%s(%s, %s, %s)" % (
            self.methodname,
            self._get_as_box(arg0),
            self._get_as_box(arg1),
            descrglob))
        self._emit_jump(lines)
        return lines
    emit_specialized_setfield_gc_r = emit_specialized_setfield_gc_i
    emit_specialized_setfield_gc_f = emit_specialized_setfield_gc_i

    def emit_specialized_arraylen_gc(self):
        lines = []
        arg, descr, res = self._get_args_and_res()
        TYPE, ITEM = _get_ptrtype_itemtype_from_arraydescr(descr)
        lines.append('%s = len(lltype.cast_opaque_ptr(%s, %s))' % (
            self._get_as_unboxed(res), self._add_global(TYPE),
            self._get_as_unboxed(arg)))
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines

    def emit_specialized_record_quasiimmut_field(self):
        arg, descr1, descr2 = self._get_args()
        lines = []
        self._emit_box_by_type(arg, lines)
        descrglob1 = self._add_global(descr1)
        if self.constant_registers:
            lines.append("if not self.metainterp.heapcache.is_quasi_immut_known(%s, %s):" % (
                descrglob1, self._get_as_box(arg)))
            indent = '    '
            methname = '_record_quasiimmut_field_no_heapcache'
        else:
            indent = ''
            methname = self.methodname
        self._emit_sync_registers(lines, indent=indent)
        lines.append("%sself.%s(%s, %s, %s, %s)" % (
            indent, methname,
            self._get_as_box_after_sync(arg),
            descrglob1,
            self._add_global(descr2),
            self.orig_pc
        ))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_record_quasiimmut_field = emit_specialized_record_quasiimmut_field

    def emit_specialized_int_neg(self):
        arg0, result = self._get_args_and_res()
        lines = ["i%s = -%s" % (result.index, self._get_as_unboxed(arg0))]
        self._emit_jump(lines)
        return lines

    def emit_specialized_int_abs(self):
        arg0, result = self._get_args_and_res()
        lines = ["i%s = abs(%s)" % (result.index, self._get_as_unboxed(arg0))]
        self._emit_jump(lines)
        return lines

    def emit_specialized_int_copy(self):
        arg0, = self._get_args()
        res = self.insn[self.resindex]
        lines = ["%s = %s" % (self._get_as_unboxed(res), self._get_as_unboxed(arg0))]
        self._emit_jump(lines, constant_registers=self.constant_registers.union({res}))
        return lines
    emit_specialized_ref_copy = emit_specialized_int_copy

    def emit_specialized_int_pop(self):
        lines = []
        res = self.insn[self.resindex]
        boxvar = self._get_new_temp_variable()
        lines.append("%s = self.%s()" % (boxvar, self.methodname))
        lines.append("%s = %s.%s()" % (
            self._get_as_unboxed(res), boxvar, _get_primval_by_kind(res.kind)))
        lines.append("self.registers_%s[%s] = %s" % (
            res.kind[0], res.index, boxvar))
        next_consts = self.constant_registers - {res}
        self._emit_jump(lines, constant_registers=next_consts)
        return lines
    emit_specialized_ref_pop = emit_specialized_int_pop
    emit_specialized_float_pop = emit_specialized_int_pop

    def emit_specialized_int_push(self):
        arg, = self._get_args()
        lines = ["self.%s(%s)" % (self.methodname, self._get_as_box(arg))]
        self._emit_jump(lines)
        return lines
    emit_specialized_ref_push = emit_specialized_int_push
    emit_specialized_float_push = emit_specialized_int_push

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
        lines.append('i%s = %s == %s' % (
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

    def emit_unspecialized_goto(self):
        target = self.get_target_pc(self.insn[1])
        return ["pc = %d" % target]

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

    def emit_specialized_goto_if_not_ptr_iszero(self):
        return self.emit_specialized_goto_if_not_absolute('ptr_iszero', 'not %s')

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

    def emit_specialized_assert_not_none(self):
        arg = self.insn[1]
        unboxed_arg = self._get_as_unboxed(arg)
        lines = ["assert bool(%s)" % (unboxed_arg)]
        self._emit_jump(lines)
        return lines

    def emit_specialized_strlen(self):
        arg0 = self.insn[1]
        result = self.insn[self.resindex]
        lines = ["i%s = len(lltype.cast_opaque_ptr(lltype.Ptr(rstr.STR), r%d).chars)" % (
            result.index, arg0.index)]
        self._emit_jump(lines)
        return lines

    def emit_specialized_unicodelen(self):
        arg0, = self._get_args()
        result = self.insn[self.resindex]
        lines = ["i%s = len(lltype.cast_opaque_ptr(lltype.Ptr(rstr.UNICODE), r%d).chars)" % (
            result.index, arg0.index)]
        self._emit_jump(lines)
        return lines

    def emit_specialized_unicodegetitem(self):
        arg0, arg1 = self._get_args()
        result = self.insn[self.resindex]
        lines = ["i%s = ord(lltype.cast_opaque_ptr(lltype.Ptr(rstr.UNICODE), r%d).chars[%s])" % (
            result.index, arg0.index, self._get_as_unboxed(arg1))]
        self._emit_jump(lines)
        return lines

    def _get_type_prefix(self, arg):
        if isinstance(arg, Constant) or isinstance(arg, Register):
            # TODO: this logic also works for the 'else' case. probably.
            if isinstance(arg, Constant):
                kind = getkind(arg.concretetype)
            else:
                kind = arg.kind
            assert kind in ('int', 'ref', 'float')
            return kind[0]
        else:
            m = re.search('%([i,r,f])[0-9]+', str(arg))
            assert m is not None, "ensure regex match"
            return m.group(1)

    def _get_as_unboxed(self, arg):
        if isinstance(arg, Constant):
            kind = getkind(arg.concretetype)
            if kind == 'int':
                return _int_as_str(arg.value, arg.concretetype, self._add_global)
            if kind == 'ref':
                return "lltype.cast_opaque_ptr(llmemory.GCREF, %s)" % self._add_global(arg.value)
            if kind == 'float':
                return _float_as_str(arg.value, arg.concretetype, self._add_global)
            raise Unsupported
        else:
            t = self._get_type_prefix(arg)
            return "%s%s" % (t, arg.index)

    def _get_as_box(self, arg):
        if isinstance(arg, Constant):
            kind = getkind(arg.concretetype)
            if kind == 'int':
                return "ConstInt(%s)" % (_int_as_str(arg.value, arg.concretetype, self._add_global), )
            elif kind == 'ref':
                return "ConstPtr(lltype.cast_opaque_ptr(llmemory.GCREF, %s))" % self._add_global(arg.value)
            elif kind == 'float':
                return "ConstFloat(%s)" % (_float_as_str(arg.value, arg.concretetype, self._add_global), )
            else:
                assert False
        elif arg in self.constant_registers:
            if arg.kind == 'int':
                return "ConstInt(i%d)" % arg.index
            elif arg.kind == 'ref':
                return "ConstPtr(r%d)" % arg.index
            elif arg.kind == 'float':
                return "ConstFloat(f%d)" % arg.index
            else:
                assert False
        else:
            t = self._get_type_prefix(arg)
            return "r%s%d" % (t, arg.index)

    def _get_as_box_after_sync(self, arg):
        if isinstance(arg, Constant):
            return self._get_as_box(arg)
        t = self._get_type_prefix(arg)
        return "self.registers_%s[%d]" % (t, arg.index)

    def _get_as_box_nosync(self, arg):
        if isinstance(arg, Constant) or arg in self.constant_registers:
            return self._get_as_box(arg)
        t = self._get_type_prefix(arg)
        return "self.registers_%s[%d]" % (t, arg.index)

    def _get_as_unboxed_nosync(self, arg):
        if isinstance(arg, Constant) or arg in self.constant_registers:
            return self._get_as_unboxed(arg)
        t = self._get_type_prefix(arg)
        if t == 'i':
            return "self.registers_i[%d].getint()" % arg.index
        elif t == 'f':
            return "self.registers_f[%d].getfloatstorage()" % arg.index
        else:
            return "self.registers_r[%d].getref_base()" % arg.index

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
        if isinstance(arg, Constant) or arg in self.constant_registers:
            return
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
        arg0, arg1, result = self._get_args_and_res()
        self._emit_n_ary_if([arg0, arg1], lines)
        self._emit_jump(lines, constant_registers=self.constant_registers.union({arg0, arg1}),
                        indent='    ', target_pc=self.orig_pc)
        lines.append("else:")
        lines.append("    self.registers_i[%d] = self.%s(%s, %s)" % (
            result.index, self.methodname,
            self._get_as_box(arg0), self._get_as_box(arg1)))
        self._emit_jump(lines)
        return lines

    def _emit_unspecialized_float_binary(self):
        lines = []
        arg0, arg1, result = self._get_args_and_res()
        self._emit_n_ary_if([arg0, arg1], lines)
        self._emit_jump(lines, constant_registers=self.constant_registers.union({arg0, arg1}),
                        indent='    ', target_pc=self.orig_pc)
        lines.append("else:")
        lines.append("    self.registers_f[%d] = self.%s(%s, %s)" % (
            result.index, self.methodname,
            self._get_as_box(arg0), self._get_as_box(arg1)))
        self._emit_jump(lines)
        return lines

    # fall back to generic binary handler
    emit_unspecialized_int_mod = _emit_unspecialized_binary
    emit_unspecialized_int_floordiv = _emit_unspecialized_binary

    def emit_unspecialized_int_neg(self):
        return self._emit_unspecialized_int_unary_fast("INT_NEG", "-")

    def emit_unspecialized_int_invert(self):
        return self._emit_unspecialized_int_unary_fast("INT_INVERT", "~")

    def emit_unspecialized_int_is_true(self):
        return self._emit_unspecialized_int_is_true_fast()

    def emit_unspecialized_int_is_zero(self):
        return self._emit_unspecialized_int_is_zero_fast()

    def emit_unspecialized_float_neg(self):
        return self._emit_unspecialized_float_unary_fast("FLOAT_NEG", "-")

    def emit_unspecialized_float_abs(self):
        arg0, result = self._get_args_and_res()
        lines = []
        box0 = self._get_as_box_nosync(arg0)
        lines.append("_v0 = %s" % self._get_as_unboxed_nosync(arg0))
        lines.append("_res = abs(_v0)")
        lines.append("_op = self.metainterp.history.record1_float(rop.FLOAT_ABS, %s, _res)" % box0)
        lines.append("self.registers_f[%d] = _op" % result.index)
        lines.append("f%d = _res" % result.index)
        next_consts = self.constant_registers - {result}
        self._emit_jump(lines, constant_registers=next_consts)
        return lines

    def emit_unspecialized_strgetitem(self):
        lines = []
        arg0, arg1 = self.insn[1], self.insn[2]
        result = self.insn[self.resindex]
        self._emit_n_ary_if([arg0, arg1], lines)
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
        lines.append('    i%d = support.ptr2int(lltype.cast_opaque_ptr(OBJECTPTR, %s.getref_base()).typeptr)' % (
            res.index, box))
        specializer = self.work_list.specialize_pc(
            self.constant_registers.union({res}), self.work_list.pc_to_nextpc[self.orig_pc])
        lines.append('    pc = %d' % specializer.get_pc())
        lines.append('    continue')

        self._emit_sync_registers(lines)
        lines.append('i%s = self.opimpl_%s(%s, %d).getint()' % (res.index, self.insn[0], self._get_as_box(arg0), self.orig_pc))
        lines.append('pc = %d' % specializer.get_pc())
        lines.append('continue')
        return lines

    def emit_unspecialized_getfield_gc_i_pure(self):
        if self.insn[2].is_always_pure():
            arg, descr, result = self._get_args_and_res()
            lines = []
            self._emit_n_ary_if([arg], lines)
            self._emit_jump(lines, constant_registers=self.constant_registers.union({arg}),
                            indent='    ', target_pc=self.orig_pc)
            lines.append("self.registers_%s[%s] = self.opimpl_%s(%s, %s)" % (
                result.kind[0], result.index,
                self.insn[0], self._get_as_box(arg), self._add_global(descr)))
            self._emit_jump(lines)
            return lines
        raise Unsupported
    emit_unspecialized_getfield_gc_r_pure = emit_unspecialized_getfield_gc_i_pure
    emit_unspecialized_getfield_gc_f_pure = emit_unspecialized_getfield_gc_i_pure

    def _emit_unspecialized_getfield_gc_common(self):
        arg, descr, res = self._get_args_and_res()
        descrglob = self._add_global(descr)
        lines = []
        self._emit_box_by_type(arg, lines)
        self._emit_sync_registers(lines)
        lines.append("self.registers_%s[%s] = self.%s(%s, %s)" % (
            res.kind[0], res.index, self.methodname,
            self._get_as_box_after_sync(arg), descrglob))
        next_consts = self.constant_registers
        if res in next_consts:
            next_consts = next_consts - {res}
        self._emit_jump(lines, constant_registers=next_consts)
        return lines

    emit_unspecialized_getfield_gc_i = _emit_unspecialized_getfield_gc_common
    emit_unspecialized_getfield_gc_r = _emit_unspecialized_getfield_gc_common
    emit_unspecialized_getfield_gc_f = _emit_unspecialized_getfield_gc_common

    def emit_unspecialized_setfield_gc_i(self):
        arg0, arg1, descr = self._get_args()
        descrglob = self._add_global(descr)
        lines = []
        self._emit_box_by_type(arg0, lines)
        self._emit_box_by_type(arg1, lines)
        self._emit_sync_registers(lines)
        lines.append("self.%s(%s, %s, %s)" % (
            self.methodname,
            self._get_as_box_after_sync(arg0),
            self._get_as_box_after_sync(arg1),
            descrglob))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_setfield_gc_r = emit_unspecialized_setfield_gc_i
    emit_unspecialized_setfield_gc_f = emit_unspecialized_setfield_gc_i

    def emit_unspecialized_getfield_gc_i_pure(self):
        if not self.insn[2].is_always_pure():
            raise Unsupported
        return self._emit_unspecialized_getfield_gc_common()

    def emit_unspecialized_getfield_vable_i(self):
        arg, descr, result = self._get_args_and_res()
        lines = []
        self._emit_box_by_type(arg, lines)
        descrglob = self._add_global(descr)
        if self.constant_registers:
            lines.append("res = self._shortcut_getfield_vable(%s, %s)" % (
                self._get_as_box(arg), descrglob))
            lines.append("if res is not None:")
            lines.append("    self.registers_%s[%s] = res" % (
                result.kind[0], result.index))
            lines.append("else:")
            indent = '    '
        else:
            indent = ''
        self._emit_sync_registers(lines, indent=indent)
        lines.append("%sself.registers_%s[%s] = self.%s(%s, %s, %s)" % (
            indent, result.kind[0], result.index,
            self.methodname, self._get_as_box(arg), descrglob,
            self.orig_pc))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_getfield_vable_r = emit_unspecialized_getfield_vable_i
    emit_unspecialized_getfield_vable_f = emit_unspecialized_getfield_vable_i

    def emit_unspecialized_setfield_vable_i(self):
        arg0, arg1, descr = self._get_args()
        lines = []
        descrglob = self._add_global(descr)
        self._emit_box_by_type(arg0, lines)
        self._emit_box_by_type(arg1, lines)
        if self.constant_registers:
            lines.append("worked = self._shortcut_setfield_vable(%s, %s, %s)" % (
                self._get_as_box(arg0),
                self._get_as_box(arg1), descrglob))
            lines.append("if not worked:")
            indent = '    '
        else:
            indent = ''
        self._emit_sync_registers(lines, indent=indent)
        lines.append("%sself.%s(%s, %s, %s, %s)" % (
            indent, self.methodname,
            self._get_as_box_after_sync(arg0),
            self._get_as_box_after_sync(arg1),
            descrglob, self.orig_pc))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_setfield_vable_r = emit_unspecialized_setfield_vable_i

    def emit_unspecialized_getarrayitem_vable_i(self):
        arg, index, descr1, descr2, result = self._get_args_and_res()
        lines = []
        self._emit_box_by_type(arg, lines)
        check = self._emit_assignment_return_const_check(index, lines)
        descrglob1 = self._add_global(descr1)
        descrglob2 = self._add_global(descr2)
        if check is None:
            # the index is constant, we can try the fast path
            lines.append("res = self._shortcut_getarrayitem_vable(%s, %s, %s, %s)" % (
                self._get_as_box(arg), self._get_as_unboxed(index), descrglob1, descrglob2))
            lines.append("if res is not None:")
            lines.append("    self.registers_%s[%s] = res" % (
                result.kind[0], result.index))
            lines.append("else:")
            indent = '    '
        else:
            indent = ''
        self._emit_sync_registers(lines, indent=indent)
        lines.append("%sself.registers_%s[%s] = self.%s(%s, %s, %s, %s, %s)" % (
            indent,
            result.kind[0], result.index,
            self.methodname, self._get_as_box(arg), self._get_as_box_after_sync(index),
            descrglob1, descrglob2,
            self.orig_pc))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_getarrayitem_vable_r = emit_unspecialized_getarrayitem_vable_i

    def emit_unspecialized_setarrayitem_vable_i(self):
        arg, index, value, descr1, descr2 = self._get_args()
        lines = []
        self._emit_box_by_type(arg, lines)
        self._emit_box_by_type(value, lines)
        check = self._emit_assignment_return_const_check(index, lines)
        descrglob1 = self._add_global(descr1)
        descrglob2 = self._add_global(descr2)
        if check is None:
            # the index is constant, we can try the fast path
            lines.append("worked = self._shortcut_setarrayitem_vable(%s, %s, %s, %s, %s)" % (
                self._get_as_box(arg), self._get_as_unboxed(index),
                self._get_as_box(value), descrglob1, descrglob2))
            lines.append("if not worked:")
            indent = '    '
        else:
            indent = ''
        self._emit_sync_registers(lines, indent=indent)
        lines.append("%sself.%s(%s, %s, %s, %s, %s, %s)" % (
            indent,
            self.methodname, self._get_as_box(arg),
            self._get_as_box_after_sync(index),
            self._get_as_box_after_sync(value),
            descrglob1, descrglob2,
            self.orig_pc))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_setarrayitem_vable_r = emit_unspecialized_setarrayitem_vable_i

    def emit_unspecialized_getarrayitem_gc_i_pure(self):
        lines = []
        array, index, descr, result = self._get_args_and_res()
        self._emit_n_ary_if([array, index], lines)
        self._emit_jump(lines, constant_registers=self.constant_registers.union({array, index}),
                        indent='    ', target_pc=self.orig_pc)
        self._emit_box_by_type(array, lines)
        self._emit_box_by_type(index, lines)
        lines.append("self.registers_%s[%s] = self.%s(%s, %s, %s)" % (
            result.kind[0], result.index,
            self.methodname, self._get_as_box(array), self._get_as_box(index),
            self._add_global(descr)))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_getarrayitem_gc_r_pure = emit_unspecialized_getarrayitem_gc_i_pure
    emit_unspecialized_getarrayitem_gc_f_pure = emit_unspecialized_getarrayitem_gc_i_pure

    def emit_unspecialized_arraylen_gc(self):
        lines = []
        array, descr, result = self._get_args_and_res()
        self._emit_box_by_type(array, lines)
        lines.append("self.registers_%s[%s] = self.%s(%s, %s)" % (
            result.kind[0], result.index,
            self.methodname, self._get_as_box(array),
            self._add_global(descr)))
        self._emit_jump(lines)
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
    emit_unspecialized_float_copy = emit_unspecialized_int_copy

    def emit_unspecialized_int_pop(self):
        res = self.insn[self.resindex]
        lines = []
        lines.append("self.registers_%s[%s] = self.%s()" % (res.kind[0], res.index, self.methodname))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_ref_pop = emit_unspecialized_int_pop
    emit_unspecialized_float_pop = emit_unspecialized_int_pop

    def emit_unspecialized_int_push(self):
        lines = []
        arg0, = self._get_args()
        cond = self._emit_assignment_return_const_check(arg0, lines)
        assert cond is not None
        lines.append("self.%s(%s)" % (self.methodname, self._get_as_box(arg0)))
        self._emit_jump(lines)
        return lines
    emit_unspecialized_ref_push = emit_unspecialized_int_push
    emit_unspecialized_float_push = emit_unspecialized_int_push

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
        self._emit_n_ary_if([arg0], lines)
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

    def emit_unspecialized_goto_if_not_ptr_iszero(self):
        return self.emit_unspecialized_goto_if_not_absolute("ptr_iszero")

    def emit_unspecialized_goto_if_not(self):
        return self.emit_unspecialized_goto_if_not_absolute("")

    def emit_unspecialized_goto_if_not_comparison(self, name, symbol):
        lines = []
        _, arg0, arg1, arg2 = self.insn # left, right, label

        target_pc = self.get_target_pc(arg2)
        self._emit_n_ary_if([arg0, arg1], lines)
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

    def _emit_goto_if_not_int_comparison_fast(self, rop_name, py_op):
        """Generate fast-path code for goto_if_not_int_* that skips heapcache."""
        lines = []
        _, arg0, arg1, arg2 = self.insn  # left, right, label

        target_pc = self.get_target_pc(arg2)

        # Handle constant folding case
        self._emit_n_ary_if([arg0, arg1], lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0, arg1}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc(),))
        lines.append("    continue")

        box0 = self._get_as_box_nosync(arg0)
        box1 = self._get_as_box_nosync(arg1)
        lines.append("_v0 = %s" % self._get_as_unboxed_nosync(arg0))
        lines.append("_v1 = %s" % self._get_as_unboxed_nosync(arg1))
        lines.append("_cond = int(_v0 %s _v1)" % py_op)
        lines.append("condbox = self.metainterp.history.record2_int(rop.%s, %s, %s, _cond)" % (
            rop_name, box0, box1))

        self._emit_sync_registers(lines)
        lines.append("self.opimpl_goto_if_not(condbox, %d, %d, replace=False)" % (target_pc, self.orig_pc))
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
        return self._emit_goto_if_not_int_comparison_fast("INT_LT", "<")

    def emit_unspecialized_goto_if_not_int_gt(self):
        return self._emit_goto_if_not_int_comparison_fast("INT_GT", ">")

    def emit_unspecialized_goto_if_not_int_le(self):
        return self._emit_goto_if_not_int_comparison_fast("INT_LE", "<=")

    def emit_unspecialized_goto_if_not_int_ge(self):
        return self._emit_goto_if_not_int_comparison_fast("INT_GE", ">=")

    def emit_unspecialized_goto_if_not_int_ne(self):
        return self._emit_goto_if_not_int_comparison_fast("INT_NE", "!=")

    def emit_unspecialized_goto_if_not_int_eq(self):
        return self._emit_goto_if_not_int_comparison_fast("INT_EQ", "==")

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

    def emit_return(self):
        lines = []
        value, = self._get_args()
        self._emit_box_by_type(value, lines)
        lines.append("try:")
        lines.append("    self.%s(%s)" % (self.methodname, self._get_as_box(value)))
        lines.append("except ChangeFrame: return")
        lines.append("assert 0, 'unreachable'")
        return lines
    emit_unspecialized_int_return = emit_return
    emit_unspecialized_ref_return = emit_return
    emit_specialized_int_return = emit_return
    emit_specialized_ref_return = emit_return

    def emit_void_return(self):
        lines = []
        lines.append("try:")
        lines.append("    self.%s()" % (self.methodname))
        lines.append("except ChangeFrame: return")
        lines.append("assert 0, 'unreachable'")
        return lines
    emit_specialized_void_return = emit_void_return
    emit_unspecialized_void_return = emit_void_return

    def emit_unspecialized_strlen(self):
        lines = []
        arg0 = self.insn[1]
        result = self.insn[self.resindex]
        self._emit_n_ary_if([arg0], lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc()))
        lines.append("    continue")
        lines.append("else:")
        lines.append("    self.registers_i[%d] = self.opimpl_strlen(%s)" % (
            result.index, self._get_as_box(arg0)))
        self._emit_jump(lines)
        return lines

    def emit_unspecialized_unicodelen(self):
        lines = []
        arg0 = self.insn[1]
        result = self.insn[self.resindex]
        self._emit_n_ary_if([arg0], lines)
        specializer = self.work_list.specialize_insn(
            self.insn, self.constant_registers.union({arg0}), self.orig_pc)
        lines.append("    pc = %d" % (specializer.get_pc()))
        lines.append("    continue")
        lines.append("else:")
        lines.append("    self.registers_i[%d] = self.opimpl_unicodelen(%s)" % (
            result.index, self._get_as_box(arg0)))
        self._emit_jump(lines)
        return lines

    def emit_unspecialized_live(self):
        lines = []
        self._emit_jump(lines)
        return lines
    emit_specialized_live = emit_unspecialized_live

    def _prepare_residual_call(self):
        args = list(self._get_args())
        descr = args[-1]
        effectinfo = descr.get_extra_info()
        if (effectinfo.check_forces_virtual_or_virtualizable() or
            not effectinfo.check_is_elidable()):
            raise Unsupported
        function = args[0]
        middle = args[1:-1]
        list_args = []
        extra_args = []
        for item in middle:
            if isinstance(item, ListOfKind):
                list_args.append(item)
            else:
                extra_args.append(item)
        for item in extra_args:
            if not isinstance(item, IndirectCallTargets):
                raise Unsupported
        flat_args = []
        for lst in list_args:
            flat_args.extend(lst.content)
        descrglob = self._add_global(descr)
        register_args = [arg for arg in [function] + flat_args
                         if isinstance(arg, Register)]
        nonconst_register_args = [arg for arg in register_args
                                  if arg not in self.constant_registers]
        return (function, flat_args, descrglob,
                register_args, nonconst_register_args)

    def _emit_residual_call_body(self, funcbox_expr, boxes_expr, descrglob,
                                 sync_reg, indent=''):
        lines = []
        if sync_reg:
            self._emit_sync_registers(lines, indent=indent)

        if isinstance(funcbox_expr, Const):
            call_target = "self.do_residual_call_or_indirect"
        else:
            call_target = "self.do_residual_call"

        call_expr = "%s(%s, %s, %s, %s)" % (
            call_target, funcbox_expr, boxes_expr, descrglob, self.orig_pc)
        res = self.insn[self.resindex] if self.resindex is not None else None
        if res is not None:
            boxvar = self._get_new_temp_variable()
            lines.append("%s%s = %s" % (indent, boxvar, call_expr))
            lines.append("%s%s = %s.%s()" % (
                indent, self._get_as_unboxed(res), boxvar,
                _get_primval_by_kind(res.kind)))
            if sync_reg:
                lines.append("%sself.registers_%s[%s] = %s" % (
                    indent, res.kind[0], res.index, boxvar))
                next_consts = self.constant_registers - {res}
            else:
                next_consts = self.constant_registers.union({res})
        else:
            lines.append("%s%s" % (indent, call_expr))
            next_consts = self.constant_registers
        self._emit_jump(lines, constant_registers=next_consts, indent=indent)
        return lines

    def _emit_unspecialized_residual_call_common(self):
        (function, flat_args, descrglob, register_args, nonconst_register_args) \
            = self._prepare_residual_call()
        lines = []
        indent = ''
        if nonconst_register_args:
            self._emit_n_ary_if(nonconst_register_args, lines)
            specializer = self.work_list.specialize_insn(
                self.insn,
                self.constant_registers.union(set(register_args)),
                self.orig_pc)
            lines.append("    pc = %d" % specializer.get_pc())
            lines.append("    continue")
            lines.append("else:")
            indent = '    '
        for arg in [function] + flat_args:
            self._emit_box_by_type(arg, lines, indent=indent)
        funcbox = self._get_as_box_after_sync(function)
        box_items = [self._get_as_box_after_sync(arg) for arg in flat_args]
        boxes_expr = '[' + ', '.join(box_items) + ']' if box_items else '[]'
        return lines + self._emit_residual_call_body(
            funcbox, boxes_expr, descrglob, sync_reg=True, indent=indent)

    def _emit_specialized_residual_call_common(self):
        function, flat_args, descrglob, _, _ = self._prepare_residual_call()
        funcbox_expr = self._get_as_box(function)
        box_items = [self._get_as_box(arg) for arg in flat_args]
        boxes_expr = '[' + ', '.join(box_items) + ']' if box_items else '[]'
        return self._emit_residual_call_body(
            funcbox_expr, boxes_expr, descrglob, sync_reg=False)

    def emit_unspecialized_residual_call_r_r(self):
        return self._emit_unspecialized_residual_call_common()

    emit_unspecialized_residual_call_r_i = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_r_f = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_r_v = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_ir_i = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_ir_r = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_ir_f = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_ir_v = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_irf_i = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_irf_r = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_irf_f = emit_unspecialized_residual_call_r_r
    emit_unspecialized_residual_call_irf_v = emit_unspecialized_residual_call_r_r

    def emit_specialized_residual_call_r_r(self):
        return self._emit_specialized_residual_call_common()

    emit_specialized_residual_call_r_i = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_r_f = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_r_v = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_ir_i = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_ir_r = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_ir_f = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_ir_v = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_irf_i = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_irf_r = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_irf_f = emit_specialized_residual_call_r_r
    emit_specialized_residual_call_irf_v = emit_specialized_residual_call_r_r

    def _emit_sync_registers(self, lines, indent=''):
        # we need to sync the registers from the unboxed values to e.g. allow a guard to be created
        if not self.constant_registers:
            return
        func, args = _make_register_syncer(self.constant_registers)
        funcname = self._add_global(func)
        lines.append("%s%s(self, %s) # %s" % (indent, funcname, ", ".join(args), func.func_name))


class Unsupported(Exception):
    pass

class CannotCompileGenExt(Exception):
    """Raised when the GenExtension compile function cannot handle a trace."""
    pass

def _get_ptrtype_itemtype_from_arraydescr(descr):
    if hasattr(descr, 'A'): # llgraph backend
        return lltype.Ptr(descr.A), descr.A.OF
    return lltype.Ptr(descr.basesize.offsets[0].TYPE), descr.itemsize.TYPE

def _get_ptrtype_fieldname_from_fielddescr(descr):
    if hasattr(descr, 'S'): # llgraph backend
        return lltype.Ptr(descr.S), descr.fieldname
    return lltype.Ptr(descr.offset.TYPE), descr.offset.fldname

def _find_field_result_cast(T, field):
    RES = getattr(T.TO, field)
    return _find_result_cast(RES)

def _find_result_cast(RES):
    kind = getkind(RES)
    if kind == 'int':
        if RES == lltype.Signed:
            return '('
        if isinstance(RES, lltype.Primitive):
            return 'lltype.cast_primitive(lltype.Signed, '
        if isinstance(RES, lltype.Ptr):
            assert RES.TO._gckind == 'raw'
            return 'support.ptr2int('
    elif kind == 'ref':
        return 'lltype.cast_opaque_ptr(llmemory.GCREF, '
    elif kind == 'float':
        return '('
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
        kind_char = reg.kind[0]
        idx = reg.index
        if reg.kind == 'int':
            lines.append('    _old = self.registers_i[%d]' % idx)
            lines.append('    if not isinstance(_old, ConstInt) or _old.getint() != i%d:' % idx)
            lines.append('        self.registers_i[%d] = ConstInt(i%d)' % (idx, idx))
        elif reg.kind == 'ref':
            lines.append('    self.registers_%s[%d] = ConstPtr(r%d)' % (kind_char, idx, idx))
        elif reg.kind == 'float':
            lines.append('    self.registers_%s[%d] = ConstFloat(f%d)' % (kind_char, idx, idx))
        else:
            assert 0
    source = py.code.Source("\n".join(lines))
    d = {"ConstInt": ConstInt, "ConstPtr": ConstPtr, "ConstFloat": ConstFloat}
    exec source.compile() in d
    res = objectmodel.dont_inline(d[name])
    cache[key] = res, args
    return res, args

def _get_primval_by_kind(kind):
    if kind == 'int': return 'getint'
    elif kind == 'ref': return 'getref_base'
    elif kind == 'float': return 'getfloat'
    else: assert 0, "unreachable path"
