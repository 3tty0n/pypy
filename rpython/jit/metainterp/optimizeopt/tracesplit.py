from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.rlib.debug import debug_print
from rpython.rtyper.lltypesystem.llmemory import AddressAsInt, cast_int_to_adr
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.rstring import find, startswith, endswith
from rpython.rlib.objectmodel import specialize, we_are_translated, r_dict
from rpython.jit.metainterp.history import (
    AbstractFailDescr, Const, ConstInt, ConstFloat, RefFrontendOp, IntFrontendOp,
    FloatFrontendOp, INT, REF, FLOAT, VOID)
from rpython.jit.metainterp import compile, jitprof, history
from rpython.jit.metainterp.history import TargetToken
from rpython.jit.metainterp.optimizeopt.optimizer import (
    Optimizer, Optimization, BasicLoopInfo)
from rpython.jit.metainterp.optimizeopt.intutils import (
    IntBound, ConstIntBound, MININT, MAXINT, IntUnbounded)
from rpython.jit.metainterp.optimizeopt.bridgeopt import (
    deserialize_optimizer_knowledge)
from rpython.jit.metainterp.optimizeopt.util import make_dispatcher_method
from rpython.jit.metainterp.opencoder import Trace, TraceIterator
from rpython.jit.metainterp.resoperation import (
    rop, OpHelpers, ResOperation, InputArgRef, InputArgInt,
    InputArgFloat, InputArgVector, GuardResOp)

# Diagnostic toggle: when False, the pc-keyed trace-merge cut is disabled
# (every reconvergence is duplicated as before).  Used to isolate whether a
# runtime regression is caused by the merge optimization or pre-existing
# splitter/threaded-code behaviour.
ENABLE_TRACE_MERGE = False


class TokenMapError(Exception):
    """Raised when KeyError happens at taking a TargetToken from token_map"""
    def __init__(self, key=None,
                 message="KeyError happens when taking token from token_map"):
        self.key = key
        self.message = message
        if key is not None:
            self.message = "%s, key is %d" % (message, key)

class mark(object):
    CALL_ASSEMBLER = "CALL_ASSEMBLER"


def prepend_stackpos_entry_shim(operations):
    "Force bridge frame.stackpos to the first recorded guard_value."
    insert_before = -1
    frame_box = None
    sp_field_descr = None
    bridge_sp = -1
    last_pc = -1
    last_entry = -1
    for i in range(len(operations)):
        op = operations[i]
        if op.getopnum() == rop.DEBUG_MERGE_POINT and op.numargs() >= 5:
            pcbox = op.getarg(3)
            entrybox = op.getarg(4)
            if isinstance(pcbox, ConstInt) and isinstance(entrybox, ConstInt):
                last_pc = pcbox.getint()
                last_entry = entrybox.getint()
            continue
        if not (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1):
            continue
        if last_pc == last_entry:
            continue
        for j in range(i + 1, len(operations)):
            guard = operations[j]
            if (guard.getopnum() == rop.GUARD_VALUE and
                    guard.numargs() == 2 and
                    guard.getarg(0) is op and
                    isinstance(guard.getarg(1), ConstInt)):
                insert_before = i
                frame_box = op.getarg(0)
                sp_field_descr = op.getdescr()
                bridge_sp = guard.getarg(1).getint()
                break
        if insert_before >= 0:
            break
    if insert_before < 0:
        return False
    if insert_before > 0:
        prev = operations[insert_before - 1]
        if (prev.getopnum() == rop.SETFIELD_GC and prev.numargs() == 2 and
                prev.getarg(0) is frame_box and
                isinstance(prev.getarg(1), ConstInt) and
                prev.getarg(1).getint() == bridge_sp and
                prev.getdescr() is sp_field_descr):
            return False
    set_sp = ResOperation(rop.SETFIELD_GC,
                          [frame_box, ConstInt(bridge_sp)],
                          descr=sp_field_descr)
    operations[insert_before:insert_before] = [set_sp]
    return True


def rewrite_call_assembler_in_ops(operations, metainterp_sd, jitdriver_sd,
                                  loop_token=None):
    "Rewrite residual interp_CALL_ASSEMBLER calls into call_assembler_*."
    jd = jitdriver_sd
    num_green_args = jd.num_green_args
    num_red_args = jd.num_red_args
    warmrunnerstate = jd.warmstate
    repl_old = []
    repl_new = []
    changed = prepend_stackpos_entry_shim(operations)
    for idx in range(len(operations)):
        op = operations[idx]
        # Remap this op against calls already rewritten earlier in the list
        # (the new call box replaces the old one wherever it was used).
        if len(repl_old) > 0:
            for i in range(op.numargs()):
                a = op.getarg(i)
                for r in range(len(repl_old)):
                    if a is repl_old[r]:
                        op.setarg(i, repl_new[r])
            if op.is_guard():
                fa = op.getfailargs()
                if fa is not None:
                    newfa = []
                    for b in fa:
                        rb = b
                        for r in range(len(repl_old)):
                            if b is repl_old[r]:
                                rb = repl_new[r]
                        newfa.append(rb)
                    op.setfailargs(newfa)
        opnum = op.getopnum()
        if not (rop.is_call_may_force(opnum) or rop.is_plain_call(opnum)):
            continue
        numargs = op.numargs()
        lastarg = op.getarg(numargs - 1)
        if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
            op.setarg(numargs - 1, ConstInt(0))
        arg0 = op.getarg(0)
        if not isinstance(arg0, ConstInt):
            continue
        adr = cast_int_to_adr(arg0.getint())
        name = metainterp_sd.get_name_from_address(adr)
        if not endswith(name, mark.CALL_ASSEMBLER):
            continue
        arglist = op.getarglist()
        greenargs = arglist[1+num_red_args:1+num_red_args+num_green_args]
        args = arglist[1:num_red_args+1]
        assert len(args) == jd.num_red_args
        if loop_token is not None:
            new_token = loop_token
        else:
            new_token = warmrunnerstate.get_assembler_token(greenargs)
        new_opnum = OpHelpers.call_assembler_for_descr(op.getdescr())
        newop = op.copy_and_change(new_opnum, args, new_token)
        operations[idx] = newop
        repl_old.append(op)
        repl_new.append(newop)
        changed = True
    return changed

class TraceSplitInfo(BasicLoopInfo):
    """ A state after splitting the trace, containing the following:

    * target_token - generated target token for a bridge ("false" branch)
    * label_op - label operations
    * inputargs - input arguments
    * faildescr - used in the case of a bridge trace; for attaching
    """
    def __init__(self, target_token, label_op, inputargs, faildescr=None):
        self.target_token = target_token
        self.label_op = label_op
        self.inputargs = inputargs
        self.faildescr = faildescr

    def final(self):
        return True

    def __copy__(self, target_token, label_op, inputargs, faildescr=None):
        return TraceSplitInfo(target_token, label_op, inputargs, faildescr)

    def set_token(self, target_token):
        self.target_token = target_token

    def set_label(self, label_op):
        self.label_op = label_op

    def set_inputargs(self, inputargs):
        self.inputargs = inputargs

    def set_faildescr(self, faildescr):
        self.faildescr = faildescr

class OptTraceSplit(Optimizer):

    def __init__(self, metainterp_sd, jitdriver_sd,
                 optimizations=None, resumekey=None):
        Optimizer.__init__(self, metainterp_sd, jitdriver_sd)
        self.metainterp_sd = metainterp_sd
        self.jitdriver_sd = jitdriver_sd
        self.trace = None
        self.optimizations = optimizations
        self.resumekey = resumekey

        self.inputargs = None
        self.token = None
        self.token_map = {}

        self.conditions = jitdriver_sd.jitdriver.conditions

        self._already_setup_current_token = False
        self._pseudoops = []
        self._specialguardop = []
        self._newopsandinfo = []
        self._fdescrstack = []

        self._newoperations_slow_path = []

        self._slow_ops = []
        self._slow_path_flag = False
        self._slow_path_newopsandinfo = []
        self._slow_path_emit_ptr_eq = None
        self._slow_path_faildescr = None
        self._slow_path_recorded = []

        # Trace-merging state.  When a DEBUG_MERGE_POINT for a bytecode pc
        # that *already* appears in a finalized segment is re-encountered in
        # a later segment, the two paths reconverge there (the diamond's E).
        # _join_labeled remembers, per pc, the shared TargetToken whose LABEL
        # was inserted in front of the first (kept) copy of E; the duplicate
        # copy is dropped by skipping until the segment terminator.
        self._join_labeled = {}
        self._merge_skip = False
        # Set right after a residual emit_ret/emit_jump call so the trailing
        # shallow-tracing verification guards (which reference the call's now
        # dropped int result) are skipped until the next segment starts.
        self._post_emit_skip = False
        # Result boxes of the emit_ret/emit_jump marker calls that the split
        # consumes (replaced by LEAVE_PORTAL_FRAME+FINISH / JUMP).  These
        # boxes exist in no segment, so any guard failarg still referencing
        # one is stale Pass-1 residue and must be scrubbed.
        self._consumed_marker_boxes = []

        self.set_optimizations(optimizations)
        self.setup()

    def setup_condition(self):
        jd = self.jitdriver_sd
        self.conditions = jd.jitdriver.conditions

    def split(self, trace, resumestorage, call_pure_results, token):
        traceiter = trace.get_iter()
        self.token = token
        # The trace iterator decodes the op stream against its OWN freshly
        # minted input boxes (opencoder.TraceIterator.inputargs), not the
        # frontend trace.inputargs.  Every decoded/optimized op references the
        # iterator's boxes, so the LABELs/JUMPs we synthesize in pass 2 must
        # use the same identities -- otherwise check_consistency sees the body
        # referencing an undefined box.  (optimize_loop relies on the same
        # traceiter.inputargs; trace.inputargs is only the frontend side.)
        self.inputargs = traceiter.inputargs
        # Two-pass design:
        #  Pass 1 (_optimize_pass): run the standard optimizer chain over the
        #    whole linear trace.  Because we never swap self._newoperations or
        #    cut the op stream mid-flight here, OptHeap/OptVirtualize caches
        #    stay consistent and interpreter stack pop/push can fold away.
        #  Pass 2 (_split_pass): walk the *already optimized* flat op list and
        #    do the purely structural split at emit_jump/emit_ret/slow-path.
        optimized_ops = self._optimize_pass(traceiter, call_pure_results)
        # If pass 1 emitted an explicit loop LABEL, prefer its (already
        # forwarded) args so the synthesized ops stay consistent with it.
        if optimized_ops and optimized_ops[0].getopnum() == rop.LABEL:
            self.inputargs = optimized_ops[0].getarglist()
        self._split_pass(optimized_ops)
        self._relink_segment_livesets()
        return self._newopsandinfo

    def _relink_segment_livesets(self):
        "Rematerialize body boxes referenced by guard-bridges."
        consumed = self._consumed_marker_boxes

        # Rematerialize consumed emit_ret/emit_jump result boxes.
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            repl_old = []
            repl_new = []
            for m in consumed:
                if m.type != 'i' or m.numargs() < 2:
                    continue
                used = False
                for op in ops:
                    for i in range(op.numargs()):
                        if op.getarg(i) is m:
                            used = True
                    if op.is_guard():
                        fa = op.getfailargs()
                        if fa is not None:
                            for b in fa:
                                if b is m:
                                    used = True
                if not used:
                    continue
                mb = ResOperation(rop.SAME_AS_I, [m.getarg(1)])
                repl_old.append(m)
                repl_new.append(mb)
            if not repl_old:
                continue
            for op in ops:
                for i in range(op.numargs()):
                    a = op.getarg(i)
                    for r in range(len(repl_old)):
                        if a is repl_old[r]:
                            op.setarg(i, repl_new[r])
                if op.is_guard():
                    fa = op.getfailargs()
                    if fa is not None:
                        newfa = []
                        for b in fa:
                            rb = b
                            for r in range(len(repl_old)):
                                if b is repl_old[r]:
                                    rb = repl_new[r]
                            newfa.append(rb)
                        op.setfailargs(newfa)
            if ops and ops[0].getopnum() == rop.LABEL:
                ops = [ops[0]] + repl_new + ops[1:]
            else:
                ops = repl_new + ops
            self._newopsandinfo[segidx] = (info, ops)

        # box -> its defining op (SSA: one def per box).
        def_boxes = []
        def_ops = []
        for (info, ops) in self._newopsandinfo:
            for op in ops:
                if op.getopnum() == rop.LABEL:
                    continue
                if op.type != 'v':
                    def_boxes.append(op)
                    def_ops.append(op)

        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            faildescr = info.faildescr
            if not isinstance(faildescr, compile.AbstractResumeGuardDescr):
                continue

            # Bridge live-in == marked guard's untouched failargs.
            guard_fa = None
            for (gi, gops) in self._newopsandinfo:
                for gop in gops:
                    if gop.is_guard() and gop.getdescr() is faildescr:
                        guard_fa = gop.getfailargs()
                        break
                if guard_fa is not None:
                    break
            if guard_fa is None:
                continue
            resume_boxes = []
            for b in guard_fa:
                if b is not None and b not in resume_boxes:
                    resume_boxes.append(b)

            # Boxes the bridge uses before defining, that resume won't
            # restore -> must be rematerialized from resume_boxes.
            defined = []
            for b in resume_boxes:
                defined.append(b)
            needed = []
            for op in ops:
                if op.getopnum() == rop.LABEL:
                    continue
                for i in range(op.numargs()):
                    box = op.getarg(i)
                    if box is None or isinstance(box, Const):
                        continue
                    if (box in defined or box in needed or
                            box in consumed):
                        continue
                    needed.append(box)
                if op.type != 'v':
                    defined.append(op)

            # Transitive closure of side-effect-free defining ops.
            remat = []
            worklist = needed[:]
            while worklist:
                box = worklist.pop()
                if box in resume_boxes or box in remat:
                    continue
                dop = None
                for k in range(len(def_boxes)):
                    if def_boxes[k] is box:
                        dop = def_ops[k]
                        break
                if dop is None:
                    continue
                opnum = dop.getopnum()
                if (not rop.has_no_side_effect(opnum) or
                        rop.is_malloc(opnum) or rop.can_raise(opnum)):
                    continue
                remat.append(dop)
                for i in range(dop.numargs()):
                    a = dop.getarg(i)
                    if a is None or isinstance(a, Const):
                        continue
                    if a not in resume_boxes:
                        worklist.append(a)

            # Emit remat ops in original definition order (deps first).
            ordered = []
            for d in def_ops:
                if d in remat:
                    ordered.append(d)
            if ordered:
                self._newopsandinfo[segidx] = (info, ordered + ops)

            info.inputargs = resume_boxes
            old_label = info.label_op
            if old_label is not None:
                info.label_op = ResOperation(rop.LABEL, resume_boxes,
                                             old_label.getdescr())

        self._apply_body_contract_shims()

    def _body_contract_for_guard(self, faildescr):
        body_ops = None
        for (bi, bops) in self._newopsandinfo:
            for bop in bops:
                if bop.is_guard() and bop.getdescr() is faildescr:
                    body_ops = bops
                    break
            if body_ops is not None:
                break
        if body_ops is None:
            return None

        frame_box = None
        stack_box = None
        stack_field_descr = None
        sp_field_descr = None
        arr_descr = None
        callee_sp = -1
        arg_slot = -1

        for i in range(len(body_ops)):
            op = body_ops[i]
            if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1):
                for j in range(i + 1, len(body_ops)):
                    guard = body_ops[j]
                    if (guard.getopnum() == rop.GUARD_VALUE and
                            guard.numargs() == 2 and
                            guard.getarg(0) is op and
                            isinstance(guard.getarg(1), ConstInt)):
                        frame_box = op.getarg(0)
                        sp_field_descr = op.getdescr()
                        callee_sp = guard.getarg(1).getint()
                        break
                if callee_sp >= 0:
                    break

        if frame_box is None:
            return None

        for i in range(len(body_ops)):
            op = body_ops[i]
            if not (op.getopnum() == rop.GETFIELD_GC_R and
                    op.numargs() == 1 and op.getarg(0) is frame_box):
                continue
            for j in range(i + 1, len(body_ops)):
                use = body_ops[j]
                if (use.numargs() > 0 and use.getarg(0) is op and
                        use.getopnum() in (rop.ARRAYLEN_GC,
                                           rop.GETARRAYITEM_GC_R,
                                           rop.SETARRAYITEM_GC)):
                    stack_box = op
                    stack_field_descr = op.getdescr()
                    break
            if stack_box is not None:
                break
        if stack_box is None:
            return None

        for op in body_ops:
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getarg(0) is stack_box and
                    isinstance(op.getarg(1), ConstInt)):
                arg_slot = op.getarg(1).getint()
                arr_descr = op.getdescr()
                break
        if arg_slot < 0 or arr_descr is None:
            return None

        return (frame_box, callee_sp, arg_slot,
                stack_field_descr, sp_field_descr, arr_descr)

    def _first_bridge_stack_read_slot(self, ops, stack_box):
        for op in ops:
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getarg(0) is stack_box and
                    isinstance(op.getarg(1), ConstInt)):
                return op.getarg(1).getint()
        return -1

    def _apply_body_contract_shims(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            faildescr = info.faildescr
            if not isinstance(faildescr, compile.AbstractResumeGuardDescr):
                continue

            contract = self._body_contract_for_guard(faildescr)
            if contract is None:
                continue
            (frame_box, callee_sp, arg_slot, stack_field_descr,
             sp_field_descr, arr_descr) = contract

            ops = self._drop_bridge_recorded_stack_bookkeeping(
                ops, frame_box, stack_field_descr, sp_field_descr, arr_descr)
            ops = self._trim_base_case_zero_return_bridge(ops)

            has_call_asm = False
            for op in ops:
                if rop.is_call_assembler(op.getopnum()):
                    has_call_asm = True
                    break

            if not has_call_asm:
                self._newopsandinfo[segidx] = (info, ops)
                continue

            stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                                     descr=stack_field_descr)
            bridge_arg_slot = -1
            for op in ops:
                if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                        op.numargs() == 2 and
                        isinstance(op.getarg(1), ConstInt)):
                    bridge_arg_slot = op.getarg(1).getint()
                    break
            if bridge_arg_slot < 0:
                continue
            n_box = ResOperation(rop.GETARRAYITEM_GC_R,
                                 [stack_box, ConstInt(arg_slot)],
                                 descr=arr_descr)
            set_arg = ResOperation(
                rop.SETARRAYITEM_GC,
                [stack_box, ConstInt(bridge_arg_slot), n_box],
                descr=arr_descr)
            shim = [stack_box, n_box, set_arg]

            insert_at = 1 if ops and ops[0].getopnum() == rop.LABEL else 0
            ops = ops[:insert_at] + shim + ops[insert_at:]
            self._newopsandinfo[segidx] = (info, ops)

    def _drop_bridge_recorded_stack_bookkeeping(self, ops, frame_box,
                                                stack_field_descr,
                                                sp_field_descr, arr_descr):
        newops = []
        stack_boxes = []
        for op in ops:
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    isinstance(op.getarg(1), ConstInt) and
                    op.getdescr() is sp_field_descr):
                continue
            if (op.getopnum() == rop.GUARD_VALUE and op.numargs() == 2 and
                    isinstance(op.getarg(1), ConstInt)):
                sp_box = op.getarg(0)
                if (sp_box.getopnum() == rop.GETFIELD_GC_I and
                        sp_box.numargs() == 1 and
                        sp_box.getarg(0) is frame_box and
                        sp_box.getdescr() is sp_field_descr):
                    continue
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
            if (op.getopnum() == rop.SETARRAYITEM_GC and op.numargs() == 3 and
                    op.getdescr() is arr_descr and
                    isinstance(op.getarg(1), ConstInt)):
                stack_box = op.getarg(0)
                for known_stack_box in stack_boxes:
                    if stack_box is known_stack_box:
                        break
                else:
                    newops.append(op)
                continue
            newops.append(op)
        return newops

    def _trim_base_case_zero_return_bridge(self, ops):
        finish_op = None
        leave_op = None
        for op in ops:
            if op.getopnum() == rop.LEAVE_PORTAL_FRAME:
                leave_op = op
            elif op.getopnum() == rop.FINISH:
                finish_op = op
        if finish_op is None or leave_op is None or finish_op.numargs() != 1:
            return ops

        finish_arg = finish_op.getarg(0)
        if not rop.is_call(finish_arg.getopnum()):
            return ops
        try:
            name = self._get_name_from_op(finish_arg)
        except Exception:
            return ops
        if not endswith(name, "Frame.POP") and not endswith(name, ".POP"):
            return ops

        zero_box = None
        zero_setfield = None
        for op in ops:
            if op.getopnum() != rop.SETFIELD_GC or op.numargs() != 2:
                continue
            if not isinstance(op.getarg(1), ConstInt):
                continue
            if op.getarg(1).getint() != 0:
                continue
            box = op.getarg(0)
            if box.getopnum() == rop.NEW_WITH_VTABLE:
                zero_box = box
                zero_setfield = op
        if zero_box is None:
            return ops

        newops = []
        if ops and ops[0].getopnum() == rop.LABEL:
            newops.append(ops[0])
        newops.append(zero_box)
        newops.append(zero_setfield)
        newops.append(leave_op)
        newops.append(ResOperation(rop.FINISH, [zero_box],
                                   descr=finish_op.getdescr()))
        return newops

    def _optimize_pass(self, traceiter, call_pure_results):
        """Pass 1: standard optimization, no splitting.  The split markers
        (JIT_EMIT_JUMP/RET, BEGIN/END_SLOW_PATH) and DEBUG_MERGE_POINT have no
        optimize_ handler so they fall through optimize_default and are emitted
        verbatim, surviving into the flat list for pass 2."""
        self._splitting = False
        Optimizer.propagate_all_forward(self, traceiter, call_pure_results)
        optimized_ops = self._newoperations
        # reset accumulator; pass 2 rebuilds segments by plain appends and
        # does not run the optimizer chain again.
        self._newoperations = []
        return optimized_ops

    def _split_pass(self, ops, flush=True):
        self._splitting = True
        last_op = None

        jd = self.jitdriver_sd
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args

        slow_ops_jump_op = None
        slow_path_label = None
        opindex = 0
        nops = len(ops)
        while opindex < nops:
            self._really_emitted_operation = None
            op = ops[opindex]
            opindex += 1
            opnum = op.getopnum()
            numargs = op.numargs()

            # remove op related to pseudo ops
            can_emit = True
            for arg in op.getarglist():
                if arg in self._pseudoops:
                    can_emit = False
                    self.emit_pseudoop(op)
                    break

            if not can_emit:
                continue

            # We merged the path being built into an already-emitted copy of
            # E (the diamond's join).  Everything from here up to and
            # including this duplicate path's terminator is the redundant
            # second copy of E -- drop it; the JUMP we already appended
            # transfers control to the single shared copy instead.
            if self._merge_skip:
                if opnum in (rop.FINISH, rop.JUMP) or \
                   rop.is_jit_emit_jump(opnum) or \
                   rop.is_jit_emit_ret(opnum):
                    self._merge_skip = False
                continue

            # Drop the shallow-tracing verification guards that trail a
            # residual emit_ret/emit_jump call.  The segment was already
            # terminated by the handler; the next DEBUG_MERGE_POINT opens the
            # following segment and must be processed normally.
            if self._post_emit_skip:
                if opnum == rop.DEBUG_MERGE_POINT:
                    self._post_emit_skip = False
                else:
                    continue

            just_setup = False
            if not self._already_setup_current_token and \
               opnum == rop.DEBUG_MERGE_POINT:
                arglist = op.getarglist()
                # TODO: look up `pc' by name
                greens = arglist[1+num_red_args:1+num_red_args+num_green_args]
                box = greens[0]
                assert isinstance(box, ConstInt)
                token = self._create_token()
                self.token_map[box.getint()] = token
                self._newoperations.append(
                    ResOperation(rop.LABEL, self.inputargs, token))
                self._already_setup_current_token = True
                just_setup = True

            # Reconvergence: this DEBUG_MERGE_POINT's pc already appears in a
            # finalized segment, so the current path rejoins it here.  Cut
            # the current segment with a JUMP to the shared LABEL that
            # _merge_join inserted in front of the kept copy, then skip the
            # duplicate copy of E.  Not for the segment's own first DMP, and
            # left to the dedicated machinery while inside a slow path.
            if ENABLE_TRACE_MERGE and \
               opnum == rop.DEBUG_MERGE_POINT and not just_setup and \
               not self._slow_path_flag:
                pc = self._dmp_pcbox(op).getint()
                merge_token = self._merge_join(pc)
                if merge_token is not None:
                    jump_op = ResOperation(rop.JUMP, self.inputargs,
                                           descr=merge_token)
                    # A merge cut is never the first segment (the join pc must
                    # already live in a finalized segment), so _newoperations[0]
                    # is this segment's setup LABEL -- not an extra parsed input
                    # label.  Keep it in the stored ops and reuse it as the
                    # info's label_op, mirroring the final-finalize path below.
                    label_op = self._newoperations[0]
                    info = TraceSplitInfo(label_op.getdescr(), label_op,
                                          self.inputargs, self.resumekey)
                    self._newopsandinfo.append(
                        (info, self._newoperations + [jump_op]))
                    self._newoperations = []
                    self._already_setup_current_token = False
                    if len(self._fdescrstack) > 0:
                        self.resumekey = self._fdescrstack.pop()
                    self._merge_skip = True
                    continue

            if opnum in (rop.FINISH, rop.JUMP):
                last_op = op
                break

            # The real interpreter never produces JIT_EMIT_JUMP/RET resops:
            # tla.py calls the @jit.dont_look_inside helpers tlib.emit_jump /
            # tlib.emit_ret, so the split markers reach us as ordinary residual
            # CALL_I/CALL_N ops.  Recognize them by callee name and route them
            # through the same handlers as the resop form.  The shallow-tracing
            # verification guards that follow such a call (int_lt / guard_true /
            # guard_value on the call's int result, terminated by the next
            # DEBUG_MERGE_POINT) reference a box we are about to drop, so mark
            # them to be skipped until the next segment's DEBUG_MERGE_POINT.
            if rop.is_jit_emit_jump(opnum):
                self._handle_emit_jump(op)
                continue
            elif rop.is_jit_emit_ret(opnum):
                self._handle_emit_ret(op)
                continue
            elif self._is_emit_marker_call(op, opnum, "emit_jump"):
                self._handle_emit_jump(op)
                self._post_emit_skip = True
                continue
            elif self._is_emit_marker_call(op, opnum, "emit_ret"):
                self._handle_emit_ret(op)
                self._post_emit_skip = True
                continue
            elif rop.is_begin_slow_path(opnum):
                self._slow_path_flag = True
                jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
                original_jitcell_token = self.token.original_jitcell_token
                token = TargetToken(jitcell_token,
                                    original_jitcell_token=original_jitcell_token)
                label = ResOperation(rop.LABEL, self.inputargs, descr=token)

                self._newoperations_slow_path = self._newoperations
                self._newoperations = self._slow_ops
                self._newoperations.append(label)

                original_jitcell_token = self.token.original_jitcell_token
                token = TargetToken(jitcell_token,
                                    original_jitcell_token=original_jitcell_token)
                label = ResOperation(rop.LABEL, self.inputargs, descr=token)
                self._slow_ops.append(label)
                continue

            if self._slow_path_flag:
                # re-encountering DEBUG_MERGE_POINT when the slow flag is True
                # means the slow path ends just before
                if rop.is_debug_merge_point(opnum):
                    assert slow_ops_jump_op is not None
                    self._newoperations.append(slow_ops_jump_op)
                    slow_ops_jump_op = None

                    assert self._slow_path_faildescr is not None
                    label = self._slow_ops[0]
                    info = TraceSplitInfo(label.getdescr(), label, self.inputargs,
                                          faildescr=self._slow_path_faildescr)
                    self._slow_path_newopsandinfo.append((info, self._slow_ops[1:]))
                    self._slow_path_recorded.append(self._slow_ops[1:])

                    self._newoperations = self._newoperations_slow_path[:]
                    self._slow_ops = []
                    self._newoperations_slow_path = []
                    self._slow_path_flag = False

                    self._newoperations.append(slow_path_label)
                    slow_path_label = None

                    self._emit2(op)
                    continue

                elif rop.is_end_slow_path(opnum):
                    jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
                    original_jitcell_token = self.token.original_jitcell_token
                    token_jump_to = TargetToken(jitcell_token,
                                                original_jitcell_token=original_jitcell_token)
                    jump_op = ResOperation(rop.JUMP, self.inputargs, descr=token_jump_to)
                    slow_path_label = ResOperation(rop.LABEL, self.inputargs, descr=token_jump_to)
                    slow_ops_jump_op = jump_op
                    continue

                self._emit2(op)
                continue

            self._emit2(op)

        if flush:
            if last_op:
                self._newoperations.append(last_op)

        if self._newoperations and \
           self._newoperations[-1].getopnum() in (rop.JUMP, rop.FINISH):
            label = self._newoperations[0]
            info = TraceSplitInfo(label.getdescr(), label, self.inputargs, self.resumekey)
            self._newopsandinfo.append((info, self._newoperations))

        self._newopsandinfo.extend(self._slow_path_newopsandinfo)

        self.resumedata_memo.update_counters(self.metainterp_sd.profiler)
        # XXX: workaround to pass the type checking
        return self._newopsandinfo[0]

    def emit_pseudoop(self, op):
        self._pseudoops.append(op)

    def optimize_default(self, op):
        self.emit(op)

    def optimize_GUARD_VALUE(self, op):
        # Pass 1 only emits the guard verbatim.  The split-specific failarg
        # rewriting / _fdescrstack bookkeeping was moved to _mark_guard, run
        # from _emit2 in pass 2 so it stays correctly interleaved (LIFO) with
        # the emit_jump/emit_ret pops that also consume _fdescrstack.
        self.emit(op)

    optimize_GUARD_TRUE = optimize_GUARD_VALUE
    optimize_GUARD_FALSE = optimize_GUARD_VALUE

    def _emit2(self, op):
        """Pass-2 structural emit: rewrite the shallow-tracing flag arg,
        detect the slow-path emit_ptr_eq marker and run the deferred
        guard-marking, then append the (already optimized) op to the current
        segment without re-running the optimizer chain."""
        opnum = op.getopnum()
        if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
            numargs = op.numargs()
            lastarg = op.getarg(numargs - 1)
            if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                op.setarg(numargs - 1, ConstInt(0))
            name = self._get_name_from_op(op)
            if endswith(name, "emit_ptr_eq"):
                self._slow_path_emit_ptr_eq = op
        elif opnum in (rop.GUARD_VALUE, rop.GUARD_TRUE, rop.GUARD_FALSE):
            self._mark_guard(op)
        self._newoperations.append(op)

    def _mark_guard(self, op):
        if self._check_if_guard_marked(op):
            newfailargs = []
            for farg in op.getfailargs():
                if not farg in self._specialguardop:
                    newfailargs.append(farg)

            op.setfailargs(newfailargs)
            self._fdescrstack.append(op.getdescr())
        elif op.getarg(0) is self._slow_path_emit_ptr_eq:
            self._slow_path_faildescr = op.getdescr()
            op.setfailargs(self.inputargs)

    def optimize_CALL_N(self, op):
        name = self._get_name_from_op(op)
        if self._check_if_cond_marked(op):
            self._specialguardop.append(op)
            self.emit(op)
        elif startswith(name, "handler_"):
            self._handle_dummy_flag(op)
        else:
            self.emit(op)

    def optimize_CALL_MAY_FORCE_R(self, op):
        name = self._get_name_from_op(op)
        if endswith(name, mark.CALL_ASSEMBLER):
            self._handle_call_assembler(op)
        elif startswith(name, "handler_"):
            self._handle_dummy_flag(op)
        else:
            self.emit(op)

    optimize_CALL_MAY_FORCE_I = optimize_CALL_MAY_FORCE_R
    optimize_CALL_MAY_FORCE_F = optimize_CALL_MAY_FORCE_R
    optimize_CALL_MAY_FORCE_N = optimize_CALL_MAY_FORCE_R

    optimize_CALL_I = optimize_CALL_N
    optimize_CALL_F = optimize_CALL_N
    optimize_CALL_R = optimize_CALL_N

    def _handle_emit_ret(self, op):
        inputargs = self.inputargs
        jd_no = self.jitdriver_sd.index
        result_type = self.jitdriver_sd.result_type
        sd = self.metainterp_sd
        numargs = op.numargs()
        assert numargs > 1, "emit_ret must have at least one argument"
        if result_type == history.VOID:
            exits = []
            finishtoken = sd.done_with_this_frame_descr_void
        elif result_type == history.INT:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_int
        elif result_type == history.REF:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_ref
        elif result_type == history.FLOAT:
            exits = [op.getarg(numargs - 1)]
            finishtoken = sd.done_with_this_frame_descr_float
        else:
            assert False

        # host-stack style
        ret_ops = [
            ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(jd_no)], None),
            ResOperation(rop.FINISH, exits, finishtoken)
        ]

        self._consumed_marker_boxes.append(op)

        label_op = self._newoperations[0]
        info = TraceSplitInfo(label_op.getdescr(), label_op, inputargs, self.resumekey)
        self._newopsandinfo.append((info, self._newoperations[1:] + ret_ops))
        self._newoperations = []

        self._already_setup_current_token = False

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_emit_jump(self, op, emit_label=False):
        jd = self.jitdriver_sd
        inputargs = self.inputargs
        numargs = op.numargs()

        # create token
        targetbox = op.getarg(numargs - 1)
        assert isinstance(targetbox, ConstInt)
        target = targetbox.getvalue()
        if target in self.token_map.keys():
            target_token = self._get_token(target)
        else:
            # TODO: should get target_token from jitcelltoken.target_tokens
            target_token = self._create_token()
            self._invest_label_jump_dest(targetbox, target_token)

        # TODO: should add target_token to jitcelltoken.target_tokens
        self.token_map[target] = target_token

        self._consumed_marker_boxes.append(op)

        jump_op = ResOperation(rop.JUMP, inputargs, descr=target_token)
        info = TraceSplitInfo(target_token, self._newoperations[0], inputargs, self.resumekey)

        self._newopsandinfo.append((info, self._newoperations[1:] + [jump_op]))
        self._newoperations = []

        self._already_setup_current_token = False

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_call_assembler(self, op):
        "convert recursive calls to an op using `call_assembler_x'"
        jd = self.jitdriver_sd

        arglist = op.getarglist()
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args
        greenargs = arglist[1+num_red_args:1+num_red_args+num_green_args]
        args = arglist[1:num_red_args+1]
        assert len(args) == jd.num_red_args
        warmrunnerstate = jd.warmstate
        # The trace being split *is* the loop we are about to compile, and
        # the recursive interp_CALL_ASSEMBLER call re-enters that very same
        # procedure (self-recursion: same pc/bytecode, only the traverse
        # stack -- a green -- differs by depth).  get_assembler_token keys on
        # those greens, so it would hand back a fresh JC_TEMPORARY callback
        # per recursion depth that nobody ever redirects, and the recursion
        # trampolines through ll_portal_runner instead of ever entering the
        # compiled loop.  Target the loop's own jitcell token directly so
        # every depth converges onto it.
        new_token = None
        if self.token is not None:
            new_token = self.token.original_jitcell_token
        if new_token is None:
            new_token = warmrunnerstate.get_assembler_token(greenargs)
        opnum = OpHelpers.call_assembler_for_descr(op.getdescr())
        newop = op.copy_and_change(opnum, args, new_token)
        op.set_forwarded(newop)
        self.emit(newop)

    def _handle_dummy_flag(self, op):
        numargs = op.numargs()
        opnum = op.getopnum()
        arglist = op.getarglist()

        newfunc = arglist[-2]
        offset = numargs - 2
        assert offset >= 0
        newargs = arglist[:offset]
        newargs[0] = newfunc

        descr = op.getdescr()
        newdescr = descr.get_calldescr_without_flag()

        newop = op.copy_and_change(opnum, newargs, descr=newdescr)
        op.set_forwarded(newop)
        self.emit(newop)

    def _dmp_pcbox(self, op):
        """Return the green `pc' ConstInt of a DEBUG_MERGE_POINT op.

        The op's arglist is [jd_index, portal_call_depth, current_call_id]
        + greenkey (see pyjitpl.debug_merge_point).  The splitter slices the
        greens at base 1+num_red_args, so greens[0] is current_call_id and
        the bytecode position (greenkey[0]) lands at greens[1]."""
        jd = self.jitdriver_sd
        num_green_args = jd.num_green_args
        num_red_args = jd.num_red_args
        arglist = op.getarglist()
        greens = arglist[1+num_red_args:1+num_red_args+num_green_args]
        assert len(greens) >= 2
        box = greens[1]
        assert isinstance(box, ConstInt)
        return box

    def _pc_in_finalized_segment(self, targetbox):
        """True if some already-finalized segment contains a
        DEBUG_MERGE_POINT for `targetbox' (= the bytecode pc).  Such a pc is
        a reconvergence point: the path being built now rejoins a path that
        has already been emitted (the diamond's E)."""
        for _, ops in self._newopsandinfo:
            for op in ops:
                if op.getopnum() == rop.DEBUG_MERGE_POINT and \
                   self._dmp_pcbox(op).same_constant(targetbox):
                    return True
        return False

    def _merge_join(self, pc):
        """If `pc' reconverges with an already-emitted segment, make that
        segment's copy of E the single shared copy: insert a LABEL (with a
        shared TargetToken) in front of its DEBUG_MERGE_POINT and return the
        token so the caller can terminate the current segment with a JUMP to
        it.  Returns None when `pc' is not (yet) a reconvergence point."""
        if pc in self._join_labeled:
            return self._join_labeled[pc]
        targetbox = ConstInt(pc)
        if not self._pc_in_finalized_segment(targetbox):
            return None
        token = self.token_map.get(pc, None)
        if token is None:
            token = self._create_token()
        self.token_map[pc] = token
        self._invest_label_jump_dest(targetbox, token)
        self._join_labeled[pc] = token
        return token

    def _check_and_insert_label(self, ops, targetbox, token):
        for i, op in enumerate(ops):
            if op.getopnum() == rop.DEBUG_MERGE_POINT:
                if self._insert_label(op, i, ops, targetbox, token):
                    return

    def _invest_label_jump_dest(self, targetbox, token):
        for _, ops in self._newopsandinfo:
            self._check_and_insert_label(ops, targetbox, token)

        self._check_and_insert_label(self._newoperations, targetbox, token)

    def _insert_label(self, op, i, ops, targetbox, token):
        # Match on the bytecode position green (greens[1] = greenkey[0]),
        # consistently with _dmp_pcbox; greens[0] is current_call_id.
        posbox = self._dmp_pcbox(op)
        if posbox.same_constant(targetbox):
            label_op = ResOperation(rop.LABEL, self.inputargs, token)
            ops.insert(i, label_op)
            return True
        return False

    def _is_emit_marker_call(self, op, opnum, suffix):
        """True if `op` is a residual call to the tlib.emit_jump/emit_ret
        helper identified by `suffix`.  These reach the splitter as ordinary
        CALL_I/CALL_N (the helpers are @jit.dont_look_inside, not the
        llop-based rlib.jit markers), so we recognize them by callee name."""
        if not (rop.is_plain_call(opnum) or rop.is_call_may_force(opnum)):
            return False
        arg0 = op.getarg(0)
        if not isinstance(arg0, ConstInt):
            return False
        name = self._get_name_from_op(op)
        return endswith(name, suffix)

    def _get_name_from_op(self, op):
        arg0 = op.getarg(0)
        assert isinstance(arg0, ConstInt)
        adr = cast_int_to_adr(arg0.getint())
        return self.metainterp_sd.get_name_from_address(adr)

    def _get_token(self, key):
        if self.token_map is None:
            raise Exception("token_map is None")

        try:
            return self.token_map[key]
        except KeyError:
            raise TokenMapError(key=key)

    def _create_token(self):
        if len(self._newopsandinfo) > 0:
            jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
            original_jitcell_token = self.token.original_jitcell_token
            return TargetToken(jitcell_token,
                               original_jitcell_token=original_jitcell_token)
        else:
            return self.token

    def _is_guard_marked(self, op, mark):
        "Check if the guard_op is marked"
        assert op.is_guard()
        failargs = op.getarglist()
        for op in self._newoperations:
            opnum = op.getopnum()
            if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
                if op in failargs:
                    name = self._get_name_from_op(op)
                    return name.find(mark) != -1
        return False

    def _check_if_guard_marked(self, op):
        conditions = self.conditions
        for cond in conditions:
            if not self._is_guard_marked(op, cond):
                continue
            return True
        return False

    def _check_if_cond_marked(self, op):
        conditions = self.conditions
        name = self._get_name_from_op(op)
        for cond in conditions:
            if not endswith(name, cond):
                continue
            return True
        return False

dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
                                      default=OptTraceSplit.optimize_default)
OptTraceSplit.propagate_forward = dispatch_opt
