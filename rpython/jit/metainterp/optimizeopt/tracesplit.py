from rpython.jit.metainterp.optimize import InvalidLoop
from rpython.rlib.debug import debug_print
from rpython.rtyper.lltypesystem.llmemory import AddressAsInt, cast_int_to_adr
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.rstring import find, startswith, endswith
from rpython.rlib.objectmodel import specialize, we_are_translated, r_dict
from rpython.jit.metainterp.history import (
    AbstractFailDescr, Const, ConstInt, ConstFloat, RefFrontendOp, IntFrontendOp,
    FloatFrontendOp, INT, REF, FLOAT, VOID, JitCellToken)
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
    rop, OpHelpers, ResOperation, AbstractResOp, InputArgRef, InputArgInt,
    InputArgFloat, InputArgVector, GuardResOp)
from rpython.rlib.jit import JitInterp

# Diagnostic toggle: when False, the pc-keyed trace-merge cut is disabled
# (every reconvergence is duplicated as before).  Used to isolate whether a
# runtime regression is caused by the merge optimization or pre-existing
# splitter/threaded-code behaviour.
ENABLE_TRACE_MERGE = False

# Helper roles live in rpython.rlib.jit.JitInterp -- use JitInterp.* directly.
# The role of a callee is read at runtime from func._jit_interp_role_ via the
# codewriter's address->annotation table (see _call_role_from_op below).


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


def shift_condition_base_case_bridge(operations, shift):
    """Shift a condition-guard base-case bridge's frame stack constants down.

    A bridge compiled out of a loop-continue condition guard (GUARD_TRUE on the
    is_true result) re-traces the deferred base case from the guard's resume.
    The condition helper pops its operand at run time but only peeks while the
    trace was recorded (shallow tracing), so the recorded bridge enters one
    stack slot too high per popped operand: its const stackpos writes, stackpos
    GUARD_VALUEs and const stack-array indices are all `shift` too large.  Bring
    them back down so the base case reads the live loop-header slots (e.g. the
    accumulator at slot 1 rather than the decremented counter at slot 2).

    Returns True if anything changed.  Mirrors _rebase_bridge_stack_constants,
    but module-level (the bridge has no OptTraceSplit instance at this point)
    and subtracting instead of adding.
    """
    if shift <= 0:
        return False
    frame_box = None
    for op in operations:
        if (op.getopnum() in (rop.GETFIELD_GC_I, rop.GETFIELD_GC_R) and
                op.numargs() == 1 and op.getarg(0).type == 'r'):
            frame_box = op.getarg(0)
            break
    if frame_box is None:
        return False
    sp_field_descr, stack_field_descr, arr_descr = \
        _find_frame_stack_descrs(operations, frame_box)
    if sp_field_descr is None or arr_descr is None:
        return False
    sp_boxes = []
    stack_boxes = []
    changed = False
    for op in operations:
        opnum = op.getopnum()
        if (opnum == rop.GETFIELD_GC_I and op.numargs() == 1 and
                op.getarg(0) is frame_box and
                op.getdescr() is sp_field_descr):
            sp_boxes.append(op)
            continue
        if (opnum == rop.GETFIELD_GC_R and op.numargs() == 1 and
                op.getarg(0) is frame_box and
                op.getdescr() is stack_field_descr):
            stack_boxes.append(op)
            continue
        if (opnum == rop.SETFIELD_GC and op.numargs() == 2 and
                op.getarg(0) is frame_box and
                op.getdescr() is sp_field_descr and
                isinstance(op.getarg(1), ConstInt)):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
            continue
        if (opnum == rop.GUARD_VALUE and op.numargs() == 2 and
                op.getarg(0) in sp_boxes and
                isinstance(op.getarg(1), ConstInt)):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
            continue
        if (opnum in (rop.GETARRAYITEM_GC_R, rop.SETARRAYITEM_GC) and
                op.numargs() >= 2 and op.getdescr() is arr_descr and
                op.getarg(0) in stack_boxes and
                isinstance(op.getarg(1), ConstInt)):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
    return changed


def _call_assembler_contract_argnum(operations, ca_index, frame_box,
                                    sp_field_descr):
    "argnum from the contract after CALL_ASSEMBLER: new_sp = INT_SUB(sp, argnum)."
    sp_box = None
    j = ca_index + 1
    limit = len(operations)
    if limit > ca_index + 24:
        limit = ca_index + 24
    while j < limit:
        op = operations[j]
        opnum = op.getopnum()
        if (opnum == rop.GETFIELD_GC_I and op.numargs() == 1 and
                op.getarg(0) is frame_box and op.getdescr() is sp_field_descr):
            sp_box = op
        elif (opnum == rop.INT_SUB and op.numargs() == 2 and
                op.getarg(0) is sp_box and isinstance(op.getarg(1), ConstInt)):
            return op.getarg(1).getint()
        j += 1
    return -1


def _is_call_assembler_like(op, metainterp_sd):
    "A converted call_assembler op, or the residual interp_CALL_ASSEMBLER call."
    if rop.is_call_assembler(op.getopnum()):
        return True
    name = _call_name_from_op(op, metainterp_sd)
    return name is not None and endswith(name, mark.CALL_ASSEMBLER)


def rebase_call_assembler_tail_loop(operations, metainterp_sd):
    """Rebase const stackpos/slots after each in-body CALL_ASSEMBLER (tak).

    Shallow tracing leaves the recorder's stackpos drifted across nested
    CALL_ASSEMBLERs; the stack contract lowers the runtime stackpos but the
    recorder's const stackpos GUARD_VALUE/SETFIELD and const-folded stack reads
    keep the drifted values.  Bring them down by the cumulative per-call drop.
    Only ops *after* a call carry drift (shift>0); the loop-entry reads of the
    fixed frame args run with shift==0 and are left untouched, so no threshold on
    the slot index is needed.  The running shift resets at each FRAME_RESET.
    """
    has_ca = False
    has_reset = False
    for op in operations:
        if _is_call_assembler_like(op, metainterp_sd):
            has_ca = True
        elif _call_role_from_op(op, metainterp_sd) == JitInterp.RESET:
            has_reset = True
    if not (has_ca and has_reset):
        return False
    frame_box = None
    for op in operations:
        if (op.getopnum() in (rop.GETFIELD_GC_I, rop.GETFIELD_GC_R) and
                op.numargs() == 1 and op.getarg(0).type == 'r'):
            frame_box = op.getarg(0)
            break
    if frame_box is None:
        return False
    sp_field_descr, stack_field_descr, arr_descr = \
        _find_frame_stack_descrs(operations, frame_box)
    if sp_field_descr is None or arr_descr is None:
        return False
    shift = 0
    sp_boxes = []
    stack_boxes = []
    changed = False
    for op in operations:
        opnum = op.getopnum()
        if _call_role_from_op(op, metainterp_sd) == JitInterp.RESET:
            shift = 0
            continue
        if _is_call_assembler_like(op, metainterp_sd):
            idx = -1
            for k in range(len(operations)):
                if operations[k] is op:
                    idx = k
                    break
            argnum = _call_assembler_contract_argnum(
                operations, idx, frame_box, sp_field_descr)
            if argnum > 0:
                shift += argnum - 1
            continue
        if shift <= 0:
            continue
        if (opnum == rop.GETFIELD_GC_I and op.numargs() == 1 and
                op.getarg(0) is frame_box and op.getdescr() is sp_field_descr):
            sp_boxes.append(op)
        elif (opnum == rop.GETFIELD_GC_R and op.numargs() == 1 and
                op.getarg(0) is frame_box and
                op.getdescr() is stack_field_descr):
            stack_boxes.append(op)
        elif (opnum == rop.SETFIELD_GC and op.numargs() == 2 and
                op.getarg(0) is frame_box and
                op.getdescr() is sp_field_descr and
                isinstance(op.getarg(1), ConstInt)):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
        elif (opnum == rop.GUARD_VALUE and op.numargs() == 2 and
                op.getarg(0) in sp_boxes and
                isinstance(op.getarg(1), ConstInt)):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
        elif (opnum in (rop.GETARRAYITEM_GC_R, rop.SETARRAYITEM_GC) and
                op.numargs() >= 2 and op.getdescr() is arr_descr and
                op.getarg(0) in stack_boxes and
                isinstance(op.getarg(1), ConstInt) and
                op.getarg(1).getint() - shift >= 0):
            op.setarg(1, ConstInt(op.getarg(1).getint() - shift))
            changed = True
    return changed


def _find_frame_stack_descrs(operations, frame_box):
    sp_field_descr = None
    stack_field_descr = None
    stack_box = None
    arr_descr = None
    sp_candidates = []
    for op in operations:
        if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1 and
                op.getarg(0) is frame_box):
            sp_candidates.append(op)
            descr = op.getdescr()
            if _descr_name_contains(descr, 'inst_stackpos'):
                sp_field_descr = descr
        elif (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                op.getarg(0) is frame_box):
            descr = op.getdescr()
            if _descr_name_contains(descr, 'inst_stackpos'):
                sp_field_descr = descr
        elif (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                op.getarg(0) is frame_box):
            descr = op.getdescr()
            if not _descr_name_contains(descr, 'inst_stack'):
                continue
            stack_box = op
            for use in operations:
                if (use.numargs() > 0 and use.getarg(0) is stack_box and
                        use.getopnum() in (rop.GETARRAYITEM_GC_R,
                                           rop.SETARRAYITEM_GC)):
                    stack_field_descr = op.getdescr()
                    stack_box = op
                    arr_descr = use.getdescr()
                    break
    if stack_box is not None and sp_field_descr is None:
        for candidate in sp_candidates:
            if _int_box_feeds_stack_array(operations, candidate, stack_box):
                sp_field_descr = candidate.getdescr()
                break
    return sp_field_descr, stack_field_descr, arr_descr


def _call_assembler_stack_effect_ops(operations, arglist, call_result):
    if len(arglist) < 5:
        return []
    frame_box = arglist[2]
    argnum_box = arglist[4]
    if not isinstance(argnum_box, ConstInt):
        return []
    sp_field_descr, stack_field_descr, arr_descr = \
        _find_frame_stack_descrs(operations, frame_box)
    if sp_field_descr is None or stack_field_descr is None or arr_descr is None:
        return []

    old_sp = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                          descr=sp_field_descr)
    new_sp = ResOperation(rop.INT_SUB, [old_sp, ConstInt(argnum_box.getint())])
    set_sp = ResOperation(rop.SETFIELD_GC, [frame_box, new_sp],
                          descr=sp_field_descr)
    effect = [old_sp, new_sp, set_sp]
    if call_result.type == 'v':
        return effect
    stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                             descr=stack_field_descr)
    resume_slot = ResOperation(rop.INT_SUB, [new_sp, ConstInt(1)])
    set_resume = ResOperation(rop.SETARRAYITEM_GC,
                              [stack_box, resume_slot, call_result],
                              descr=arr_descr)
    set_result = ResOperation(rop.SETARRAYITEM_GC,
                              [stack_box, new_sp, call_result],
                              descr=arr_descr)
    pushed_sp = ResOperation(rop.INT_ADD, [new_sp, ConstInt(1)])
    set_pushed_sp = ResOperation(rop.SETFIELD_GC, [frame_box, pushed_sp],
                                 descr=sp_field_descr)
    effect.extend([stack_box, resume_slot, set_resume, set_result])
    effect.extend([pushed_sp, set_pushed_sp])
    return effect


def _call_name_from_op(op, metainterp_sd):
    if op is None or op.numargs() == 0:
        return None
    if not (rop.is_plain_call(op.getopnum()) or
            rop.is_call_may_force(op.getopnum())):
        return None
    arg0 = op.getarg(0)
    if not isinstance(arg0, ConstInt):
        return None
    adr = cast_int_to_adr(arg0.getint())
    return metainterp_sd.get_name_from_address(adr)


def _call_role_from_op(op, metainterp_sd):
    """Return the JitInterp role of a call op's target, or JitInterp.NONE.

    The role is looked up via the codewriter's address->annotation table,
    which is populated from each helper's _jit_interp_role_ attribute.
    """
    if op is None or op.numargs() == 0:
        return JitInterp.NONE
    if not (rop.is_plain_call(op.getopnum()) or
            rop.is_call_may_force(op.getopnum())):
        return JitInterp.NONE
    arg0 = op.getarg(0)
    if not isinstance(arg0, ConstInt):
        return JitInterp.NONE
    adr = cast_int_to_adr(arg0.getint())
    # get_annotation_from_address is always defined on the real staticdata;
    # the hasattr is only for untranslated test doubles, so skip it under
    # translation (RPython has no hasattr).
    if not we_are_translated() and not hasattr(
            metainterp_sd, 'get_annotation_from_address'):
        return JitInterp.NONE
    role = metainterp_sd.get_annotation_from_address(adr)
    return role if role else JitInterp.NONE


def _is_frame_pop_call(op, metainterp_sd):
    return _call_role_from_op(op, metainterp_sd) == JitInterp.POP


def _is_frame_stack_read_call(op, metainterp_sd):
    role = _call_role_from_op(op, metainterp_sd)
    return role == JitInterp.POP or role == JitInterp.POP_RAW


def _is_frame_helper_role(role):
    return (role == JitInterp.POP or
            role == JitInterp.POP_RAW or
            role == JitInterp.PUSH or
            role == JitInterp.PUSH_RAW or
            role == JitInterp.DROP or
            role == JitInterp.RET or
            role == JitInterp.RESET or
            role == JitInterp.CONDITION)


def _jitcell_token_backend_ready(token):
    if token is None:
        return False
    clt = token.compiled_loop_token
    if clt is None:
        return False
    return clt.frame_info is not None


def _jitcell_token_inputarg_types(token):
    if token is None:
        return None
    # token is typed as the broad AbstractDescr at the call site; narrow to
    # JitCellToken (the only descr carrying _threaded_inputarg_types /
    # compiled_loop_token) so the attribute reads annotate.
    if not isinstance(token, JitCellToken):
        return None
    types = token._threaded_inputarg_types
    if types is not None:
        return types
    # The remaining fallbacks read llgraph-backend-only attributes
    # (_llgraph_loop / _debug_nbargs) that do not exist in a translated build.
    # Guard them with `not we_are_translated()` so the flow analyzer prunes the
    # whole block during translation (and never annotates the dynamic getattrs);
    # in a translated build the recorded _threaded_inputarg_types is used.
    if not we_are_translated():
        clt = token.compiled_loop_token
        if clt is not None and hasattr(clt, '_llgraph_loop'):
            return [box.type for box in clt._llgraph_loop.inputargs]
        nbargs = -1
        if clt is not None:
            nbargs = getattr(clt, '_debug_nbargs', -1)
        if nbargs >= 0 and nbargs != 1:
            return ['?'] * nbargs
    return None


def _adapt_call_assembler_args_for_token(args, token):
    types = _jitcell_token_inputarg_types(token)
    if types is None or len(types) == len(args):
        return args
    missing = len(types) - len(args)
    if missing <= 0:
        return args
    for i in range(len(args)):
        typ = types[missing + i]
        if typ != '?' and args[i].type != typ:
            return args
    prefix = []
    for i in range(missing):
        typ = types[i]
        if typ == 'i':
            prefix.append(ConstInt(0))
        elif typ == 'r':
            if not args or args[0].type != 'r':
                return args
            prefix.append(args[0])
        else:
            return args
    return prefix + args


def _descr_name_contains(descr, name):
    if descr is None:
        return False
    text = descr.repr_of_descr()
    return find(text, name, 0, len(text)) >= 0


def _box_depends_on(box, source):
    if box is source:
        return True
    depth = 0
    while depth < 4:
        if not isinstance(box, AbstractResOp):
            return False
        opnum = box.getopnum()
        if opnum != rop.INT_ADD and opnum != rop.INT_SUB:
            return False
        if box.numargs() < 2:
            return False
        arg0 = box.getarg(0)
        arg1 = box.getarg(1)
        if arg0 is source or arg1 is source:
            return True
        if isinstance(arg0, AbstractResOp):
            box = arg0
        elif isinstance(arg1, AbstractResOp):
            box = arg1
        else:
            return False
        depth += 1
    return False


def _int_box_feeds_stack_array(operations, int_box, stack_box):
    for use in operations:
        opnum = use.getopnum()
        if (opnum != rop.GETARRAYITEM_GC_R and
                opnum != rop.SETARRAYITEM_GC):
            continue
        if use.numargs() < 2 or use.getarg(0) is not stack_box:
            continue
        if _box_depends_on(use.getarg(1), int_box):
            return True
    return False


def rewrite_frame_helper_dummy_flags(operations, metainterp_sd):
    changed = False
    for op in operations:
        role = _call_role_from_op(op, metainterp_sd)
        if not _is_frame_helper_role(role):
            continue
        if role == JitInterp.POP_RAW:
            continue
        numargs = op.numargs()
        if numargs == 0:
            continue
        lastarg = op.getarg(numargs - 1)
        if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
            op.setarg(numargs - 1, ConstInt(0))
            changed = True
    return changed


def _debug_merge_point_location(op, jitdriver_sd):
    if op.getopnum() != rop.DEBUG_MERGE_POINT:
        return None
    if op.numargs() < 4:
        return None
    jd_box = op.getarg(0)
    if not isinstance(jd_box, ConstInt):
        return None
    if jd_box.getint() != jitdriver_sd.index:
        return None
    return jitdriver_sd.warmstate.get_location_str(op.getarglist()[3:])


def _tstack_is_empty_at_last_dmp(operations, jitdriver_sd):
    idx = len(operations) - 1
    while idx >= 0:
        location = _debug_merge_point_location(operations[idx], jitdriver_sd)
        if location is not None:
            pos = find(location, "tstack: ", 0, len(location))
            if pos < 0:
                return True
            return find(location, "tstack: None", 0, len(location)) >= 0
        idx -= 1
    return True


def _tstack_pc_at_last_dmp(operations, jitdriver_sd):
    idx = len(operations) - 1
    while idx >= 0:
        location = _debug_merge_point_location(operations[idx], jitdriver_sd)
        if location is not None:
            marker = "tstack: "
            pos = find(location, marker, 0, len(location))
            if pos < 0:
                return -1
            pos += len(marker)
            if (pos + 4 <= len(location) and
                    location[pos:pos + 4] == "None"):
                return -1
            value = 0
            seen = False
            while pos < len(location):
                ch = location[pos]
                if ch < '0' or ch > '9':
                    break
                value = value * 10 + (ord(ch) - ord('0'))
                seen = True
                pos += 1
            if seen:
                return value
            return -1
        idx -= 1
    return -1


def _materialize_pop_stack_read(operations, pop_index):
    pop = operations[pop_index]
    if pop.numargs() < 2:
        return None
    frame_box = pop.getarg(1)
    sp_field_descr, stack_field_descr, arr_descr = \
        _find_frame_stack_descrs(operations[:pop_index], frame_box)
    if sp_field_descr is None or stack_field_descr is None or arr_descr is None:
        return None
    sp_box = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                          descr=sp_field_descr)
    index_box = ResOperation(rop.INT_SUB, [sp_box, ConstInt(1)])
    stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                             descr=stack_field_descr)
    read_box = ResOperation(rop.GETARRAYITEM_GC_R, [stack_box, index_box],
                            descr=arr_descr)
    operations[pop_index:pop_index] = [sp_box, index_box, stack_box, read_box]
    return read_box


def _find_materialized_pop_read(operations, pop_index):
    if pop_index >= 4:
        read = operations[pop_index - 1]
        if read.getopnum() == rop.GETARRAYITEM_GC_R:
            return read
    return None


def _find_previous_stack_ref_read(operations, pop_index, frame_box):
    stack_field_descr = None
    stack_boxes = []
    for idx in range(pop_index):
        op = operations[idx]
        if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                op.getarg(0) is frame_box):
            descr = op.getdescr()
            if _descr_name_contains(descr, 'inst_stack'):
                stack_field_descr = descr
                stack_boxes.append(op)
    if stack_field_descr is None:
        return None
    idx = pop_index - 1
    while idx >= 0:
        op = operations[idx]
        if (op.getopnum() == rop.GETARRAYITEM_GC_R and op.numargs() == 2 and
                op.getarg(0) in stack_boxes):
            return op
        idx -= 1
    return None


def _rewrite_finish_to_materialized_read(operations):
    if len(operations) < 2:
        return False
    finish = operations[-1]
    if finish.getopnum() != rop.FINISH or finish.numargs() != 1:
        return False
    finish_arg = finish.getarg(0)
    if (not isinstance(finish_arg, AbstractResOp) or
            not rop.is_call(finish_arg.getopnum())):
        return False
    for idx in range(1, len(operations)):
        if operations[idx].same_box(finish_arg):
            read = _find_materialized_pop_read(operations, idx)
            if read is None:
                return False
            finish.setarg(0, read)
            return True
    return False


def _trim_bridge_to_finish_value(operations):
    finish_op = None
    leave_op = None
    for op in operations:
        if op.getopnum() == rop.LEAVE_PORTAL_FRAME:
            leave_op = op
        elif op.getopnum() == rop.FINISH:
            finish_op = op
    if finish_op is None or leave_op is None or finish_op.numargs() != 1:
        return False

    finish_arg = finish_op.getarg(0)
    if finish_arg is None or isinstance(finish_arg, Const):
        return False

    finish_def = -1
    last_keep = -1
    for i in range(len(operations)):
        op = operations[i]
        if op is finish_arg:
            finish_def = i
            last_keep = i
        elif (finish_def >= 0 and op.is_guard() and
                op.numargs() > 0 and op.getarg(0) is finish_arg):
            last_keep = i
    if last_keep < 0:
        return False

    trimmed = []
    for i in range(last_keep + 1):
        if operations[i].getopnum() != rop.LEAVE_PORTAL_FRAME:
            trimmed.append(operations[i])
    trimmed.append(leave_op)
    trimmed.append(finish_op)
    if len(trimmed) >= len(operations):
        return False
    # RPython's rtyper has no full-slice list assignment (operations[:] = ...);
    # clear and refill in place instead.
    del operations[:]
    operations.extend(trimmed)
    return True


def materialize_frame_pop_reads(operations, metainterp_sd):
    changed = False
    idx = 0
    while idx < len(operations):
        op = operations[idx]
        if _is_frame_stack_read_call(op, metainterp_sd):
            if _find_materialized_pop_read(operations, idx) is None:
                read = _materialize_pop_stack_read(operations, idx)
                if read is not None:
                    changed = True
                    idx += 4
        idx += 1
    return changed


def _bridge_target_token(loop_token):
    if loop_token is None:
        return None
    tokens = loop_token.target_tokens
    if tokens is None or len(tokens) == 0:
        return None
    for i in range(len(tokens)):
        token = tokens[i]
        if token._ll_loop_code != 0:
            return token
    return None


def _bridge_jump_args(operations, marker_index, inputargs):
    if len(inputargs) != 1:
        return inputargs
    inputarg = inputargs[0]
    idx = marker_index - 1
    while idx >= 0:
        op = operations[idx]
        if op.is_guard():
            failargs = op.getfailargs()
            if failargs is not None:
                for box in failargs:
                    if box is not None and box.type == inputarg.type:
                        return [box]
        idx -= 1
    return inputargs


def _rewrite_emit_marker_in_ops(operations, metainterp_sd, jitdriver_sd,
                                loop_token, inputargs):
    if inputargs is None:
        return False
    jd_no = jitdriver_sd.index
    idx = 0
    has_final_jump = (len(operations) > 0 and
                      operations[-1].getopnum() == rop.JUMP)
    while idx < len(operations):
        op = operations[idx]
        opnum = op.getopnum()
        is_jump = rop.is_jit_emit_jump(opnum)
        is_ret = rop.is_jit_emit_ret(opnum)
        if not (is_jump or is_ret):
            name = _call_name_from_op(op, metainterp_sd)
            if name is not None:
                is_jump = endswith(name, "emit_jump")
                is_ret = endswith(name, "emit_ret")
        if not (is_jump or is_ret):
            idx += 1
            continue
        if is_jump:
            if idx + 1 < len(operations):
                operations.pop(idx)
                continue
            target_token = _bridge_target_token(loop_token)
            if target_token is None:
                return False
            jump_args = _bridge_jump_args(operations, idx, inputargs)
            while len(operations) > idx:
                operations.pop()
            operations.append(
                ResOperation(rop.JUMP, jump_args, descr=target_token))
            return True
        if has_final_jump:
            operations.pop(idx)
            continue
        numargs = op.numargs()
        if numargs == 0:
            return False
        retbox = op.getarg(numargs - 1)
        for pop_idx in range(idx - 1, 0, -1):
            if operations[pop_idx].same_box(retbox):
                read = _find_materialized_pop_read(operations, pop_idx)
                if read is not None:
                    retbox = read
                break
        if (isinstance(retbox, AbstractResOp) and
                rop.is_call(retbox.getopnum())):
            read = None
            for read_idx in range(idx - 1, -1, -1):
                candidate = operations[read_idx]
                if candidate.getopnum() == rop.GETARRAYITEM_GC_R:
                    read = candidate
                    break
            if read is not None:
                retbox = read
        pop_index = -1
        for pop_idx in range(idx - 1, -1, -1):
            if (operations[pop_idx].same_box(retbox) and
                    _is_frame_pop_call(operations[pop_idx], metainterp_sd)):
                pop_index = pop_idx
                break
        if pop_index >= 0:
            read = _find_materialized_pop_read(operations, pop_index)
            if read is not None:
                retbox = read
        result_type = jitdriver_sd.result_type
        if result_type == history.VOID:
            exits = []
            finishtoken = metainterp_sd.done_with_this_frame_descr_void
        elif result_type == history.INT:
            exits = [retbox]
            finishtoken = metainterp_sd.done_with_this_frame_descr_int
        elif result_type == history.REF:
            exits = [retbox]
            finishtoken = metainterp_sd.done_with_this_frame_descr_ref
        elif result_type == history.FLOAT:
            exits = [retbox]
            finishtoken = metainterp_sd.done_with_this_frame_descr_float
        else:
            return False
        while len(operations) > idx:
            operations.pop()
        operations.append(
            ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(jd_no)], None))
        operations.append(
            ResOperation(rop.FINISH, exits, finishtoken))
        return True
    return False


def _rewrite_final_celltoken_jump(operations, loop_token):
    if len(operations) == 0:
        return False
    op = operations[-1]
    if op.getopnum() != rop.JUMP:
        return False
    descr = op.getdescr()
    if isinstance(descr, TargetToken):
        return False
    if isinstance(descr, JitCellToken):
        target_token = _bridge_target_token(descr)
    else:
        target_token = _bridge_target_token(loop_token)
    if target_token is None:
        return False
    op.setdescr(target_token)
    return True


def _drop_redundant_call_assembler_helpers(operations, records,
                                           metainterp_sd):
    if not records:
        return False
    newops = []
    changed = False
    for op in operations:
        role = _call_role_from_op(op, metainterp_sd)
        if role == JitInterp.DROP and op.numargs() >= 3:
            frame_box = op.getarg(1)
            n_box = op.getarg(2)
            for i in range(len(records)):
                rec_frame, rec_argnum, rec_result = records[i]
                if (frame_box is rec_frame and
                        isinstance(n_box, ConstInt) and
                        n_box.same_constant(rec_argnum)):
                    changed = True
                    break
            else:
                newops.append(op)
            continue
        if (role == JitInterp.PUSH or role == JitInterp.PUSH_RAW) and \
                op.numargs() >= 3:
            frame_box = op.getarg(1)
            value_box = op.getarg(2)
            for i in range(len(records)):
                rec_frame, rec_argnum, rec_result = records[i]
                if frame_box is rec_frame and value_box is rec_result:
                    changed = True
                    break
            else:
                newops.append(op)
            continue
        newops.append(op)
    if changed:
        del operations[:]
        operations.extend(newops)
    return changed


def has_threaded_tstack_fallthrough_to_empty_jump(operations, jitdriver_sd):
    if len(operations) == 0 or operations[-1].getopnum() != rop.JUMP:
        return False
    seen_nonempty_tstack = False
    last_location = None
    for op in operations:
        location = _debug_merge_point_location(op, jitdriver_sd)
        if location is None:
            continue
        last_location = location
        if find(location, "tstack:", 0, len(location)) >= 0:
            if find(location, "tstack: None", 0, len(location)) < 0:
                seen_nonempty_tstack = True
    if not seen_nonempty_tstack or last_location is None:
        return False
    return find(last_location, "tstack: None", 0, len(last_location)) >= 0


def rewrite_call_assembler_in_ops(operations, metainterp_sd, jitdriver_sd,
                                  loop_token=None, inputargs=None):
    "Rewrite residual interp_CALL_ASSEMBLER calls into call_assembler_*."
    jd = jitdriver_sd
    num_green_args = jd.num_green_args
    num_red_args = jd.num_red_args
    warmrunnerstate = jd.warmstate
    repl_old = []
    repl_new = []
    stack_effect_records = []
    changed = prepend_stackpos_entry_shim(operations)
    current_tstack_empty = True
    if materialize_frame_pop_reads(operations, metainterp_sd):
        changed = True
    if _rewrite_finish_to_materialized_read(operations):
        changed = True
    if _trim_bridge_to_finish_value(operations):
        changed = True
    if rewrite_frame_helper_dummy_flags(operations, metainterp_sd):
        changed = True
    for idx in range(len(operations)):
        op = operations[idx]
        if op.getopnum() == rop.DEBUG_MERGE_POINT:
            location = _debug_merge_point_location(op, jitdriver_sd)
            if location is not None:
                current_tstack_empty = (
                    find(location, "tstack: None", 0, len(location)) >= 0)
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
        if rop.is_call_assembler(opnum):
            newargs = _adapt_call_assembler_args_for_token(
                op.getarglist(), op.getdescr())
            if len(newargs) != op.numargs():
                newop = op.copy_and_change(opnum, newargs, op.getdescr())
                operations[idx] = newop
                repl_old.append(op)
                repl_new.append(newop)
                changed = True
            continue
        if not (rop.is_call_may_force(opnum) or rop.is_plain_call(opnum)):
            continue
        arg0 = op.getarg(0)
        if not isinstance(arg0, ConstInt):
            continue
        adr = cast_int_to_adr(arg0.getint())
        name = metainterp_sd.get_name_from_address(adr)
        if startswith(name, "handler_"):
            numargs = op.numargs()
            lastarg = op.getarg(numargs - 1)
            if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                op.setarg(numargs - 1, ConstInt(0))
        if not endswith(name, mark.CALL_ASSEMBLER):
            continue
        if not current_tstack_empty:
            if op.numargs() > 0:
                lastarg = op.getarg(op.numargs() - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(op.numargs() - 1, ConstInt(0))
                    changed = True
            continue
        arglist = op.getarglist()
        greenargs = arglist[1+num_red_args:1+num_red_args+num_green_args]
        args = arglist[1:num_red_args+1]
        assert len(args) == jd.num_red_args
        if loop_token is not None:
            new_token = loop_token
        else:
            new_token = warmrunnerstate.get_assembler_token(greenargs)
        args = _adapt_call_assembler_args_for_token(args, new_token)
        new_opnum = OpHelpers.call_assembler_for_descr(op.getdescr())
        newop = op.copy_and_change(new_opnum, args, new_token)
        operations[idx] = newop
        effect = _call_assembler_stack_effect_ops(operations, arglist, newop)
        if effect:
            operations[idx + 1:idx + 1] = effect
            if len(arglist) >= 5 and isinstance(arglist[4], ConstInt):
                stack_effect_records.append((arglist[2], arglist[4], newop))
        repl_old.append(op)
        repl_new.append(newop)
        changed = True
    if _drop_redundant_call_assembler_helpers(
            operations, stack_effect_records, metainterp_sd):
        changed = True
    if _rewrite_emit_marker_in_ops(operations, metainterp_sd, jitdriver_sd,
                                   loop_token, inputargs):
        changed = True
    if _rewrite_final_celltoken_jump(operations, loop_token):
        changed = True
    if rebase_call_assembler_tail_loop(operations, metainterp_sd):
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
        self._pending_is_true_frame = None
        self._shift_after_is_true_frame = None
        self._shift_after_is_true_spdescr = None
        self._shift_after_is_true_stackdescr = None
        self._shift_after_is_true_stackboxes = []
        # Result box of the most recently emitted CONDITION helper call (e.g.
        # is_true).  The loop-continue GUARD_TRUE that reads it is a condition
        # guard: its base-case bridge must be shifted down by the helper's pop.
        self._last_condition_call = None
        # Descr of a loop-continue condition GUARD_TRUE whose base case may be
        # deferred.  Only stamped with tcg_cond_pop once we confirm the segment
        # is a FRAME_RESET tail-loop (RESET role seen, then closed by emit_jump):
        # only then is the deferred base case recorded one slot too high.  A
        # plain CALL_ASSEMBLER recursion (sum/fib) records its base case inline
        # with the pop already applied, so it must NOT be shifted.
        self._pending_cond_guard_descr = None
        self._segment_saw_reset = False
        # Set if the current segment (recursive loop body) itself issues a
        # CALL_ASSEMBLER.  Such a loop (e.g. tak's 3 nested calls) has its
        # base-case stack already reconciled by the call-assembler stack
        # contract, so the is_true base-case shift must NOT also be applied --
        # doing so reads the wrong slot and corrupts the frame.  Only a *pure*
        # FRAME_RESET tail-loop (no in-body CALL_ASSEMBLER, e.g. mb_pass) needs
        # the shift.
        self._segment_saw_call_assembler = False

        self.set_optimizations(optimizations)
        self.setup()

    def setup_condition(self):
        jd = self.jitdriver_sd
        self.conditions = jd.jitdriver.conditions

    def split(self, trace, resumestorage, call_pure_results, token,
              expected_inputargs_count=-1):
        traceiter = trace.get_iter()
        self.token = token
        self.expected_inputargs_count = expected_inputargs_count
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
                if m.numargs() < 2:
                    continue
                payload = m.getarg(1)
                if payload is None or isinstance(payload, Const):
                    continue
                if payload.type == 'v':
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
                mb = ResOperation(rop.same_as_for_type(payload.type),
                                  [payload])
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

        jump_target_descrs = []
        for (info, ops) in self._newopsandinfo:
            for op in ops:
                if op.getopnum() == rop.JUMP:
                    descr = op.getdescr()
                    if descr is not None and descr not in jump_target_descrs:
                        jump_target_descrs.append(descr)

        if len(self._newopsandinfo) > 1:
            kept = [self._newopsandinfo[0]]
            for segidx in range(1, len(self._newopsandinfo)):
                info, ops = self._newopsandinfo[segidx]
                label = info.label_op
                if (info.faildescr is None and label is not None and
                        label.getdescr() not in jump_target_descrs):
                    continue
                kept.append((info, ops))
            self._newopsandinfo = kept

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
                ops = ordered + ops
                self._newopsandinfo[segidx] = (info, ops)

            old_label = info.label_op
            input_boxes = resume_boxes
            if (ops and ops[0].getdescr() is faildescr and
                    ops[0].getopnum() in (rop.GUARD_TRUE,
                                          rop.GUARD_FALSE)):
                cond_box = ops[0].getarg(0)
                ops = ops[1:]
                input_boxes = []
                for b in resume_boxes:
                    if b is not cond_box:
                        input_boxes.append(b)
                self._newopsandinfo[segidx] = (info, ops)

            # Later guards in the bridge must only snapshot boxes that are
            # actually live in this segment: the origin guard's resume boxes
            # plus local definitions/remats.  Stale failargs from the pre-jump
            # path can otherwise shift resume-data slots and mis-type them.
            live = []
            for b in input_boxes:
                live.append(b)
            for d in ordered:
                live.append(d)
            for op in ops:
                if op.is_guard():
                    fa = op.getfailargs()
                    if fa is not None:
                        newfa = []
                        changed = False
                        for b in fa:
                            if b is None or b in live:
                                newfa.append(b)
                            else:
                                changed = True
                        if changed:
                            op.setfailargs(newfa)
                if op.type != 'v':
                    live.append(op)

            info.inputargs = input_boxes
            if old_label is not None:
                info.label_op = ResOperation(rop.LABEL, input_boxes,
                                             old_label.getdescr())
                for idx in range(len(ops)):
                    op = ops[idx]
                    if (op.getopnum() == rop.LABEL and
                            op.getdescr() is old_label.getdescr()):
                        ops[idx] = info.label_op
                        self._newopsandinfo[segidx] = (info, ops)
                        break

        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            local_defs = []
            for op in ops:
                if op.getopnum() != rop.LABEL and op.type != 'v':
                    local_defs.append(op)
            if not local_defs:
                continue
            inputargs = info.inputargs
            new_inputargs = []
            changed = False
            for b in inputargs:
                if b in local_defs:
                    changed = True
                else:
                    new_inputargs.append(b)
            if not changed:
                continue
            old_label = info.label_op
            info.inputargs = new_inputargs
            if old_label is not None:
                info.label_op = ResOperation(rop.LABEL, new_inputargs,
                                             old_label.getdescr())
                for idx in range(len(ops)):
                    op = ops[idx]
                    if (op.getopnum() == rop.LABEL and
                            op.getdescr() is old_label.getdescr()):
                        ops[idx] = info.label_op
                        break
            self._newopsandinfo[segidx] = (info, ops)

        # If a bridge LABEL grew resume-only live-ins (for example a
        # GUARD_TRUE condition box), retarget local jumps that can already see
        # those boxes so the backend receives the same argument shape as the
        # label.
        target_descrs = []
        target_args = []
        for (info, ops) in self._newopsandinfo:
            label = info.label_op
            if label is not None:
                target_descrs.append(label.getdescr())
                target_args.append(label.getarglist())
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            live = []
            for b in info.inputargs:
                live.append(b)
            for op in ops:
                if op.getopnum() == rop.JUMP:
                    descr = op.getdescr()
                    for i in range(len(target_descrs)):
                        if descr is target_descrs[i]:
                            args = target_args[i]
                            can_rewrite = True
                            for b in args:
                                if (b is not None and not isinstance(b, Const)
                                        and b not in live):
                                    can_rewrite = False
                                    break
                            if can_rewrite:
                                op.initarglist(args[:])
                            break
                if op.type != 'v':
                    live.append(op)

        self._fold_constant_extra_inputargs()
        self._trim_extra_inputargs()
        self._rematerialize_label_crossing_defs()
        self._apply_body_contract_shims()
        # self._replace_unused_target_inputargs()
        self._sync_segment_labels_with_inputargs()
        self._prefer_embedded_body_label()
        self._trim_extra_inputargs()
        self._drop_embedded_nonbody_labels()
        self._drop_duplicate_body_target_segments()

    def _drop_duplicate_body_target_segments(self):
        if not self._newopsandinfo:
            return
        body_label = self._newopsandinfo[0][0].label_op
        if body_label is None:
            return
        body_descr = body_label.getdescr()
        kept = [self._newopsandinfo[0]]
        changed = False
        for segidx in range(1, len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            label = info.label_op
            if label is not None and label.getdescr() is body_descr:
                changed = True
                continue
            kept.append((info, ops))
        if changed:
            self._newopsandinfo = kept

    def _prefer_embedded_body_label(self):
        if not self._newopsandinfo:
            return
        info, ops = self._newopsandinfo[0]
        if not ops or ops[0].getopnum() != rop.LABEL:
            return
        label = ops[0]
        if not isinstance(label.getdescr(), TargetToken):
            return
        info.label_op = label
        info.target_token = label.getdescr()
        info.inputargs = label.getarglist()
        self._newopsandinfo[0] = (info, ops)

    def _drop_embedded_nonbody_labels(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            if not ops:
                continue
            newops = []
            changed = False
            for op in ops:
                if (op.getopnum() == rop.LABEL and
                        not isinstance(op.getdescr(), TargetToken)):
                    changed = True
                    continue
                newops.append(op)
            if changed:
                self._newopsandinfo[segidx] = (info, newops)

    def _sync_segment_labels_with_inputargs(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            old_label = info.label_op
            if old_label is None:
                continue
            new_label = ResOperation(rop.LABEL, info.inputargs[:],
                                     old_label.getdescr())
            info.label_op = new_label
            for idx in range(len(ops)):
                op = ops[idx]
                if (op.getopnum() == rop.LABEL and
                        op.getdescr() is old_label.getdescr()):
                    ops[idx] = new_label
                    break
            self._newopsandinfo[segidx] = (info, ops)

    def _drop_unavailable_guard_failargs(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            live = []
            for box in info.inputargs:
                live.append(box)
            for op in ops:
                if op.getopnum() == rop.LABEL:
                    for box in op.getarglist():
                        if box not in live:
                            live.append(box)
                    continue
                if op.is_guard():
                    failargs = op.getfailargs()
                    if failargs is not None:
                        newfailargs = []
                        changed = False
                        for box in failargs:
                            if box is None or box in live:
                                newfailargs.append(box)
                            else:
                                changed = True
                        if changed:
                            op.setfailargs(newfailargs)
                if op.type != 'v':
                    live.append(op)

    def _trim_extra_inputargs(self):
        expected = self.expected_inputargs_count
        if expected < 0:
            return
        for segidx in range(min(len(self._newopsandinfo), 1)):
            info, ops = self._newopsandinfo[segidx]
            if len(info.inputargs) <= expected:
                continue

            inputargs = info.inputargs
            used = []
            for op in ops:
                opnum = op.getopnum()
                if opnum in (rop.LABEL, rop.JUMP):
                    continue
                for i in range(op.numargs()):
                    arg = op.getarg(i)
                    if arg in inputargs and arg not in used:
                        used.append(arg)

            drop_positions = []
            drop_boxes = []
            need_to_drop = len(inputargs) - expected
            for i in range(len(inputargs) - 1, -1, -1):
                if need_to_drop == 0:
                    break
                box = inputargs[i]
                if box in used:
                    continue
                drop_positions.append(i)
                drop_boxes.append(box)
                need_to_drop -= 1
            if need_to_drop != 0:
                continue
            drop_positions.reverse()

            can_drop = True
            for op in ops:
                opnum = op.getopnum()
                if opnum in (rop.LABEL, rop.JUMP):
                    continue
                for i in range(op.numargs()):
                    if op.getarg(i) in drop_boxes:
                        can_drop = False
                        break
                if not can_drop:
                    break
            if not can_drop:
                continue

            info.inputargs = self._drop_positions(inputargs, drop_positions)
            if info.label_op is not None:
                info.label_op = ResOperation(
                    rop.LABEL, info.inputargs[:], info.label_op.getdescr())

            replacement_old = []
            replacement_new = []
            prefix_ops = []
            for box in drop_boxes:
                replacement = self._rematerialized_inputarg_replacement(
                    box, info.inputargs, ops)
                if replacement is not None:
                    replacement_old.append(box)
                    replacement_new.append(replacement)
                    prefix_ops.append(replacement)

            newops = []
            own_descr = None
            if info.label_op is not None:
                own_descr = info.label_op.getdescr()
            for op in ops:
                opnum = op.getopnum()
                if opnum == rop.LABEL:
                    if op.getdescr() is own_descr:
                        op = ResOperation(rop.LABEL,
                                          self._drop_positions(
                                              op.getarglist(),
                                              drop_positions),
                                          op.getdescr())
                elif opnum == rop.JUMP:
                    if op.getdescr() is own_descr:
                        op.initarglist(self._drop_positions(
                            op.getarglist(), drop_positions))
                else:
                    if op.is_guard():
                        failargs = op.getfailargs()
                        if failargs is not None:
                            newfailargs = []
                            for box in failargs:
                                newbox = self._replace_box(
                                    box, replacement_old, replacement_new)
                                if (newbox is box and box in drop_boxes):
                                    newbox = None
                                newfailargs.append(newbox)
                            op.setfailargs(newfailargs)
                newops.append(op)
            if prefix_ops:
                if newops and newops[0].getopnum() == rop.LABEL:
                    newops = [newops[0]] + prefix_ops + newops[1:]
                else:
                    newops = prefix_ops + newops
            self._newopsandinfo[segidx] = (info, newops)

    def _rematerialized_inputarg_replacement(self, box, available, ops):
        candidates = []
        for op in ops:
            if op.getopnum() in (rop.LABEL, rop.JUMP):
                continue
            if op.type != box.type:
                continue
            opnum = op.getopnum()
            if (not rop.has_no_side_effect(opnum) or
                    rop.is_malloc(opnum) or rop.can_raise(opnum)):
                continue
            ok = True
            for i in range(op.numargs()):
                arg = op.getarg(i)
                if (arg is not None and not isinstance(arg, Const) and
                        arg not in available):
                    ok = False
                    break
            if ok:
                candidates.append(op)
        if not candidates:
            return None

        selected = candidates[0]
        for op in ops:
            if (op.getopnum() == rop.GUARD_VALUE and op.numargs() == 2 and
                    op.getarg(0) in candidates and
                    isinstance(op.getarg(1), Const)):
                selected = op.getarg(0)
                break
        return selected.copy_and_change(selected.getopnum(),
                                        args=selected.getarglist(),
                                        descr=selected.getdescr())

    def _select_positions(self, boxes, positions):
        newboxes = []
        for i in positions:
            newboxes.append(boxes[i])
        return newboxes

    def _replace_boxes_with_none(self, boxes, old_boxes):
        newboxes = []
        for box in boxes:
            if box in old_boxes:
                newboxes.append(None)
            else:
                newboxes.append(box)
        return newboxes

    def _drop_unused_target_inputargs(self):
        target_descrs = []
        target_drop_positions = []
        target_drop_boxes = []
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            if info.faildescr is not None or info.label_op is None:
                continue
            inputargs = info.inputargs
            if not inputargs:
                continue
            used = []
            for op in ops:
                opnum = op.getopnum()
                if opnum in (rop.LABEL, rop.JUMP):
                    continue
                for i in range(op.numargs()):
                    arg = op.getarg(i)
                    if arg in inputargs and arg not in used:
                        used.append(arg)
            drop_positions = []
            drop_boxes = []
            for i in range(len(inputargs)):
                if inputargs[i] not in used:
                    drop_positions.append(i)
                    drop_boxes.append(inputargs[i])
            if not drop_positions:
                continue
            info.inputargs = self._drop_positions(inputargs, drop_positions)
            info.label_op = ResOperation(rop.LABEL, info.inputargs[:],
                                         info.label_op.getdescr())
            target_descrs.append(info.label_op.getdescr())
            target_drop_positions.append(drop_positions)
            target_drop_boxes.append(drop_boxes)
            replacements = []
            for box in drop_boxes:
                replacements.append(None)
            newops = []
            for op in ops:
                if op.getopnum() == rop.LABEL:
                    op = ResOperation(rop.LABEL,
                                      self._drop_positions(op.getarglist(),
                                                           drop_positions),
                                      op.getdescr())
                elif (op.getopnum() == rop.GUARD_VALUE and
                        op.numargs() == 2 and
                        isinstance(op.getarg(1), Const)):
                    guarded = op.getarg(0)
                    for i in range(len(drop_boxes)):
                        if (replacements[i] is None and
                                guarded.type == drop_boxes[i].type):
                            replacements[i] = guarded
                            break
                if op.is_guard():
                    failargs = op.getfailargs()
                    if failargs is not None:
                        newfailargs = []
                        for box in failargs:
                            repl = self._replace_box(box, drop_boxes,
                                                     replacements)
                            if repl is not None:
                                newfailargs.append(repl)
                        op.setfailargs(newfailargs)
                newops.append(op)
            self._newopsandinfo[segidx] = (info, newops)

        if not target_descrs:
            return
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            for op in ops:
                if op.getopnum() != rop.JUMP:
                    continue
                descr = op.getdescr()
                for i in range(len(target_descrs)):
                    if descr is target_descrs[i]:
                        op.initarglist(self._drop_positions(
                            op.getarglist(), target_drop_positions[i]))
                        break

    def _drop_positions(self, boxes, positions):
        newboxes = []
        for i in range(len(boxes)):
            if i not in positions:
                newboxes.append(boxes[i])
        return newboxes

    def _replace_unused_target_inputargs(self):
        jump_target_descrs = []
        for (info, ops) in self._newopsandinfo:
            for op in ops:
                if (op.getopnum() == rop.JUMP and
                        op.getdescr() not in jump_target_descrs):
                    jump_target_descrs.append(op.getdescr())
        target_descrs = []
        target_old_boxes = []
        target_new_boxes = []
        for segidx in range(1, len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            if info.label_op is None:
                continue
            if info.label_op.getdescr() not in jump_target_descrs:
                continue
            inputargs = info.inputargs
            if len(inputargs) <= 1:
                continue
            used = []
            for op in ops:
                opnum = op.getopnum()
                if opnum in (rop.LABEL, rop.JUMP):
                    continue
                for i in range(op.numargs()):
                    arg = op.getarg(i)
                    if arg in inputargs and arg not in used:
                        used.append(arg)
            old_boxes = []
            new_boxes = []
            for box in inputargs:
                if box in used:
                    continue
                replacement = self._fresh_inputarg_for_type(box.type)
                if replacement is not None:
                    old_boxes.append(box)
                    new_boxes.append(replacement)
            if not old_boxes:
                continue
            info.inputargs = [self._replace_box(box, old_boxes, new_boxes)
                              for box in inputargs]
            info.label_op = ResOperation(rop.LABEL, info.inputargs[:],
                                         info.label_op.getdescr())
            target_descrs.append(info.label_op.getdescr())
            target_old_boxes.append(old_boxes)
            target_new_boxes.append(new_boxes)
            for op in ops:
                if op.getopnum() == rop.LABEL:
                    op.initarglist([self._replace_box(box, old_boxes,
                                                      new_boxes)
                                    for box in op.getarglist()])
                elif op.is_guard():
                    failargs = op.getfailargs()
                    if failargs is not None:
                        op.setfailargs(self._drop_boxes(failargs, old_boxes))
            self._newopsandinfo[segidx] = (info, ops)

        if not target_descrs:
            return
        for (info, ops) in self._newopsandinfo:
            for op in ops:
                if op.getopnum() != rop.JUMP:
                    continue
                descr = op.getdescr()
                for i in range(len(target_descrs)):
                    if descr is target_descrs[i]:
                        op.initarglist([self._replace_box(
                            box, target_old_boxes[i], target_new_boxes[i])
                            for box in op.getarglist()])
                        break

    def _fresh_inputarg_for_type(self, typ):
        if typ == 'r':
            return InputArgRef()
        if typ == 'i':
            return InputArgInt()
        if typ == 'f':
            return InputArgFloat()
        return None

    def _fold_constant_extra_inputargs(self):
        if not self._newopsandinfo:
            return
        first_info, first_ops = self._newopsandinfo[0]
        inputargs = first_info.inputargs
        if len(inputargs) <= 1:
            return

        old_boxes = []
        new_consts = []
        for op in first_ops:
            if op.getopnum() == rop.LABEL:
                break
            if (op.getopnum() == rop.GUARD_VALUE and op.numargs() == 2 and
                    not isinstance(op.getarg(0), Const) and
                    op.getarg(0).type == 'r' and
                    isinstance(op.getarg(1), Const)):
                box = op.getarg(0)
                if box not in old_boxes:
                    old_boxes.append(box)
                    new_consts.append(op.getarg(1))
                    break
        if not old_boxes:
            return

        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            info.inputargs = self._drop_boxes_or_matching_suffix(
                info.inputargs, old_boxes)
            if info.label_op is not None:
                newargs = self._drop_boxes_or_matching_suffix(
                    info.label_op.getarglist(), old_boxes)
                info.label_op = ResOperation(rop.LABEL, newargs,
                                             info.label_op.getdescr())

            newops = []
            for op in ops:
                if (op.getopnum() == rop.GUARD_VALUE and op.numargs() == 2 and
                        op.getarg(0) in old_boxes):
                    continue
                if op.getopnum() == rop.LABEL:
                    args = self._drop_boxes_or_matching_suffix(
                        op.getarglist(), old_boxes)
                    op = ResOperation(rop.LABEL, args, op.getdescr())
                elif op.getopnum() == rop.JUMP:
                    op.initarglist(self._drop_boxes_or_matching_suffix(
                        op.getarglist(), old_boxes))
                else:
                    for i in range(op.numargs()):
                        arg = op.getarg(i)
                        newarg = self._replace_box(arg, old_boxes, new_consts)
                        if newarg is not arg:
                            op.setarg(i, newarg)
                    if op.is_guard():
                        failargs = op.getfailargs()
                        if failargs is not None:
                            op.setfailargs(self._drop_boxes(failargs,
                                                            old_boxes))
                newops.append(op)
            self._newopsandinfo[segidx] = (info, newops)

    def _drop_boxes(self, boxes, drop_boxes):
        newboxes = []
        for box in boxes:
            if box not in drop_boxes:
                newboxes.append(box)
        return newboxes

    def _drop_boxes_or_matching_suffix(self, boxes, drop_boxes):
        newboxes = self._drop_boxes(boxes, drop_boxes)
        missing = len(drop_boxes) - (len(boxes) - len(newboxes))
        while missing > 0 and newboxes:
            dropbox = drop_boxes[missing - 1]
            found = -1
            for i in range(len(newboxes) - 1, -1, -1):
                if newboxes[i].type == dropbox.type:
                    found = i
                    break
            if found < 0:
                break
            del newboxes[found]
            missing -= 1
        return newboxes

    def _replace_box(self, box, old_boxes, new_boxes):
        for i in range(len(old_boxes)):
            if box is old_boxes[i]:
                return new_boxes[i]
        return box

    def _replace_op_boxes(self, op, old_boxes, new_boxes):
        for i in range(op.numargs()):
            arg = op.getarg(i)
            newarg = self._replace_box(arg, old_boxes, new_boxes)
            if newarg is not arg:
                op.setarg(i, newarg)
        if op.is_guard():
            failargs = op.getfailargs()
            if failargs is not None:
                newfailargs = []
                changed = False
                for arg in failargs:
                    newarg = self._replace_box(arg, old_boxes, new_boxes)
                    newfailargs.append(newarg)
                    if newarg is not arg:
                        changed = True
                if changed:
                    op.setfailargs(newfailargs)

    def _rematerialize_label_crossing_defs(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            idx = 0
            while idx < len(ops):
                if ops[idx].getopnum() != rop.LABEL:
                    idx += 1
                    continue

                end = idx + 1
                while end < len(ops) and ops[end].getopnum() != rop.LABEL:
                    end += 1

                def_boxes = []
                def_ops = []
                for op in ops[:idx]:
                    if op.getopnum() != rop.LABEL and op.type != 'v':
                        def_boxes.append(op)
                        def_ops.append(op)
                if not def_boxes:
                    idx += 1
                    continue

                label_args = ops[idx].getarglist()
                defined = label_args[:]
                needed = []
                for op in ops[idx + 1:end]:
                    for i in range(op.numargs()):
                        box = op.getarg(i)
                        if box is None or isinstance(box, Const):
                            continue
                        if box in defined or box in needed:
                            continue
                        if box in def_boxes:
                            needed.append(box)
                    if op.type != 'v':
                        defined.append(op)
                if not needed:
                    idx += 1
                    continue

                remat = []
                worklist = needed[:]
                while worklist:
                    box = worklist.pop()
                    if box in label_args or box in remat:
                        continue
                    dop = None
                    for i in range(len(def_boxes)):
                        if def_boxes[i] is box:
                            dop = def_ops[i]
                            break
                    if dop is None:
                        continue
                    opnum = dop.getopnum()
                    if (not rop.has_no_side_effect(opnum) or
                            rop.is_malloc(opnum) or rop.can_raise(opnum)):
                        continue
                    remat.append(dop)
                    for i in range(dop.numargs()):
                        arg = dop.getarg(i)
                        if arg is None or isinstance(arg, Const):
                            continue
                        if arg not in label_args:
                            worklist.append(arg)

                ordered = []
                for op in def_ops:
                    if op in remat:
                        ordered.append(op)
                if not ordered:
                    idx += 1
                    continue

                safe = True
                for op in ordered:
                    for i in range(op.numargs()):
                        arg = op.getarg(i)
                        if arg is None or isinstance(arg, Const):
                            continue
                        if arg not in label_args and arg not in ordered:
                            safe = False
                            break
                    if not safe:
                        break
                if not safe:
                    idx += 1
                    continue

                old_boxes = []
                new_boxes = []
                clones = []
                for op in ordered:
                    args = []
                    for i in range(op.numargs()):
                        arg = op.getarg(i)
                        args.append(self._replace_box(arg, old_boxes,
                                                      new_boxes))
                    clone = op.copy_and_change(op.getopnum(), args=args)
                    old_boxes.append(op)
                    new_boxes.append(clone)
                    clones.append(clone)

                insert_count = len(clones)
                ops = ops[:idx + 1] + clones + ops[idx + 1:]
                end += insert_count
                for opidx in range(idx + 1 + insert_count, end):
                    self._replace_op_boxes(ops[opidx], old_boxes, new_boxes)
                self._newopsandinfo[segidx] = (info, ops)
                idx = end

    def _body_contract_for_guard(self, faildescr):
        empty = (None, -1, -1, None, None, None, -1)
        body_ops = None
        body_inputargs = None
        guard_failargs = None
        for (bi, bops) in self._newopsandinfo:
            for bop in bops:
                if bop.is_guard() and bop.getdescr() is faildescr:
                    body_ops = bops
                    body_inputargs = bi.inputargs
                    guard_failargs = bop.getfailargs()
                    break
            if body_ops is not None:
                break
        if body_ops is None:
            return empty

        frame_box = None
        stack_box = None
        stack_field_descr = None
        sp_field_descr = None
        arr_descr = None
        callee_sp = -1
        arg_slot = -1
        preferred_frames = []
        if guard_failargs is not None:
            for box in guard_failargs:
                if (box is not None and not isinstance(box, Const) and
                        box.type == 'r' and box not in preferred_frames):
                    preferred_frames.append(box)

        for i in range(len(body_ops)):
            op = body_ops[i]
            if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1):
                if (body_inputargs is not None and
                        op.getarg(0) not in body_inputargs):
                    continue
                if (preferred_frames and
                        op.getarg(0) not in preferred_frames):
                    continue
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
            return empty

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
            return empty

        for op in body_ops:
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getarg(0) is stack_box and
                    isinstance(op.getarg(1), ConstInt)):
                arg_slot = op.getarg(1).getint()
                arr_descr = op.getdescr()
                break
        if arg_slot < 0 or arr_descr is None:
            return empty

        # Highest stackpos the recursive body wrote.  The deferred base-case
        # branch is recorded inline *after* the body's FRAME_RESET, which (being
        # a @reset_helper) does not apply its stackpos:=callee_sp to the
        # recorder's frame during tracing -- so the base-case stack reads were
        # baked at this loop-body-max stackpos rather than at callee_sp.
        body_max_sp = callee_sp
        for op in body_ops:
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr and
                    isinstance(op.getarg(1), ConstInt)):
                v = op.getarg(1).getint()
                if v > body_max_sp:
                    body_max_sp = v

        return (frame_box, callee_sp, arg_slot,
                stack_field_descr, sp_field_descr, arr_descr, body_max_sp)

    def _first_bridge_stack_read_slot(self, ops, stack_box):
        for op in ops:
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getarg(0) is stack_box and
                    isinstance(op.getarg(1), ConstInt)):
                return op.getarg(1).getint()
        return -1

    def _first_bridge_stack_read_slot_for_descr(self, ops, frame_box,
                                                stack_field_descr, arr_descr):
        stack_boxes = []
        for op in ops:
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
                continue
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getdescr() is arr_descr and
                    op.getarg(0) in stack_boxes and
                    isinstance(op.getarg(1), ConstInt)):
                return op.getarg(1).getint()
        return -1

    def _fix_base_case_return_slots(self, ops, callee_sp, body_max_sp,
                                    arr_descr):
        # The deferred base-case (guard-exit) branch is recorded inline after
        # the recursive body's reset helper.  That helper does not apply its
        # stackpos:=callee_sp to the recorder's frame during tracing, so the
        # base case's stack reads were baked at the loop-body max stackpos
        # (body_max_sp) instead of the loop-header stackpos (callee_sp).  When
        # the loop-exit guard actually fails the frame is at the loop-header
        # state (stackpos == callee_sp), so a base-case stack read that should
        # read relative to callee_sp instead reads relative to body_max_sp.
        #
        # Every base-case stack read is offset by the same constant
        # (body_max_sp - callee_sp), because the base-case ops are identical in
        # both framings and only differ by the base stackpos they start from.
        # Shift the const stack-read indices that sit in the grown region
        # (>= callee_sp) back down so the bridge reads the loop-header slots.
        shift = body_max_sp - callee_sp
        if shift <= 0:
            return ops
        has_finish = False
        for op in ops:
            if op.getopnum() == rop.FINISH:
                has_finish = True
                break
        if not has_finish:
            return ops
        for op in ops:
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and op.numargs() == 2 and
                    op.getdescr() is arr_descr and
                    isinstance(op.getarg(1), ConstInt)):
                slot = op.getarg(1).getint()
                if slot >= callee_sp and slot - shift >= 0:
                    op.setarg(1, ConstInt(slot - shift))
        return ops

    def _guard_consumed_condition_slots(self, faildescr):
        for (bi, bops) in self._newopsandinfo:
            for idx in range(len(bops)):
                guard = bops[idx]
                if not (guard.is_guard() and guard.getdescr() is faildescr):
                    continue
                if guard.numargs() == 0:
                    return 0
                cond_box = guard.getarg(0)
                for prev in range(idx - 1, -1, -1):
                    op = bops[prev]
                    if op is cond_box:
                        if self._call_role_from_op(op) == JitInterp.CONDITION:
                            return 1
                        return 0
                return 0
        return 0

    def _rebase_bridge_stack_constants(self, ops, frame_box,
                                       stack_field_descr, sp_field_descr,
                                       arr_descr, delta):
        if delta == 0:
            return ops
        stack_boxes = []
        sp_boxes = []
        for op in ops:
            if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr):
                sp_boxes.append(op)
                continue
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
                continue
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr and
                    isinstance(op.getarg(1), ConstInt)):
                op.setarg(1, ConstInt(op.getarg(1).getint() + delta))
                continue
            if (op.getopnum() == rop.GUARD_VALUE and op.numargs() == 2 and
                    op.getarg(0) in sp_boxes and
                    isinstance(op.getarg(1), ConstInt)):
                op.setarg(1, ConstInt(op.getarg(1).getint() + delta))
                continue
            if (op.getopnum() in (rop.GETARRAYITEM_GC_R,
                                  rop.SETARRAYITEM_GC) and
                    op.numargs() >= 2 and op.getdescr() is arr_descr and
                    op.getarg(0) in stack_boxes and
                    isinstance(op.getarg(1), ConstInt)):
                op.setarg(1, ConstInt(op.getarg(1).getint() + delta))
        return ops

    def _apply_body_contract_shims(self):
        for segidx in range(len(self._newopsandinfo)):
            info, ops = self._newopsandinfo[segidx]
            faildescr = info.faildescr
            if not isinstance(faildescr, compile.AbstractResumeGuardDescr):
                continue

            (frame_box, callee_sp, arg_slot, stack_field_descr,
             sp_field_descr, arr_descr, body_max_sp) = \
                self._body_contract_for_guard(faildescr)
            if frame_box is None:
                continue

            has_call_asm = False
            for op in ops:
                if rop.is_call_assembler(op.getopnum()):
                    has_call_asm = True
                    break

            if ops and ops[-1].getopnum() == rop.JUMP:
                if not has_call_asm:
                    consumed = self._guard_consumed_condition_slots(faildescr)
                    if consumed:
                        ops = self._rebase_bridge_stack_constants(
                            ops, frame_box, stack_field_descr, sp_field_descr,
                            arr_descr, consumed)
                        self._newopsandinfo[segidx] = (info, ops)
                continue

            # if not has_call_asm:
            #     ops = self._drop_bridge_recorded_stack_bookkeeping(
            #         ops, frame_box, stack_field_descr, sp_field_descr, arr_descr)
            ops = self._trim_base_case_zero_return_bridge(
                ops, frame_box, callee_sp, sp_field_descr)

            if not has_call_asm:
                consumed = self._guard_consumed_condition_slots(faildescr)
                if consumed:
                    ops = self._rebase_bridge_stack_constants(
                        ops, frame_box, stack_field_descr, sp_field_descr,
                        arr_descr, consumed)
                ops = self._fix_base_case_return_slots(
                    ops, callee_sp, body_max_sp, arr_descr)
                self._newopsandinfo[segidx] = (info, ops)
                continue

            bridge_arg_slot = -1
            for op in ops:
                if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                        op.numargs() == 2 and
                        isinstance(op.getarg(1), ConstInt)):
                    bridge_arg_slot = op.getarg(1).getint()
                    break
            if bridge_arg_slot < 0:
                continue
            ops = self._shift_bridge_stack_slots(
                ops, frame_box, stack_field_descr, sp_field_descr, arr_descr,
                bridge_arg_slot, arg_slot)
            ops = self._apply_call_assembler_stack_contract(
                ops, frame_box, stack_field_descr, sp_field_descr, arr_descr)
            ops = self._fix_frame_push_slots(
                ops, frame_box, stack_field_descr, sp_field_descr, arr_descr)
            self._newopsandinfo[segidx] = (info, ops)

    def _call_assembler_argnum(self, ops, call_op, sp_field_descr):
        if call_op.numargs() == 0:
            return -1
        callee_frame = call_op.getarg(0)
        callee_sp = -1
        for op in ops:
            if op is call_op:
                break
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is callee_frame and
                    op.getdescr() is sp_field_descr and
                    isinstance(op.getarg(1), ConstInt)):
                callee_sp = op.getarg(1).getint()
        if callee_sp < 2:
            return -1
        return callee_sp - 2

    def _const_stackpos_before(self, ops, call_op, frame_box, sp_field_descr):
        stackpos = -1
        for op in ops:
            if op is call_op:
                break
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr and
                    isinstance(op.getarg(1), ConstInt)):
                stackpos = op.getarg(1).getint()
        return stackpos

    def _caller_result_push_call(self, op, frame_box, result_box):
        if result_box is None or result_box.type == 'v':
            return False
        if op is None or op.type != 'v':
            return False
        opnum = op.getopnum()
        if not (rop.is_plain_call(opnum) or rop.is_call_may_force(opnum)):
            return False
        if op.numargs() < 4:
            return False
        if op.getarg(1) is not frame_box or op.getarg(2) is not result_box:
            return False
        return isinstance(op.getarg(op.numargs() - 1), ConstInt)

    def _call_assembler_stack_contract_ops(self, frame_box, result_box, argnum,
                                           stack_field_descr, sp_field_descr,
                                           arr_descr):
        if argnum < 0:
            return []
        old_sp = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                              descr=sp_field_descr)
        after_drop = ResOperation(rop.INT_SUB, [old_sp, ConstInt(argnum)])
        set_drop_sp = ResOperation(rop.SETFIELD_GC, [frame_box, after_drop],
                                   descr=sp_field_descr)
        ops = [old_sp, after_drop, set_drop_sp]
        if result_box.type == 'v':
            return ops
        stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                                 descr=stack_field_descr)
        resume_slot = ResOperation(rop.INT_SUB, [after_drop, ConstInt(1)])
        set_resume = ResOperation(rop.SETARRAYITEM_GC,
                                  [stack_box, resume_slot, result_box],
                                  descr=arr_descr)
        set_result = ResOperation(rop.SETARRAYITEM_GC,
                                  [stack_box, after_drop, result_box],
                                  descr=arr_descr)
        after_push = ResOperation(rop.INT_ADD, [after_drop, ConstInt(1)])
        set_push_sp = ResOperation(rop.SETFIELD_GC, [frame_box, after_push],
                                   descr=sp_field_descr)
        ops.extend([stack_box, resume_slot, set_resume, set_result])
        ops.extend([after_push, set_push_sp])
        return ops

    def _apply_call_assembler_stack_contract(self, ops, frame_box,
                                             stack_field_descr,
                                             sp_field_descr, arr_descr):
        newops = []
        pending_result = None
        pending_argnum = -1
        for op in ops:
            if rop.is_call_assembler(op.getopnum()):
                pending_result = op
                pending_argnum = self._call_assembler_argnum(
                    ops, op, sp_field_descr)
                newops.append(op)
                continue
            if self._caller_result_push_call(op, frame_box, pending_result):
                effect = self._call_assembler_stack_contract_ops(
                    frame_box, pending_result, pending_argnum,
                    stack_field_descr, sp_field_descr, arr_descr)
                if effect:
                    newops.extend(effect)
                    pending_result = None
                    pending_argnum = -1
                    continue
            newops.append(op)
        return newops

    def _fix_frame_push_slots(self, ops, frame_box, stack_field_descr,
                              sp_field_descr, arr_descr):
        stack_boxes = []
        last_const_sp = -1
        for op in ops:
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
                continue
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr):
                if isinstance(op.getarg(1), ConstInt):
                    last_const_sp = op.getarg(1).getint()
                else:
                    last_const_sp = -1
                continue
            if (op.getopnum() == rop.SETARRAYITEM_GC and op.numargs() == 3 and
                    op.getdescr() is arr_descr and op.getarg(0) in stack_boxes
                    and isinstance(op.getarg(1), ConstInt)):
                slot = op.getarg(1).getint()
                if last_const_sp >= 1 and slot <= last_const_sp - 2:
                    op.setarg(1, ConstInt(slot + 1))
        return ops

    def _shift_bridge_stack_slots(self, ops, frame_box, stack_field_descr,
                                  sp_field_descr, arr_descr,
                                  bridge_arg_slot, body_arg_slot):
        shift = bridge_arg_slot - body_arg_slot
        stack_boxes = []
        newops = []
        after_call_assembler = False
        last_const_sp = -1
        pending_result = None
        pending_result_slot = -1
        for op in ops:
            if (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is sp_field_descr):
                if isinstance(op.getarg(1), ConstInt):
                    last_const_sp = op.getarg(1).getint()
                else:
                    last_const_sp = -1
            if rop.is_call_assembler(op.getopnum()):
                after_call_assembler = True
                newops.append(op)
                argnum = self._call_assembler_argnum(ops, op, sp_field_descr)
                result_slot = last_const_sp - argnum - 1
                if argnum >= 0 and result_slot >= 0 and op.type != 'v':
                    pending_result = op
                    pending_result_slot = result_slot
                continue
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
                newops.append(op)
                if pending_result is not None:
                    newops.append(ResOperation(
                        rop.SETARRAYITEM_GC,
                        [op, ConstInt(pending_result_slot), pending_result],
                        descr=arr_descr))
                    pending_result = None
                    pending_result_slot = -1
                continue
                continue
            if (op.getopnum() in (rop.GETARRAYITEM_GC_R,
                                  rop.SETARRAYITEM_GC) and
                    op.numargs() >= 2 and op.getdescr() is arr_descr and
                    op.getarg(0) in stack_boxes and
                    isinstance(op.getarg(1), ConstInt)):
                slot = op.getarg(1).getint()
                if shift > 0 and slot >= bridge_arg_slot:
                    extra_shift = 0
                    if after_call_assembler and slot == bridge_arg_slot:
                        extra_shift = 1
                    new_slot = slot - shift - extra_shift
                    op.setarg(1, ConstInt(new_slot))
            newops.append(op)
        return newops

    def _shift_bridge_stack_read_slots(self, ops, frame_box, stack_field_descr,
                                       sp_field_descr, arr_descr,
                                       bridge_arg_slot, body_arg_slot):
        shift = bridge_arg_slot - body_arg_slot
        if shift <= 0:
            return ops
        stack_boxes = []
        for op in ops:
            if (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box and
                    op.getdescr() is stack_field_descr):
                stack_boxes.append(op)
                continue
            if (op.getopnum() == rop.GETARRAYITEM_GC_R and
                    op.numargs() == 2 and op.getdescr() is arr_descr and
                    op.getarg(0) in stack_boxes and
                    isinstance(op.getarg(1), ConstInt)):
                slot = op.getarg(1).getint()
                if slot >= bridge_arg_slot:
                    op.setarg(1, ConstInt(slot - shift))
        return ops

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

    def _trim_base_case_zero_return_bridge(self, ops, frame_box, callee_sp,
                                           sp_field_descr):
        finish_op = None
        leave_op = None
        for op in ops:
            if op.getopnum() == rop.LEAVE_PORTAL_FRAME:
                leave_op = op
            elif op.getopnum() == rop.FINISH:
                finish_op = op
        if finish_op is not None and leave_op is not None and (
                finish_op.numargs() == 1):
            finish_arg = finish_op.getarg(0)
            finish_def = -1
            last_keep = -1
            for i in range(len(ops)):
                op = ops[i]
                if op is finish_arg:
                    finish_def = i
                    last_keep = i
                elif (finish_def >= 0 and op.is_guard() and
                        op.numargs() > 0 and op.getarg(0) is finish_arg):
                    last_keep = i
            if last_keep >= 0:
                newops = []
                for i in range(last_keep + 1):
                    if ops[i].getopnum() != rop.LEAVE_PORTAL_FRAME:
                        newops.append(ops[i])
                newops.append(leave_op)
                newops.append(finish_op)
                return newops

        # DISABLED: this shim rewrote the base-case bridge to finish(W_IntObject(0)),
        # which is only correct when the base case really returns the shallow
        # placeholder 0.  With the frame restored to real values on loop entry
        # (see tla._tier1_confirm_enter_jit), the base case returns the real
        # popped value (e.g. the accumulator), so the bridge must keep its real
        # finish(<popped slot>) instead of a hardcoded zero.
        return ops
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
        if not self._is_frame_pop_call(finish_arg):
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
                box = self._dmp_pcbox(op)
                if box.getint() in self.token_map.keys():
                    token = self._get_token(box.getint())
                elif len(self.token_map) == 0 and self.token is not None:
                    token = self.token
                else:
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
                    self._reset_after_is_true_shift()
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

    def _emit_pop_stack_read(self, op):
        if op.numargs() < 2:
            return None
        frame_box = op.getarg(1)
        sp_field_descr, stack_field_descr, arr_descr = \
            self._frame_stack_descriptors(frame_box)
        if sp_field_descr is None or stack_field_descr is None or arr_descr is None:
            return None
        sp_box = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                              descr=sp_field_descr)
        index_box = ResOperation(rop.INT_SUB, [sp_box, ConstInt(1)])
        stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                                 descr=stack_field_descr)
        read_box = ResOperation(rop.GETARRAYITEM_GC_R, [stack_box, index_box],
                                descr=arr_descr)
        self._newoperations.extend([sp_box, index_box, stack_box, read_box])
        return read_box

    def _emit_frame_drop(self, op):
        if op.numargs() < 3:
            return False
        frame_box = op.getarg(1)
        n_box = op.getarg(2)
        if not isinstance(n_box, ConstInt):
            return False
        sp_field_descr, stack_field_descr, arr_descr = \
            self._frame_stack_descriptors(frame_box)
        if sp_field_descr is None:
            return False
        old_sp = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                              descr=sp_field_descr)
        new_sp = ResOperation(rop.INT_SUB, [old_sp, n_box])
        set_sp = ResOperation(rop.SETFIELD_GC, [frame_box, new_sp],
                              descr=sp_field_descr)
        self._newoperations.extend([old_sp, new_sp, set_sp])
        return True

    def _emit_frame_push(self, op):
        if op.numargs() < 3:
            return False
        frame_box = op.getarg(1)
        value_box = op.getarg(2)
        return self._emit_frame_push_box(frame_box, value_box)

    def _emit_frame_push_box(self, frame_box, value_box):
        sp_field_descr, stack_field_descr, arr_descr = \
            self._frame_stack_descriptors(frame_box)
        if sp_field_descr is None or stack_field_descr is None or arr_descr is None:
            return False
        return self._emit_frame_push_box_with_descrs(
            frame_box, value_box, sp_field_descr, stack_field_descr, arr_descr)

    def _emit_frame_push_box_with_descrs(self, frame_box, value_box,
                                         sp_field_descr, stack_field_descr,
                                         arr_descr):
        old_sp = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                              descr=sp_field_descr)
        stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                                 descr=stack_field_descr)
        set_value = ResOperation(rop.SETARRAYITEM_GC,
                                 [stack_box, old_sp, value_box],
                                 descr=arr_descr)
        new_sp = ResOperation(rop.INT_ADD, [old_sp, ConstInt(1)])
        set_sp = ResOperation(rop.SETFIELD_GC, [frame_box, new_sp],
                              descr=sp_field_descr)
        self._newoperations.extend([old_sp, stack_box, set_value,
                                    new_sp, set_sp])
        return True

    def _has_stack_push_of(self, frame_box, value_box):
        sp_boxes = []
        stack_boxes = []
        for op in self._newoperations:
            if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1 and
                    op.getarg(0) is frame_box):
                sp_boxes.append(op)
            elif (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box):
                stack_boxes.append(op)
            elif (op.getopnum() == rop.SETARRAYITEM_GC and op.numargs() == 3 and
                    op.getarg(2) is value_box):
                for stack_box in stack_boxes:
                    if op.getarg(0) is stack_box:
                        return True
        return False

    def _last_materialized_pop_read(self):
        if len(self._newoperations) < 4:
            return None
        read = self._newoperations[-1]
        if read.getopnum() == rop.GETARRAYITEM_GC_R:
            return read
        return None

    def _maybe_rewrite_ret_pop_result(self, op):
        numargs = op.numargs()
        ret_box = op.getarg(numargs - 1)
        read = self._materialized_pop_read_for_retbox(ret_box)
        if read is not None:
            op.setarg(numargs - 1, read)

    def _materialized_pop_read_for_retbox(self, ret_box):
        pop_index = -1
        for idx in range(len(self._newoperations)):
            if self._newoperations[idx].same_box(ret_box):
                pop_index = idx
                break
        if pop_index < 0:
            return None
        read = _find_materialized_pop_read(self._newoperations, pop_index)
        return read

    def _adjacent_materialized_read_for_retbox(self, ret_box):
        for idx in range(1, len(self._newoperations)):
            if self._newoperations[idx].same_box(ret_box):
                read = self._newoperations[idx - 1]
                if read.getopnum() == rop.GETARRAYITEM_GC_R:
                    return read
                return None
        return None

    def _ret_pop_read(self):
        pop_index = -1
        for idx in range(len(self._newoperations) - 1, -1, -1):
            if self._is_frame_stack_read_call(self._newoperations[idx]):
                pop_index = idx
                break
        if pop_index < 0:
            return None
        return _find_materialized_pop_read(self._newoperations, pop_index)

    def _is_frame_pop_call(self, op):
        return self._call_role_from_op(op) == JitInterp.POP

    def _is_frame_stack_read_call(self, op):
        role = self._call_role_from_op(op)
        return role == JitInterp.POP or role == JitInterp.POP_RAW

    def _call_role_from_op(self, op):
        if op is None:
            return JitInterp.NONE
        opnum = op.getopnum()
        if not (rop.is_plain_call(opnum) or rop.is_call_may_force(opnum)):
            return JitInterp.NONE
        arg0 = op.getarg(0)
        if not isinstance(arg0, ConstInt):
            return JitInterp.NONE
        adr = cast_int_to_adr(arg0.getint())
        if not we_are_translated() and not hasattr(
                self.metainterp_sd, 'get_annotation_from_address'):
            return JitInterp.NONE
        role = self.metainterp_sd.get_annotation_from_address(adr)
        return role if role else JitInterp.NONE

    def _emit2(self, op):
        """Pass-2 structural emit: rewrite the shallow-tracing flag arg,
        detect the slow-path emit_ptr_eq marker and run the deferred
        guard-marking, then append the (already optimized) op to the current
        segment without re-running the optimizer chain."""
        opnum = op.getopnum()
        if rop.is_call_assembler(opnum):
            # An in-body CALL_ASSEMBLER marks this as a non-pure tail-loop
            # (e.g. tak); its base case must not get the is_true shift.
            self._segment_saw_call_assembler = True
        if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
            numargs = op.numargs()
            name = self._get_name_from_op(op)
            role = self._call_role_from_op(op)
            if endswith(name, mark.CALL_ASSEMBLER):
                self._segment_saw_call_assembler = True
            if startswith(name, "handler_"):
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
            elif role == JitInterp.POP:
                self._emit_pop_stack_read(op)
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
            elif role == JitInterp.POP_RAW:
                self._emit_pop_stack_read(op)
            elif role == JitInterp.DROP:
                if self._emit_frame_drop(op):
                    return
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
            elif role == JitInterp.PUSH:
                if self._emit_frame_push(op):
                    return
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
            elif role == JitInterp.PUSH_RAW:
                if self._emit_frame_push(op):
                    return
            elif (role == JitInterp.CONDITION or
                  role == JitInterp.RESET or
                  role == JitInterp.RET):
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
                if role == JitInterp.CONDITION:
                    self._last_condition_call = op
                elif role == JitInterp.RESET:
                    # FRAME_RESET: this segment is a tail-loop, so a pending
                    # condition guard's base case is the deferred (off-by-one)
                    # branch.  Confirmed at emit_jump (the tail-loop close).
                    self._segment_saw_reset = True
            if endswith(name, "emit_ptr_eq"):
                self._slow_path_emit_ptr_eq = op
        elif opnum in (rop.GUARD_VALUE, rop.GUARD_TRUE, rop.GUARD_FALSE):
            # A GUARD_TRUE/FALSE reading the CONDITION helper's result is the
            # loop-continue guard.  Remember it; only once we confirm this is a
            # FRAME_RESET tail-loop (at emit_jump) do we stamp tcg_cond_pop so
            # compile_and_attach shifts its deferred base-case bridge down.
            if (opnum in (rop.GUARD_TRUE, rop.GUARD_FALSE) and
                    self._last_condition_call is not None and
                    op.numargs() >= 1 and
                    op.getarg(0) is self._last_condition_call):
                self._pending_cond_guard_descr = op.getdescr()
                self._last_condition_call = None
            self._mark_guard(op)
        self._newoperations.append(op)

    def _shift_const_arg_down(self, op, index):
        arg = op.getarg(index)
        if isinstance(arg, ConstInt):
            op.setarg(index, ConstInt(arg.getint() - 1))

    def _reset_after_is_true_shift(self):
        self._pending_is_true_frame = None
        self._shift_after_is_true_frame = None
        self._shift_after_is_true_spdescr = None
        self._shift_after_is_true_stackdescr = None
        self._shift_after_is_true_stackboxes = []

    def _activate_after_is_true_shift(self, frame_box):
        self._shift_after_is_true_frame = frame_box
        self._shift_after_is_true_spdescr = None
        self._shift_after_is_true_stackdescr = None
        self._shift_after_is_true_stackboxes = []
        for prev in self._newoperations:
            if (prev.getopnum() == rop.GETFIELD_GC_I and
                    prev.numargs() == 1 and prev.getarg(0) is frame_box):
                self._shift_after_is_true_spdescr = prev.getdescr()
            elif (prev.getopnum() == rop.GETFIELD_GC_R and
                    prev.numargs() == 1 and prev.getarg(0) is frame_box):
                self._shift_after_is_true_stackdescr = prev.getdescr()
                self._shift_after_is_true_stackboxes.append(prev)

    def _adjust_after_is_true_pop(self, op):
        frame_box = self._shift_after_is_true_frame
        opnum = op.getopnum()
        if (opnum == rop.GETFIELD_GC_I and op.numargs() == 1 and
                op.getarg(0) is frame_box):
            self._shift_after_is_true_spdescr = op.getdescr()
            return
        if (opnum == rop.GETFIELD_GC_R and op.numargs() == 1 and
                op.getarg(0) is frame_box):
            self._shift_after_is_true_stackdescr = op.getdescr()
            self._shift_after_is_true_stackboxes.append(op)
            return
        spdescr = self._shift_after_is_true_spdescr
        if spdescr is not None:
            if (opnum == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box and op.getdescr() is spdescr):
                self._shift_const_arg_down(op, 1)
                return
            if (opnum == rop.GUARD_VALUE and op.numargs() == 2 and
                    op.getarg(0).getopnum() == rop.GETFIELD_GC_I and
                    op.getarg(0).numargs() == 1 and
                    op.getarg(0).getarg(0) is frame_box and
                    op.getarg(0).getdescr() is spdescr):
                self._shift_const_arg_down(op, 1)
                return
        if opnum in (rop.GETARRAYITEM_GC_R, rop.SETARRAYITEM_GC):
            for stack_box in self._shift_after_is_true_stackboxes:
                if op.numargs() >= 2 and op.getarg(0) is stack_box:
                    self._shift_const_arg_down(op, 1)
                    return

    def _mark_guard(self, op):
        if self._check_if_guard_marked(op):
            if op.getopnum() not in (rop.GUARD_TRUE, rop.GUARD_FALSE):
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
        original_retbox = op.getarg(numargs - 1)
        self._maybe_rewrite_ret_pop_result(op)
        retbox = op.getarg(numargs - 1)
        pop_read = self._adjacent_materialized_read_for_retbox(retbox)
        if pop_read is None:
            pop_read = self._materialized_pop_read_for_retbox(retbox)
        if pop_read is None:
            pop_read = self._ret_pop_read()
        if pop_read is not None:
            retbox = pop_read

        if result_type == history.VOID:
            exits = []
            finishtoken = sd.done_with_this_frame_descr_void
        elif result_type == history.INT:
            exits = [retbox]
            finishtoken = sd.done_with_this_frame_descr_int
        elif result_type == history.REF:
            exits = [retbox]
            finishtoken = sd.done_with_this_frame_descr_ref
        elif result_type == history.FLOAT:
            exits = [retbox]
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

        # A segment ending in RET is not a tail-loop, so drop any pending
        # condition guard without stamping it for a base-case shift.
        self._pending_cond_guard_descr = None
        self._segment_saw_reset = False
        self._segment_saw_call_assembler = False

        self._already_setup_current_token = False
        self._reset_after_is_true_shift()

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_emit_jump(self, op, targetbox=None, emit_label=False):
        jd = self.jitdriver_sd
        inputargs = self.inputargs
        numargs = op.numargs()

        # create token
        if targetbox is None:
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

        # The segment just closed as a FRAME_RESET tail-loop (emit_jump back to
        # the loop head).  If it carried a loop-continue condition guard AND is
        # a *pure* tail-loop (no in-body CALL_ASSEMBLER), its FALSE branch is
        # the deferred base case recorded one slot too high per condition pop;
        # stamp the guard so its bridge gets shifted down.  Loops whose body
        # issues CALL_ASSEMBLER (e.g. tak) already reconcile the base-case stack
        # through the call-assembler contract, so they must not be shifted.
        if (self._pending_cond_guard_descr is not None and
                self._segment_saw_reset and
                not self._segment_saw_call_assembler):
            descr = self._pending_cond_guard_descr
            if isinstance(descr, compile.AbstractResumeGuardDescr):
                descr.tcg_cond_pop = 1
        self._pending_cond_guard_descr = None
        self._segment_saw_reset = False
        self._segment_saw_call_assembler = False

        self._already_setup_current_token = False
        self._reset_after_is_true_shift()

        if len(self._fdescrstack) > 0:
            self.resumekey = self._fdescrstack.pop()

    def _handle_call_assembler(self, op):
        "convert recursive calls to an op using `call_assembler_x'"
        jd = self.jitdriver_sd

        arglist = op.getarglist()
        if not _tstack_is_empty_at_last_dmp(self._newoperations,
                                            self.jitdriver_sd):
            numargs = op.numargs()
            if numargs > 0:
                lastarg = op.getarg(numargs - 1)
                if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                    op.setarg(numargs - 1, ConstInt(0))
            self.emit(op)
            return
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
        self._emit_call_assembler_stack_effect(arglist, newop)

    def _frame_stack_descriptors(self, frame_box):
        sp_field_descr = None
        stack_field_descr = None
        stack_box = None
        arr_descr = None
        sp_candidates = []
        for op in self._newoperations:
            if (op.getopnum() == rop.GETFIELD_GC_I and op.numargs() == 1 and
                    op.getarg(0) is frame_box):
                sp_candidates.append(op)
                descr = op.getdescr()
                if _descr_name_contains(descr, 'inst_stackpos'):
                    sp_field_descr = descr
            elif (op.getopnum() == rop.SETFIELD_GC and op.numargs() == 2 and
                    op.getarg(0) is frame_box):
                descr = op.getdescr()
                if _descr_name_contains(descr, 'inst_stackpos'):
                    sp_field_descr = descr
            elif (op.getopnum() == rop.GETFIELD_GC_R and op.numargs() == 1 and
                    op.getarg(0) is frame_box):
                descr = op.getdescr()
                if not _descr_name_contains(descr, 'inst_stack'):
                    continue
                candidate_stack = op
                for use in self._newoperations:
                    if (use.numargs() > 0 and use.getarg(0) is candidate_stack and
                            use.getopnum() in (rop.GETARRAYITEM_GC_R,
                                               rop.SETARRAYITEM_GC,
                                               rop.ARRAYLEN_GC)):
                        stack_field_descr = op.getdescr()
                        stack_box = candidate_stack
                        if use.getopnum() in (rop.GETARRAYITEM_GC_R,
                                              rop.SETARRAYITEM_GC):
                            arr_descr = use.getdescr()
                            break
        if stack_box is not None and sp_field_descr is None:
            for candidate in sp_candidates:
                if _int_box_feeds_stack_array(
                        self._newoperations, candidate, stack_box):
                    sp_field_descr = candidate.getdescr()
                    break
        return sp_field_descr, stack_field_descr, arr_descr

    def _emit_call_assembler_stack_effect(self, arglist, call_result):
        if len(arglist) < 5:
            return
        frame_box = arglist[2]
        argnum_box = arglist[4]
        if not isinstance(argnum_box, ConstInt):
            return
        sp_field_descr, stack_field_descr, arr_descr = \
            self._frame_stack_descriptors(frame_box)
        if sp_field_descr is None or stack_field_descr is None or arr_descr is None:
            return

        old_sp = ResOperation(rop.GETFIELD_GC_I, [frame_box],
                              descr=sp_field_descr)
        drop_count = ConstInt(argnum_box.getint())
        new_sp = ResOperation(rop.INT_SUB, [old_sp, drop_count])
        set_sp = ResOperation(rop.SETFIELD_GC, [frame_box, new_sp],
                              descr=sp_field_descr)
        self.emit(old_sp)
        self.emit(new_sp)
        self.emit(set_sp)

        if call_result.type == 'v':
            return
        stack_box = ResOperation(rop.GETFIELD_GC_R, [frame_box],
                                 descr=stack_field_descr)
        resume_slot = ResOperation(rop.INT_SUB, [new_sp, ConstInt(1)])
        set_resume = ResOperation(rop.SETARRAYITEM_GC,
                                  [stack_box, resume_slot, call_result],
                                  descr=arr_descr)
        set_result = ResOperation(rop.SETARRAYITEM_GC,
                                  [stack_box, new_sp, call_result],
                                  descr=arr_descr)
        pushed_sp = ResOperation(rop.INT_ADD, [new_sp, ConstInt(1)])
        set_pushed_sp = ResOperation(rop.SETFIELD_GC, [frame_box, pushed_sp],
                                     descr=sp_field_descr)
        self.emit(stack_box)
        self.emit(resume_slot)
        self.emit(set_resume)
        self.emit(set_result)
        self.emit(pushed_sp)
        self.emit(set_pushed_sp)

    def _handle_dummy_flag(self, op):
        numargs = op.numargs()
        opnum = op.getopnum()
        arglist = op.getarglist()

        newfunc = arglist[-2]
        offset = numargs - 2
        assert offset >= 0
        newargs = arglist[:offset]
        newargs[0] = newfunc
        if len(newargs) > 1:
            lastarg = newargs[-1]
            if isinstance(lastarg, ConstInt) and lastarg.getint() == 1:
                newargs[-1] = ConstInt(0)

        descr = op.getdescr()
        newdescr = descr.get_calldescr_without_flag()

        newop = op.copy_and_change(opnum, newargs, descr=newdescr)
        op.set_forwarded(newop)
        self.emit(newop)

    def _dmp_pcbox(self, op):
        """Return the green `pc' ConstInt of a DEBUG_MERGE_POINT op.

        The op's arglist is [jd_index, portal_call_depth, current_call_id]
        + greenkey (see pyjitpl.debug_merge_point).  Threaded-code drivers use
        the first green key item as their dispatch position."""
        jd = self.jitdriver_sd
        num_green_args = jd.num_green_args
        arglist = op.getarglist()
        greens = arglist[3:3+num_green_args]
        assert len(greens) >= 1
        box = greens[0]
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
        # Match on the bytecode position green, consistently with
        # _dmp_pcbox.
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
        jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
        original_jitcell_token = self.token.original_jitcell_token
        return TargetToken(jitcell_token,
                           original_jitcell_token=original_jitcell_token)

    def _create_continuation_token(self):
        jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
        original_jitcell_token = self.token.original_jitcell_token
        return TargetToken(jitcell_token,
                           original_jitcell_token=original_jitcell_token)

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
