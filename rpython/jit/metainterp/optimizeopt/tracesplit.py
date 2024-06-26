from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rstring import find
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.objectmodel import specialize, we_are_translated
from rpython.jit.metainterp.history import (
    ConstInt, ConstFloat, RefFrontendOp, IntFrontendOp, FloatFrontendOp)
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


class mark(object):
    JUMP = "emit_jump"
    RET = "emit_ret"
    IS_TRUE = "is_true"

    @staticmethod
    def is_pseudo_jump(name):
        return name.find(mark.JUMP) != -1

    @staticmethod
    def is_pseudo_ret(name):
        return name.find(mark.RET) != -1

    @staticmethod
    def is_pseudo_op(name):
        return name.find(mark.JUMP) != -1 or name.find(mark.RET) != -1

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


class TraceSplitOpt(object):

    def __init__(self, metainterp_sd, jitdriver_sd, optimizations=None,
                 resumekey=None):
        self.metainterp_sd = metainterp_sd
        self.jitdriver_sd = jitdriver_sd
        self.optimizations = optimizations
        self.resumekey = resumekey
        self.first_cut = True

    def split_ops(self, trace, ops, inputargs, token):
        "Threaded code: splitting the given ops into several op lists"

        t_lst = []                # result

        current_ops = []          # store ops temporarily
        pseudo_ops = []           # for removing useless guards
        fdescr_stack = []         # for bridges

        for op in self.remove_guards(ops):
            opnum = op.getopnum()
            if op.is_guard():
                if self._is_guard_marked(op, ops, mark.IS_TRUE):
                    descr = op.getdescr()
                    fdescr_stack.append(descr)
                    failargs = op.getfailargs()
                    newfailargs = []
                    for farg in failargs:
                        if not farg in pseudo_ops:
                            newfailargs.append(farg)
                    op.setfailargs(newfailargs)

                current_ops.append(op)
            elif rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
                arg = op.getarg(0)
                name = self._get_name_from_arg(arg)
                if name.find(mark.JUMP) != -1:
                    pseudo_ops.append(op)
                    target_token = self._create_token(token)
                    jump_op = ResOperation(rop.JUMP, inputargs, target_token)
                    label_op = ResOperation(rop.LABEL, inputargs, target_token)
                    info = TraceSplitInfo(target_token, label_op, inputargs, self.resumekey)
                    t_lst.append((info, current_ops + [jump_op]))
                    current_ops = []
                    if len(fdescr_stack) > 0:
                        self.resumekey = fdescr_stack.pop()
                elif name.find(mark.RET) != -1:
                    pseudo_ops.append(op)
                    jd_no = self.jitdriver_sd.index
                    result_type = self.jitdriver_sd.result_type
                    sd = self.metainterp_sd
                    if result_type == history.VOID:
                        exits = []
                        finishtoken = sd.done_with_this_frame_descr_void
                    elif result_type == history.INT:
                        exits = [op.getarg(2)]
                        finishtoken = sd.done_with_this_frame_descr_int
                    elif result_type == history.REF:
                        exits = [op.getarg(2)]
                        finishtoken = sd.done_with_this_frame_descr_ref
                    elif result_type == history.FLOAT:
                        exits = [op.getarg(2)]
                        finishtoken = sd.done_with_this_frame_descr_float
                    else:
                        assert False

                    # host-stack style
                    ret_ops = [
                        ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(jd_no)], None),
                        ResOperation(rop.FINISH, exits, finishtoken)
                    ]

                    target_token = self._create_token(token)
                    label_op = ResOperation(rop.LABEL, inputargs, target_token)
                    info = TraceSplitInfo(target_token, label_op, inputargs, self.resumekey)
                    t_lst.append((info, current_ops + ret_ops))
                    current_ops = []
                    if len(fdescr_stack) > 0:
                        self.resumekey = fdescr_stack.pop()
                elif name.find(mark.IS_TRUE) != -1:
                    pseudo_ops.append(op)
                    current_ops.append(op)
                else:
                    current_ops.append(op)
            elif op.getopnum() == rop.FINISH:
                # fdescr, target_token = self._take_fdescr_and_gen_token(token)
                target_token = self._create_token(token)

                label = ResOperation(rop.LABEL, inputargs, target_token)
                info = TraceSplitInfo(target_token, label, inputargs, self.resumekey)
                current_ops.append(op)
                t_lst.append((info, current_ops))
                current_ops = []
                break
            elif op.getopnum() == rop.JUMP:
                # fdescr, target_token = self._take_fdescr_and_gen_token(token)
                target_token = self._create_token(token)
                label = ResOperation(rop.LABEL, inputargs, target_token)
                info = TraceSplitInfo(target_token, label, inputargs, faildescr=self.resumekey)
                current_ops.append(op)
                t_lst.append((info, current_ops))
                current_ops = []
                break
            else:
                current_ops.append(op)

        return t_lst

    def _create_token(self, token):
        if self.first_cut:
            self.first_cut = False
            return token
        else:
            jitcell_token = compile.make_jitcell_token(self.jitdriver_sd)
            original_jitcell_token = token.original_jitcell_token
            return TargetToken(jitcell_token,
                               original_jitcell_token=original_jitcell_token)

    def remove_guards(self, oplist):
        "Remove guard_ops assosiated with pseudo ops"
        pseudo_ops = []
        for op in oplist:
            if self._is_pseudo_op(op):
                pseudo_ops.append(op)

        newops = []
        for op in oplist:
            can_be_recorded = True
            args = op.getarglist()
            for arg in args:
                if arg in pseudo_ops:
                    can_be_recorded = False
                    pseudo_ops.append(op)
                    break
            if can_be_recorded:
                newops.append(op)

        return newops

    def copy_from_body_to_bridge(self, ops_body, ops_bridge):

        def copy_transitively(oplist, arg, res=[]):
            for op in oplist:
                if op == arg:
                    res.append(op)
                    for arg in op.getarglist():
                        copy_transitively(oplist, arg, res)
            return res

        l = []
        for op in ops_bridge:
            args = op.getarglist()
            for arg in args:
                if arg in ops_body:
                    l = copy_transitively(ops_body, arg)

        return l + ops_bridge

    def _has_op(self, op1, oplist):
        for op2 in oplist:
            if op1 in op2.getarglist():
                return True
        return False

    def _is_pseudo_op(self, op):
        opnum = op.getopnum()
        if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
            arg = op.getarg(0)
            name = self._get_name_from_arg(arg)
            if name:
                return mark.is_pseudo_op(name)
            else:
                return False
        return False

    def _get_name_from_arg(self, arg):
        if isinstance(arg, ConstInt):
            addr = arg.getaddr()
            res = self.metainterp_sd.get_name_from_address(addr)
            if res:
                return res

        # TODO: explore more precise way
        return ''


    def _is_guard_marked(self, guard_op, ops, mark):
        "Check if the guard_op is marked"
        assert guard_op.is_guard()
        guard_args = guard_op.getarglist()
        for op in ops:
            opnum = op.getopnum()
            if rop.is_plain_call(opnum) or rop.is_call_may_force(opnum):
                if op in guard_args:
                    name = self._get_name_from_arg(op.getarg(0))
                    end = len(name)
                    if name is None:
                        return False
                    else:
                        return name.find(mark) != -1
        return False

    def _has_marker(self, oplist, arg, marker):
        metainterp_sd = self.metainterp_sd
        for op in oplist:
            if op == arg:
                call_to = op.getarg(0)
                name = self._get_name_from_arg(call_to)
                if name.find(marker) != -1:
                    return True
        return False


class OptTraceSplit(Optimization):

    def __init__(self, metainterp_sd, jitdriver_sd):
        Optimizer.__init__(self, metainterp_sd, jitdriver_sd)
        self.split_at = None
        self.guard_at = None
        self.body_ops = []
        self.bridge_ops = []


dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
                                      default=OptTraceSplit.emit)
OptTraceSplit.propagate_forward = dispatch_opt
