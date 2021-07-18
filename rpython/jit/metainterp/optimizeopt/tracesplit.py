from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.objectmodel import specialize, we_are_translated
from rpython.jit.metainterp.history import (
    ConstInt, ConstFloat, RefFrontendOp, IntFrontendOp, FloatFrontendOp)
from rpython.jit.metainterp import compile, jitprof
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

from pprint import pprint

class TraceSplitInfo(BasicLoopInfo):
    """ A state after splitting the trace, containing the following:

    * target_token - generated target token for a bridge ("false" branch)
    * label_op - label operations
    """
    def __init__(self, target_token, label_op, inputargs,
                 quasi_immutable_deps, fail_descr=None):
        self.target_token = target_token
        self.label_op = label_op
        self.inputargs = inputargs
        self.quasi_immutable_deps = quasi_immutable_deps
        self.fail_descr = fail_descr

    def final(self):
        return True

class TraceSplitOpt(object):

    def __init__(self, metainterp_sd, jitdriver_sd, optimizations=None,
                 resumekey=None, split_at=None, guard_at=None):
        self.metainterp_sd = metainterp_sd
        self.jitdriver_sd = jitdriver_sd
        self.optimizations = optimizations
        self.resumekey = resumekey
        self.split_at = split_at
        self.guard_at = guard_at

    def split(self, trace, oplist, inputs, body_token, bridge_token):
        ops_body, ops_bridge, inputs_body, inputs_bridge = [], [], inputs, []
        cut_at = 0
        last_op = None
        newops = []
        for i in range(len(oplist)):
            op = oplist[i]
            if op.getopnum() in (rop.CALL_I, rop.CALL_R, rop.CALL_F, rop.CALL_N):
                arg = op.getarg(0)
                name = self._get_name_from_arg(arg)
                assert name is not None

                if name.find(self.split_at) != -1:
                    if self.split_at.find("jump"):
                        last_op = ResOperation(rop.JUMP, inputs, body_token)
                        cut_at = i

                newops.append(op)
            else:
                newops.append(op)


        assert last_op is not None

        ops_body = newops[:cut_at] + [last_op]
        ops_bridge = newops[cut_at:]

        ops_body, ops_bridge, inputs_bridge = self.set_guard_descr_and_bridge_inputs(
            ops_body, ops_bridge, bridge_token)

        body_label = ResOperation(rop.LABEL, inputs, descr=body_token)
        bridge_label = ResOperation(rop.LABEL, inputs_bridge, descr=bridge_token)

        return (TraceSplitInfo(body_token, body_label, inputs, None, None), ops_body), \
            (TraceSplitInfo(bridge_token, bridge_label, inputs_bridge, None, None), ops_bridge)

    def set_guard_descr_and_bridge_inputs(self, ops_body, ops_bridge, bridge_token):
        inputs_bridge = []
        l = []
        for op in ops_body:
            if op.is_guard():
                arg = op.getarg(0)
                if self._has_marker(ops_body, arg, self.guard_at):
                    op.setdescr(bridge_token)
                    # setting up inputargs for the bridge_ops
                    inputs_bridge = self._invent_failargs(op.getfailargs())
                    op.setfailargs(inputs_bridge)
            l.append(op)

        return l, ops_bridge, inputs_bridge

    def _invent_failargs(self, failargs):
        l = []
        for arg in failargs:
            if isinstance(arg, InputArgInt) or isinstance(arg, InputArgFloat) or \
               isinstance(arg, InputArgRef) or isinstance(arg, InputArgVector):
                l.append(arg)
        return l

    def _has_op(self, op1, oplist):
        for op2 in oplist:
            if op1 in op2.getarglist():
                return True
        return False


    # def split(self, ops, inputargs, fname, gmark, tc_jump, tc_guard, body_token, bridge_token):
    #     cut_point = 0
    #     for op in ops:
    #         if op.getopnum() in (rop.CALL_I,
    #                              rop.CALL_R,
    #                              rop.CALL_F):
    #             arg = op.getarg(0)
    #             if arg is None:
    #                 raise IndexError
    #             name = self._get_name_from_arg(arg)

    #             if name is None:
    #                 raise IndexError

    #             if name.find(fname) != -1:
    #                 break
    #         cut_point += 1

    #     prev = ops[:cut_point+1]
    #     latter = ops[cut_point+1:]
    #     if len(latter) == 0:
    #         return None

    #     undefined = []

    #     def get_undefined_ops_from_args(args):
    #         l = []
    #         for arg in args:
    #             for op in prev:
    #                 if op == arg:
    #                     if op not in undefined:
    #                         l.insert(0, op)
    #                     args = op.getarglist()
    #                     get_undefined_ops_from_args(args)
    #         undefined.extend(l)

    #     for op in latter:
    #         args = op.getarglist()
    #         get_undefined_ops_from_args(args)

    #     body_ops = self._invent_op(rop.JUMP, inputargs, body_token, prev, fname, tc_jump=tc_jump)
    #     body_ops, guard_op = self._invent_and_find(body_ops, inputargs, gmark)
    #     fail_descr = guard_op.getdescr()

    #     bridge_ops = undefined + latter

    #     body_label_op = ResOperation(rop.LABEL, inputargs, descr=body_token)
    #     bridge_label_op = ResOperation(rop.LABEL, inputargs, descr=bridge_token)

    #     return (TraceSplitInfo(body_token, body_label_op, inputargs, None, fail_descr), body_ops), \
    #         (TraceSplitInfo(bridge_token, bridge_label_op, inputargs, None), bridge_ops)

    # def _invent_op(self, opnum, orig_inputs, target_token, ops, fname, tc_jump=None):
    #     last_op = ops[-1]
    #     jump_op = None
    #     if last_op.getopnum() == rop.CALL_I:
    #         arg = last_op.getarg(0)
    #         box = arg.getvalue()
    #         if isinstance(box, AddressAsInt):
    #             name = str(box.adr.ptr)
    #         else:
    #             name = self.metainterp_sd.get_name_from_address(box)
    #         if name.find(fname) != -1:
    #             numargs = last_op.numargs()
    #             arg = last_op.getarg(numargs-1)
    #             jump_op = ResOperation(opnum, orig_inputs, descr=target_token)

    #     if jump_op is None:
    #         return None

    #     ops[-1] = jump_op
    #     return ops

    # def _invent_last_op(self, ops, orig_inputs, target_token):
    #     pseudo_ret = None
    #     for op in ops:
    #         if op.getopnum() == rop.CALL_I:
    #             arg = op.getarg(0)
    #             name = self._get_name_from_arg(arg)
    #             if name.find("emit_ret") != -1:
    #                 pseudo_ret = op
    #                 ops.remove(op)
    #                 break
    #     if pseudo_ret is None:
    #         return ops

    #     last_op = ops[-1]
    #     if last_op.getopnum() == rop.GUARD_FUTURE_CONDITION:
    #         newops = [ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(0)]),
    #                   ResOperation(rop.FINISH, [pseudo_ret.getarg(1)],
    #                                descr=compile.DoneWithThisFrameDescrInt())]
    #         ops.pop()
    #         return ops + newops
    #     else:
    #         return ops

    # def _invent_and_find(self, ops, orig_inputs, marker):
    #     guard_op_with_marker = None
    #     for i in range(len(ops)):
    #         op = ops[i]
    #         if op.is_guard():
    #             arg = op.getarg(0)
    #             if _has_marker(self.metainterp_sd, ops, arg, marker):
    #                 # change failargs for trace-stitching
    #                 op.setfailargs(orig_inputs)
    #                 ops[i] = op
    #                 guard_op_with_marker = op

    #     return ops, guard_op_with_marker

    # def find_guard(self, oplist):
    #     metainterp_sd = self.metainterp_sd
    #     guard_at = self.guard_at
    #     for op in oplist:
    #         if op.is_guard():
    #             if op.getopnum() in (rop.GUARD_TRUE,
    #                                  rop.GUARD_FALSE):
    #                 for i in range(op.numargs()):
    #                     arg = op.getarg(i)
    #                     if self._has_marker(oplist, arg, guard_at):
    #                         return op
    #     return None

    def _get_name_from_arg(self, arg):
        marker = self.metainterp_sd
        box = arg.getvalue()
        if isinstance(box, AddressAsInt):
            return str(box.adr.ptr)
        else:
            return metainterp_sd.get_name_from_address(box)

    def _has_marker(self, oplist, arg, marker):
        metainterp_sd = self.metainterp_sd
        for op in oplist:
            if op.getopnum() in (rop.CALL_I, rop.CALL_F, rop.CALL_R):
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
