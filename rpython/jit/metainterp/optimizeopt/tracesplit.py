from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.rlib.objectmodel import specialize, we_are_translated
from rpython.jit.metainterp.history import (
    ConstInt, RefFrontendOp, IntFrontendOp, FloatFrontendOp)
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
        self.ops_body = []
        self.ops_bridge = []
        self.splitted = False
        self._split_point = 0

    def split(self, ops, inputargs, fname, gmark, tc_jump, tc_guard, body_token, bridge_token):
        cut_point = 0
        for op in ops:
            if op.getopnum() in (rop.CALL_I,
                                 rop.CALL_R,
                                 rop.CALL_F):
                arg = op.getarg(0)
                if arg is None:
                    raise IndexError
                name = _get_name_from_arg(self.metainterp_sd, arg)

                if name is None:
                    raise IndexError

                if name.find(fname) != -1:
                    break
            cut_point += 1

        prev = ops[:cut_point+1]
        latter = ops[cut_point+1:]
        if len(latter) == 0:
            return None

        undefined = []

        def get_undefined_ops_from_args(args):
            l = []
            for arg in args:
                for op in prev:
                    if op == arg:
                        if op not in undefined:
                            l.insert(0, op)
                        args = op.getarglist()
                        get_undefined_ops_from_args(args)
            undefined.extend(l)

        for op in latter:
            args = op.getarglist()
            get_undefined_ops_from_args(args)

        body_ops = self._invent_op(rop.JUMP, inputargs, body_token, prev, fname, tc_jump=tc_jump)
        body_ops, guard_op = self._invent_and_find(body_ops, inputargs, gmark)
        fail_descr = guard_op.getdescr()

        bridge_ops = self._invent_last_op(undefined + latter, inputargs, bridge_token)

        body_label_op = ResOperation(rop.LABEL, inputargs, descr=body_token)
        bridge_label_op = ResOperation(rop.LABEL, inputargs, descr=bridge_token)

        return (TraceSplitInfo(body_token, body_label_op, inputargs, None, fail_descr), body_ops), \
            (TraceSplitInfo(bridge_token, bridge_label_op, inputargs, None), bridge_ops)

    def _invent_op(self, opnum, orig_inputs, target_token, ops, fname, tc_jump=None):
        last_op = ops[-1]
        jump_op = None
        if last_op.getopnum() == rop.CALL_I:
            arg = last_op.getarg(0)
            box = arg.getvalue()
            if isinstance(box, AddressAsInt):
                name = str(box.adr.ptr)
            else:
                name = self.metainterp_sd.get_name_from_address(box)
            if name.find(fname) != -1:
                numargs = last_op.numargs()
                arg = last_op.getarg(numargs-1)
                jump_op = ResOperation(opnum, orig_inputs, descr=target_token)

        if jump_op is None:
            return None

        ops[-1] = jump_op
        return ops

    def _invent_last_op(self, ops, orig_inputs, target_token):
        pseudo_ret = None
        for op in ops:
            if op.getopnum() == rop.CALL_I:
                arg = op.getarg(0)
                name = _get_name_from_arg(self.metainterp_sd, arg)
                if name.find("emit_ret") != -1:
                    pseudo_ret = op
                    ops.remove(op)
                    break
        assert pseudo_ret is not None

        last_op = ops[-1]
        if last_op.getopnum() == rop.GUARD_FUTURE_CONDITION:
            newops = [ResOperation(rop.LEAVE_PORTAL_FRAME, [ConstInt(0)]),
                      ResOperation(rop.FINISH, [pseudo_ret.getarg(1)],
                                   descr=compile.DoneWithThisFrameDescrInt())]
            ops.pop()
            return ops + newops
        else:
            return ops

    def _invent_and_find(self, ops, orig_inputs, marker):
        guard_op_with_marker = None
        for i in range(len(ops)):
            op = ops[i]
            if op.is_guard():
                arg = op.getarg(0)
                if _has_marker(self.metainterp_sd, ops, arg, marker):
                    # change failargs for trace-stitching
                    op.setfailargs(orig_inputs)
                    ops[i] = op
                    guard_op_with_marker = op

        return ops, guard_op_with_marker

    def _invent_inputargs(self, orig_inputs):
        from copy import copy

        l = []
        for input in orig_inputs:
            typ = input.type
            if typ == 'i':
                l.append(InputArgInt(0))
            elif typ == 'f':
                l.append(InputArgFloat(0))
            elif typ == 'r':
                l.append(InputArgRef(0))
            elif typ == 'v':
                l.append(InputArgVector(0))
        return l


class OptTraceSplit(Optimizer):
    def __init__(self, metainterp_sd, jitdriver_sd, optimizations=None,
                 split_at=None, guard_at=None):
        super(OptTraceSplit, self).__init__(metainterp_sd, jitdriver_sd, optimizations=optimizations)
        self.split_at = split_at
        self.guard_at = guard_at
        self.ops_body = []
        self.ops_bridge = []
        self.splitted = False
        self._split_point = 0

    def optimize_and_split(self, trace, resumestorage, call_pure_results):
        traceiter = trace.get_iter()
        if resumestorage:
            frontend_inputargs = trace.inputargs
            deserialize_optimizer_knowledge(
                self, resumestorage, frontend_inputargs, traceiter.inputargs)
        return self.propagate_all_forward(traceiter, call_pure_results)

    def emit(self, op):
        result = Optimization.emit(self, op)
        print result.op
        if result.op.is_guard():
            print result.op.getfailargs()
        return result

    def optimize_CALL_I(self, op):
        arg0 = op.getarg(0)
        if isinstance(arg0, ConstInt):
            if jl.int_could_be_an_address(arg0.value):
                metainterp = self.optimizer.metainterp_sd
                addr = arg0.getaddr()
                name = metainterp.get_name_from_address(addr)
                if name.find(self.split_at) != -1:
                    self.splitted = True

        return self.emit(op)

def find_guard(metainterp_sd, oplist, marker):
    for op in oplist:
        if op.is_guard():
            if op.getopnum() in (rop.GUARD_TRUE,
                                 rop.GUARD_FALSE):
                for i in range(op.numargs()):
                    arg = op.getarg(i)
                    if _has_marker(metainterp_sd, oplist, arg, marker):
                        return op
    return None

def _get_name_from_arg(metainterp_sd, arg):
    box = arg.getvalue()
    if isinstance(box, AddressAsInt):
        return str(box.adr.ptr)
    else:
        return metainterp_sd.get_name_from_address(box)

def _has_marker(metainterp_sd, oplist, arg, marker):
    for op in oplist:
        if op.getopnum() in (rop.CALL_I, rop.CALL_F, rop.CALL_R):
            call_to = op.getarg(0)
            name = _get_name_from_arg(metainterp_sd, call_to)
            if name.find(marker) != -1:
                return True
    return False

dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
                                      default=OptTraceSplit.emit)
OptTraceSplit.propagate_forward = dispatch_opt
