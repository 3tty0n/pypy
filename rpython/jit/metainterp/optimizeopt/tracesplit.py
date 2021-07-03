from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.jit.metainterp.history import ConstInt, \
    RefFrontendOp, IntFrontendOp, FloatFrontendOp
from rpython.jit.metainterp.optimizeopt.optimizer import Optimizer, \
    Optimization, BasicLoopInfo
from rpython.jit.metainterp.optimizeopt.util import make_dispatcher_method
from rpython.jit.metainterp.opencoder import Trace, TraceIterator
from rpython.jit.metainterp.resoperation import rop, OpHelpers, ResOperation, \
    InputArgRef, InputArgInt, InputArgFloat, InputArgVector

def split_trace_at(trace, at_fname):
    import copy

    if at_fname is None:
        return None

    cut_point = trace.cut_point_by_fname(at_fname)
    (c_start, c_count, c_index) = cut_point

    c_after_point = c_start, trace._count - c_count + 1, c_index # important hack
    t_after_cutted = trace.cut_trace_from(c_after_point, trace.inputargs)

    t = copy.copy(trace)
    t.cut_at(list(cut_point))

    return t_after_cutted, t


class SplittedTrace:
    def __init__(self, ops, inputargs):
        self.ops = ops
        self.inputargs = inputargs

    def __repr__(self):
        return "ResSplitTrace(%s, %s, %s)" % \
            (self.oplist, self.inputs)


class TraceSplitInfo(BasicLoopInfo):
    """ A state after splitting the trace, containing the following:

    * target_token - generated target token for a bridge ("false" branch)
    * label_op - label operations
    """
    def __init__(self, target_token, label_op, inputargs,
                 quasi_immutable_deps):
        self.target_token = target_token
        self.label_op = label_op
        self.inputargs = inputargs
        self.quasi_immutable_deps = quasi_immutable_deps

    def final(self):
        return True

class TraceSplitOpt(Optimizer):

    def __init__(self, metainterp_sd, jitdriver_sd, optimizations=None,
                 resumekey=None, runtime_boxes=None):
        super(TraceSplitOpt, self).__init__(metainterp_sd, jitdriver_sd,
                                            optimizations=optimizations)
        self.resumekey = resumekey
        self.runtime_boxes = runtime_boxes

    def split_ops(self, inputargs, ops, fname, target_token):
        cut_point = 0
        for op in ops:
            if op.getopnum() in (rop.CALL_I,
                                 rop.CALL_R,
                                 rop.CALL_F):
                arg = op.getarg(0)
                if arg is None:
                    raise IndexError
                v = arg.getvalue()
                name = self.metainterp_sd.get_name_from_address(v)
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

        body_ops = self._invent_op(rop.JUMP, target_token, prev, fname)
        body_ops = self._invent_failargs(body_ops, inputargs, marker="is_true")

        return (TraceSplitInfo(target_token, body_ops[-1], inputargs, None), body_ops), \
            (TraceSplitInfo(None, latter[-1], inputargs, None), undefined + latter)

    def _invent_op(self, opnum, target_token, ops, fname):
        last_op = ops[-1]
        jump_op = None
        if last_op.getopnum() == rop.CALL_I:
            arg = last_op.getarg(0)
            v = arg.getvalue()
            name = self.metainterp_sd.get_name_from_address(v)
            if name.find(fname) != -1:
                arg = last_op.getarg(3)
                jump_op = ResOperation(opnum, [arg], descr=target_token)

        if jump_op is None:
            return None

        ops[-1] = jump_op
        return ops

    def _invent_failargs(self, ops, orig_inputs, marker):
        for i in range(len(ops)):
            op = ops[i]
            if op.is_guard():
                arg = op.getarg(0)
                if self._has_marker(ops, arg, marker):
                    op.setfailargs(orig_inputs)
                    ops[i] = op
        return ops

    def _invent_inputargs(self, ops, orig_inputs, marker):
        from copy import copy

        for op in ops:
            if op.is_guard():
                arg = op.getarg(0)
                if self._has_marker(ops, arg, marker):
                    guard_with_mark = op

        assert guard_with_mark is not None
        failargs = guard_with_mark.getfailargs()

        assert failargs is not None

        l = copy(orig_inputs)
        # create dummy inputargs
        for failarg in failargs:
            if failarg not in orig_inputs:
                typ = failarg.type
                if typ == 'i':
                    l.insert(0, InputArgInt(0))
                elif typ == 'f':
                    l.insert(0, InputArgFloat(0))
                elif typ == 'v':
                    l.insert(0, InputArgVector(0))
        return l

    def find_guard(self, oplist, marker):
        for op in oplist:
            if op.is_guard():
                if op.getopnum() in (rop.GUARD_TRUE,
                                     rop.GUARD_FALSE):
                    for i in range(op.numargs()):
                        arg = op.getarg(i)
                        if self._has_marker(oplist, arg, marker):
                            return op
        return None

    def _has_marker(self, oplist, arg, marker):
        for op in oplist:
            if op.getopnum() in (rop.CALL_I, rop.CALL_F, rop.CALL_R):
                call_to = op.getarg(0)
                v = call_to.getvalue()
                name = self.metainterp_sd.get_name_from_address(v)
                if name.find(marker) != -1:
                    return True
        return False

    def _get_name_from_arg(self, arg):
        addr = arg.getvalue()
        return self.metainterp_sd.get_name_from_address(addr)

    def emit(self, op):
        return Optimization.emit(self, op)

    def optimize_CALL_I(self, op):
        arg0 = op.getarg(0)
        if isinstance(arg0, ConstInt):
            if jl.int_could_be_an_address(arg0.value):
                metainterp = self.optimizer.metainterp_sd
                addr = arg0.getaddr()
                name = metainterp.get_name_from_address(addr)
                if name.find('cut_here') != -1:
                    return None

        return self.emit(op)

# dispatch_opt = make_dispatcher_method(OptTraceSplit, 'optimize_',
#                                       default=OptTraceSplit.emit)
# OptTraceSplit.propagate_forward = dispatch_opt
