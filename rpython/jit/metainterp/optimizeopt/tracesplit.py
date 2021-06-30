from rpython.rtyper.lltypesystem.llmemory import AddressAsInt
from rpython.rlib.rjitlog import rjitlog as jl
from rpython.jit.metainterp.history import ConstInt
from rpython.jit.metainterp.optimizeopt.optimizer import Optimizer, \
    Optimization, BasicLoopInfo
from rpython.jit.metainterp.optimizeopt.util import make_dispatcher_method
from rpython.jit.metainterp.opencoder import Trace, TraceIterator
from rpython.jit.metainterp.resoperation import rop, OpHelpers, ResOperation

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
    def __init__(self, prev, latter, inputs):
        self.prev = prev
        self.latter = latter
        self.inputs = inputs

    def __repr__(self):
        return "ResSplitTrace(%s, %s, %s)" % \
            (self.prev, self.latter, self.inputs)


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
    def split_ops(self, inputargs, ops, fname, target_token):
        cut_point = 0
        for op in ops:
            if op.getopnum() == rop.CALL_I:
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

        prev = self._fillup_op(rop.JUMP, target_token, prev, fname)
        return SplittedTrace(prev, undefined + latter, inputargs)


    def _fillup_op(self, opnum, target_token, ops, fname):
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
