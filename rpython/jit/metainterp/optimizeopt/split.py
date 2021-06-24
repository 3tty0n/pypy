from rpython.jit.metainterp.resoperation import rop, ResOperation

def split_trace_at(trace, at_fname):
    import copy

    if at_fname is None:
        return None

    cut_point = trace.cut_point_by_fname("cut_here")
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


def split_trace(trace, at_fname):
    assert at_fname is not None

    cut_point = trace.cut_point_by_fname(at_fname)
    iter = trace.get_iter()
    prev, undef, latter = iter.split_at(cut_point)
    return SplittedTrace(prev, undef + latter, trace.inputargs)


def split_ops(metainterp_sd, inputargs, ops, fname, target_token):
    cut_point = 0
    for op in ops:
        if op.getopnum() == rop.CALL_I:
            arg = op.getarg(0)
            if arg is None:
                raise IndexError
            v = arg.getvalue()
            name = metainterp_sd.get_name_from_address(v)
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

    prev = _fillup_jump(metainterp_sd, prev, target_token)
    return SplittedTrace(prev, undefined + latter, inputargs)

def _fillup_jump(metainterp_sd, ops, target_token):
    last_op = ops[-1]
    jump_op = None
    if last_op.getopnum() == rop.CALL_I:
        arg = last_op.getarg(0)
        v = arg.getvalue()
        name = metainterp_sd.get_name_from_address(v)
        if name.find("emit_jump") != -1:
            target = last_op.getarg(2)
            jump_op = ResOperation(rop.JUMP, [target], descr=target_token)

    if jump_op is None:
        return None
    return ops + [jump_op]
