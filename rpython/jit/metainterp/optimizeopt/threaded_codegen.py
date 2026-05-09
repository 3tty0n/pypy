"""Threaded code generation helpers (split traces at JIT_EMIT_JUMP / JIT_EMIT_RET)."""
from rpython.jit.metainterp.resoperation import rop


def peek_has_nested_threaded_marker_before_loop_end(trace_iter):
    saved_pos = trace_iter.pos
    try:
        while not trace_iter.done():
            op = trace_iter.next()
            opnum = op.getopnum()
            if rop.is_jit_emit_jump(opnum) or rop.is_jit_emit_ret(opnum):
                return True
            if opnum in (rop.FINISH, rop.JUMP):
                return False
        return False
    finally:
        trace_iter.pos = saved_pos


def should_elide_void_handler_call(opt, op, callee_name):
    if opt._inline_depth <= 0:
        return False
    if op.type != 'v':
        return False
    if not callee_name.startswith("handler_"):
        return False
    return True
