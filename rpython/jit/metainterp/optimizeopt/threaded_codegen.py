"""Threaded code generation helpers (split traces at JIT_EMIT_JUMP / JIT_EMIT_RET)."""
from rpython.jit.metainterp.resoperation import rop


def peek_has_nested_threaded_marker_before_loop_end(trace_iter):
    """True iff another ``JIT_EMIT_JUMP`` appears after the current
    iterator position but before the trace's closing ``FINISH`` /
    ``JUMP`` (or a ``JIT_EMIT_RET``, which terminates the same way as
    a return-style exit and must keep its own trace block).

    This drives ``OptTraceSplit``'s ``threaded_inline_handler``
    merging: when one ``emit_jump`` is followed by another
    ``emit_jump`` in the same trace, the first is folded into the
    body so two adjacent threaded sub-traces become a single compiled
    block. We deliberately do NOT merge across an ``emit_ret`` (e.g.
    the EXIT-branch path after a JUMP back-edge): the first
    ``emit_jump`` there is the loop back-edge — eliding it produces a
    ``FINISH``-only trace with no jump-back-to-top and so a one-shot
    loop that bails after one iteration.

    ``TraceIterator.next()`` advances three pieces of mutable state on
    the iterator: ``pos`` (byte position in the encoded trace),
    ``_index`` (the next SSA slot index, used to assign cache positions
    for value-producing ops), and ``_count`` (a separate sequential
    counter). The cache is sized to the trace's total value-producing
    op count, so iterating past the marker and then resuming the main
    walk from a restored ``pos`` overflows ``_cache`` (an ``IndexError``
    in ``opencoder.py`` is the symptom). Save and restore all three
    fields here so the peek is fully side-effect-free.
    """
    saved_pos = trace_iter.pos
    saved_index = trace_iter._index
    saved_count = trace_iter._count
    try:
        while not trace_iter.done():
            op = trace_iter.next()
            opnum = op.getopnum()
            if rop.is_jit_emit_jump(opnum):
                return True
            if rop.is_jit_emit_ret(opnum):
                return False
            if opnum in (rop.FINISH, rop.JUMP):
                return False
        return False
    finally:
        trace_iter.pos = saved_pos
        trace_iter._index = saved_index
        trace_iter._count = saved_count
