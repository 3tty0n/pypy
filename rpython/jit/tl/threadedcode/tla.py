"""TLA threaded-code interpreter -- public facade.

Split for readability into:
  * interp_helpers.py -- pure helpers, constants, tier-switch exceptions
  * frames.py         -- the Frame / JitFrame classes and the JIT drivers
This module re-exports their public names so the historical ``tla.<name>`` API
keeps working, and defines run(), the tier-dispatch entry point.

Tiers: 0=interp (JIT off), 1=threaded code, 2=stack-manip inliner, 3=tracing JIT.
"""
from rpython.jit.tl.threadedcode.object import W_Object, W_IntObject, \
    W_FloatObject, W_StringObject, W_ListObject, OperationError
from rpython.jit.tl.threadedcode.bytecode import *
from rpython.jit.tl.threadedcode.interp_helpers import (
    ContinueInTracingJIT, ContinueInThreadedJIT,
    TRACE_THRESHOLD, MAX_INTERP_DEPTH,
    get_printable_location, get_printable_location_tier1,
    _construct_value, _branch_reaches_backedge,
    _entry_has_foreign_call_assembler, _entry_has_wide_call_assembler,
    _compute_stackdepth, _power_01, _construct_float,
    _tier1_confirm_enter_jit, _tier1_use_frame_inliner_for_plain_loops,
)
from rpython.jit.tl.threadedcode.frames import (
    Frame, JitFrame, JitFrame3, tier1driver, tier2driver, tier2vdriver,
    tier3driver, _t4cfg,
)


# def run(bytecode, w_arg, debug=False, tier=None):
#     frame = Frame(bytecode)
#     frame.push(w_arg)
#     if tier >= 2:
#         w_result = frame._interp()
#     else:
#         w_result = frame.interp()
#     return w_result


def run(bytecode, w_arg, debug=False, tier=1):
    "tier 0=interp, 1=threaded code, 2=stack-manip inliner, 3=tracing JIT."
    # NB: reuse the caller's Bytecode object as-is; do NOT re-wrap it into a
    # fresh `Bytecode(bytecode.code)` here.  `bytecode` is a tier1driver
    # greenkey (its greens are pc/entry/bytecode/tstack); a fresh wrapper on
    # every call gives a fresh lltype.identityhash, so the greenkey hash differs
    # each run, the JitCell holding the compiled loop is never found again, and
    # the threaded-code loop is recompiled on *every* run (confirmed via
    # compile_threaded_code: existing_token=N, chainlen=0, hash changes per
    # run).  Reusing the caller's object keeps the greenkey hash stable so the
    # loop persists and is reused across runs.
    if tier == 0 or tier == 2:
        # Tier 2 (inlined): run the *deep*, virtualizable interpreter under the
        # recursive meta-tracing driver (tier2vdriver).  Unlike the tier-1
        # threaded-code path (Frame.interp), it traces straight through the
        # data-stack helpers and inlines calls, and -- because JitFrame is
        # virtualizable -- keeps the value stack and the loop-carried counter
        # boxes unboxed across the compiled loop.  That box elimination is where
        # tier 2's speedup over tier 1 comes from.
        #
        # Tier 0 (the benchmark ground truth) runs the *same* JitFrame._interp
        # with the JIT turned off (targettla sets --jit off), so it is a pure
        # interpreter and is guaranteed to agree with tier 2's result.
        stacksize = _compute_stackdepth(bytecode) + 1
        jframe2 = JitFrame(bytecode, stacksize=stacksize)
        jframe2._push(w_arg)
        return jframe2._interp()
    if tier == 3 or tier == 4:
        # Tier 3 (conventional tracing JIT): its own virtualizable frame
        # (JitFrame3) and driver (tier3driver).  Same virtualizable value stack
        # as tier 2's JitFrame -- so the stack and loop-carried counters stay
        # unboxed -- but the arithmetic is traced *inline* (raw *_inline ops)
        # rather than left as tier 2's residual @jit.dont_look_inside _t2_*
        # calls.  Inlining lets the optimizer fold the integer work and keep the
        # W_IntObject results virtual, so the loop runs as raw machine
        # arithmetic.  tier 3 thus differs from tier 2 only by inlining the
        # arithmetic: the tier2-vs-tier3 gap measures exactly what that buys --
        # not an artificial boxed-stack handicap.
        stacksize = _compute_stackdepth(bytecode) + 1
        jframe3 = JitFrame3(bytecode, stacksize=stacksize, hybrid=(tier == 4),
                            inline_arith=(tier == 4 and _t4cfg.ratio == 1))
        jframe3._push(w_arg)
        return jframe3._interp()
    if tier == 1 and _tier1_use_frame_inliner_for_plain_loops(bytecode):
        frame = Frame(bytecode)
        frame.push(w_arg)
        return frame._interp()
    frame = Frame(bytecode)
    frame.push(w_arg)
    pc = 0
    while True:
        try:
            return frame.interp(pc=pc)
        except ContinueInTracingJIT as e:
            print "switching to tracing", e.pc
            pc = e.pc

        try:
            return frame._interp(pc=pc)
        except ContinueInThreadedJIT as e:
            print "swiching to threaded", e.pc
            pc = e.pc
