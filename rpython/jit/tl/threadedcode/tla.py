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
    TRACE_THRESHOLD, MAX_INTERP_DEPTH,
    get_printable_location, get_printable_location_tier1,
    _construct_value, _branch_reaches_backedge,
    _entry_has_foreign_call_assembler, _entry_has_wide_call_assembler,
    _compute_stackdepth, _power_01, _construct_float,
    _tier1_confirm_enter_jit,
)
from rpython.jit.tl.threadedcode.frames import (
    Frame, JitFrame, JitFrame3, tier1driver, tier2vdriver,
    tier3driver, _t4cfg,
)


ADAPTIVE_PROFILE_INVOCATIONS = 2


def _run_jitframe2(bytecode, w_arg, jitted):
    stacksize = _compute_stackdepth(bytecode) + 1
    frame = JitFrame(bytecode, stacksize=stacksize, jitted=jitted)
    frame._push(w_arg)
    return frame._interp()


def _run_jitframe3(bytecode, w_arg, hybrid=False, inline_arith=False):
    stacksize = _compute_stackdepth(bytecode) + 1
    frame = JitFrame3(bytecode, stacksize=stacksize, hybrid=hybrid,
                      inline_arith=inline_arith)
    frame._push(w_arg)
    return frame._interp()


def _has_mixed_operand_profile(bytecode):
    i = 0
    n = len(bytecode)
    while i < n:
        if bytecode.cnt_a[i] != 0 and bytecode.cnt_b[i] != 0:
            return True
        if bytecode.bails[i] != 0:
            return True
        i += 1
    return False


def _adaptive_tier4_legacy(bytecode, w_arg):
    # Original tier-4 controller; runs when the CB4 gate is off (cbmodel == 0).
    bytecode.adaptive_invocations += 1
    if bytecode.adaptive_tier == 0:
        if bytecode.adaptive_invocations <= ADAPTIVE_PROFILE_INVOCATIONS:
            # Baseline profiling compiler: collect operand-shape counters
            # without relying on wall-clock sampling.
            return _run_jitframe3(bytecode, w_arg, hybrid=True,
                                  inline_arith=False)
        if _has_mixed_operand_profile(bytecode):
            bytecode.adaptive_tier = 4
        else:
            bytecode.adaptive_tier = 3
    elif bytecode.adaptive_tier == 3 and _has_mixed_operand_profile(bytecode):
        # JSCore/Jikes-style recompile trigger: new profile evidence can move a
        # previously monomorphic program to the guarded hybrid compiler.
        bytecode.adaptive_tier = 4

    if bytecode.adaptive_tier == 4:
        return _run_jitframe3(bytecode, w_arg, hybrid=True, inline_arith=False)
    if bytecode.adaptive_tier == 2:
        return _run_jitframe2(bytecode, w_arg, True)
    return _run_jitframe3(bytecode, w_arg, hybrid=False, inline_arith=False)


# CB4 adaptive hybrid-tier controller; all helpers run off-trace, integer-only.

def _cb_observed_ops(bytecode):
    # execution counter: sum of per-site operand counts (counts[] is tier-1 only)
    i = 0
    n = len(bytecode)
    s = 0
    while i < n:
        s += bytecode.cnt_a[i] + bytecode.cnt_b[i]
        i += 1
    return s


def _cb_sum_bails(bytecode):
    # OSR-exit count: off-type guard-bail replays across sites
    i = 0
    n = len(bytecode)
    s = 0
    while i < n:
        s += bytecode.bails[i]
        i += 1
    return s


def _cb_is_cmp(c):
    return c == LT or c == GT or c == EQ


def _cb_is_ari(c):
    return c == ADD or c == SUB or c == MUL or c == DIV or c == MOD


def _cb_estimate(bytecode):
    # Returns (t3, t4): modelled steady-state cost under inline-all vs the hybrid.
    # t4 is priced to match what _profile actually does (residualize any
    # polymorphic predicate), so the global choice agrees with the per-site one.
    i = 0
    n = len(bytecode)
    t3 = 0
    t4 = 0
    while i < n:
        a = bytecode.cnt_a[i]
        b = bytecode.cnt_b[i]
        total = a + b
        if total > 0:
            c = ord(bytecode.code[i])
            is_cmp = _cb_is_cmp(c)
            if is_cmp or _cb_is_ari(c):
                if a < b:
                    minority = a
                else:
                    minority = b
                if is_cmp:
                    br = _t4cfg.c_br_cmp
                else:
                    br = _t4cfg.c_br_ari
                inl = total * _t4cfg.c_inl + minority * br
                res = total * _t4cfg.c_res
                t3 += inl
                if is_cmp and minority > 0:
                    t4 += res
                elif inl < res:
                    t4 += inl
                else:
                    t4 += res
        i += 1
    return t3, t4


def _cb_should_tier4(bytecode):
    # Pick the hybrid only when the saving, scaled to its expected future
    # (horizon), beats the code-size-scaled recompile cost.  Monomorphic and
    # arithmetic-poly programs have t3 == t4 and stay at tier 3.
    t3, t4 = _cb_estimate(bytecode)
    recomp = _t4cfg.recomp_base + _t4cfg.recomp_slope * len(bytecode)
    return (t3 - t4) * _t4cfg.horizon > recomp


def _cb_reopt_threshold(retry):
    # OSR-exit budget for the next reopt: reopt_base * reopt_mult**min(retry, cap)
    if retry < _t4cfg.reopt_cap:
        r = retry
    else:
        r = _t4cfg.reopt_cap
    m = 1
    k = 0
    while k < r:
        m *= _t4cfg.reopt_mult
        k += 1
    return _t4cfg.reopt_base * m


def _adaptive_tier4(bytecode, w_arg):
    if not _t4cfg.cbmodel:
        return _adaptive_tier4_legacy(bytecode, w_arg)
    bytecode.adaptive_invocations += 1
    if bytecode.adaptive_tier == 0:
        # profile until the size-scaled counter overflows (or the commit floor),
        # then commit via the cost model
        thr = _t4cfg.cnt_base + _t4cfg.cnt_slope * len(bytecode)
        if (_cb_observed_ops(bytecode) < thr and
                bytecode.adaptive_invocations < _t4cfg.cnt_maxinv):
            return _run_jitframe3(bytecode, w_arg, hybrid=True,
                                  inline_arith=False)
        if _cb_should_tier4(bytecode):
            bytecode.adaptive_tier = 4
        else:
            bytecode.adaptive_tier = 3
        bytecode.reopt_baseline = _cb_sum_bails(bytecode)
    elif bytecode.adaptive_tier == 3:
        # OSR-exit reopt with backoff (rarely fires: tier 3 emits no bails; the
        # per-site DRR in _profile is the main exit-driven mechanism)
        exits = _cb_sum_bails(bytecode) - bytecode.reopt_baseline
        if (bytecode.reopt_retry < _t4cfg.reopt_cap and
                exits >= _cb_reopt_threshold(bytecode.reopt_retry)):
            if _cb_should_tier4(bytecode):
                bytecode.adaptive_tier = 4
            bytecode.reopt_retry += 1
            bytecode.reopt_baseline = _cb_sum_bails(bytecode)

    if bytecode.adaptive_tier == 4:
        return _run_jitframe3(bytecode, w_arg, hybrid=True, inline_arith=False)
    return _run_jitframe3(bytecode, w_arg, hybrid=False, inline_arith=False)


def run(bytecode, w_arg, debug=False, tier=1):
    "tier 0=interp, 1=threaded code, 2=stack-manip inliner, 3=tracing JIT."
    # Reuse the caller's Bytecode object: tier1driver greenkeys include
    # bytecode identity; a fresh wrapper each run changes the hash and forces
    # recompilation.
    if tier == 0:
        return _run_jitframe2(bytecode, w_arg, False)
    if tier == 2:
        return _run_jitframe2(bytecode, w_arg, True)
    if tier == 3:
        return _run_jitframe3(bytecode, w_arg, hybrid=False,
                              inline_arith=False)
    if tier == 4:
        return _adaptive_tier4(bytecode, w_arg)
    frame = Frame(bytecode)
    frame.push(w_arg)
    return frame.interp()
