"""Differential correctness tests for the genextension pure-arith x86 shortcut."""
import py
from rpython.jit.backend.detect_cpu import autodetect, MODEL_X86_64
from rpython.jit.metainterp.history import (
    JitCellToken, TargetToken, BasicFailDescr, BasicFinalDescr)
from rpython.jit.codewriter import longlong
from rpython.jit.tool.oparser import parse

if autodetect() != MODEL_X86_64:
    py.test.skip("genext x86 shortcut only on x86-64")

from rpython.jit.backend.detect_cpu import getcpuclass
CPU = getcpuclass()


class FakeStats(object):
    pass


class _FakeAssembler(object):
    insns = {}


class _FakeJitCode(object):
    name = "genext_shortcut_test"
    genext_is_pure_arithmetic = True
    genext_compile_function = None


def _make_compile_shortcut():
    from rpython.jit.codewriter.genextension import GenExtension
    gen = GenExtension(_FakeAssembler(), None, _FakeJitCode())
    gen.jitcode.genext_is_pure_arithmetic = True
    gen._generate_compile_function()
    fn = gen.jitcode.genext_compile_function
    assert fn is not None
    return fn


def _new_cpu():
    cpu = CPU(rtyper=None, stats=FakeStats())
    cpu.setup_once()
    return cpu


def _compile_and_run(use_shortcut, loop_src, namespace, inputvals,
                     result_types):
    cpu = _new_cpu()
    cpu.assembler.set_debug(False)
    loop = parse(loop_src, namespace=namespace)
    looptoken = JitCellToken()
    if use_shortcut:
        looptoken.genext_compile_function = _make_compile_shortcut()
    cpu.compile_loop(loop.inputargs, loop.operations, looptoken, log=False)
    deadframe = cpu.execute_token(looptoken, *inputvals)
    descr = cpu.get_latest_descr(deadframe)
    ident = getattr(descr, "identifier", None)
    out = []
    for i, t in enumerate(result_types):
        if t == "f":
            out.append(longlong.getrealfloat(cpu.get_float_value(deadframe, i)))
        else:
            out.append(cpu.get_int_value(deadframe, i))
    return ident, out


def _check(loop_src, namespace_factory, inputvals, result_types,
           expected=None):
    ref_ident, ref_out = _compile_and_run(
        False, loop_src, namespace_factory(), inputvals, result_types)
    sc_ident, sc_out = _compile_and_run(
        True, loop_src, namespace_factory(), inputvals, result_types)
    assert sc_ident == ref_ident, (
        "exit descr differs: shortcut=%r reference=%r" % (sc_ident, ref_ident))
    assert sc_out == ref_out, (
        "result differs: shortcut=%r reference=%r" % (sc_out, ref_out))
    if expected is not None:
        assert ref_out == expected, (
            "oracle sanity failed: reference=%r expected=%r"
            % (ref_out, expected))
    return sc_out


def _ns():
    return {"targettoken": TargetToken(),
            "fdescr": BasicFailDescr(7),
            "gdescr": BasicFailDescr(8),
            "findescr": BasicFinalDescr(9)}


def test_int_countdown_sum():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_acc2 = int_add(i_acc, i_n)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [10, 0], ["i", "i"], expected=[0, 55])


def test_int_bitops_const_shift():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_a = int_and(i_acc, 255)
    i_o = int_or(i_a, 1)
    i_x = int_xor(i_o, i_n)
    i_sl = int_lshift(i_x, 1)
    i_sr = int_rshift(i_sl, 1)
    i_acc2 = int_add(i_acc, i_sr)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [12, 0], ["i", "i"])


def test_int_variable_shift_count():
    src = """
    [i_n, i_acc, i_sh]
    label(i_n, i_acc, i_sh, descr=targettoken)
    i_v = int_lshift(i_n, i_sh)
    i_acc2 = int_add(i_acc, i_v)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, i_sh, descr=targettoken)
    """
    _check(src, _ns, [8, 0, 2], ["i", "i"])


def test_int_many_live_spill():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i1 = int_add(i_n, 1)
    i2 = int_add(i_n, 2)
    i3 = int_add(i_n, 3)
    i4 = int_add(i_n, 4)
    i5 = int_add(i_n, 5)
    i6 = int_add(i_n, 6)
    i7 = int_add(i_n, 7)
    i8 = int_add(i_n, 8)
    s1 = int_add(i1, i2)
    s2 = int_add(s1, i3)
    s3 = int_add(s2, i4)
    s4 = int_add(s3, i5)
    s5 = int_add(s4, i6)
    s6 = int_add(s5, i7)
    s7 = int_add(s6, i8)
    i_acc2 = int_add(i_acc, s7)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [6, 0], ["i", "i"])


def test_jump_swap_loop_carried():
    src = """
    [i_n, i_a, i_b]
    label(i_n, i_a, i_b, descr=targettoken)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_a, i_b]
    jump(i_n2, i_b, i_a, descr=targettoken)
    """
    _check(src, _ns, [5, 100, 200], ["i", "i"])


def test_int_add_ovf_guard():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_acc2 = int_add_ovf(i_acc, i_n)
    guard_no_overflow(descr=gdescr) [i_acc2, i_n]
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [9, 0], ["i", "i"], expected=[0, 45])


def test_float_accumulate():
    src = """
    [i_n, f_acc]
    label(i_n, f_acc, descr=targettoken)
    f_n = cast_int_to_float(i_n)
    f_acc2 = float_add(f_acc, f_n)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, f_acc2]
    jump(i_n2, f_acc2, descr=targettoken)
    """
    out = _check(src, _ns, [5, 0.0], ["i", "f"])
    assert out[1] == 15.0


def test_float_mixed_ops():
    src = """
    [i_n, f_acc]
    label(i_n, f_acc, descr=targettoken)
    f_n = cast_int_to_float(i_n)
    f_sq = float_mul(f_n, f_n)
    f_h = float_truediv(f_sq, 2.0)
    f_acc2 = float_add(f_acc, f_h)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, f_acc2]
    jump(i_n2, f_acc2, descr=targettoken)
    """
    _check(src, _ns, [6, 0.0], ["i", "f"])


def test_int_unary_ops():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_neg = int_neg(i_n)
    i_inv = int_invert(i_neg)
    i_t = int_is_true(i_inv)
    i_z = int_is_zero(i_n)
    i_s = int_add(i_t, i_z)
    i_acc2 = int_add(i_acc, i_s)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [7, 0], ["i", "i"])


def test_guard_false_exit():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_acc2 = int_add(i_acc, i_n)
    i_n2 = int_sub(i_n, 1)
    i_done = int_le(i_n2, 0)
    guard_false(i_done, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [8, 0], ["i", "i"], expected=[0, 36])


def test_uint_comparisons():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_b = uint_lt(i_n, 100)
    i_a = uint_ge(i_n, 1)
    i_s = int_add(i_b, i_a)
    i_acc2 = int_add(i_acc, i_s)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [6, 0], ["i", "i"])


def test_cast_roundtrip():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    f_n = cast_int_to_float(i_n)
    f_2 = float_mul(f_n, 1.5)
    i_back = cast_float_to_int(f_2)
    i_acc2 = int_add(i_acc, i_back)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [9, 0], ["i", "i"])


def test_float_neg_abs():
    src = """
    [i_n, f_acc]
    label(i_n, f_acc, descr=targettoken)
    f_n = cast_int_to_float(i_n)
    f_neg = float_neg(f_n)
    f_abs = float_abs(f_neg)
    f_acc2 = float_add(f_acc, f_abs)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, f_acc2]
    jump(i_n2, f_acc2, descr=targettoken)
    """
    out = _check(src, _ns, [5, 0.0], ["i", "f"])
    assert out[1] == 15.0


def test_float_many_live_spill():
    src = """
    [i_n, f_acc]
    label(i_n, f_acc, descr=targettoken)
    f_n = cast_int_to_float(i_n)
    f1 = float_add(f_n, 1.0)
    f2 = float_add(f_n, 2.0)
    f3 = float_add(f_n, 3.0)
    f4 = float_add(f_n, 4.0)
    f5 = float_add(f_n, 5.0)
    f6 = float_add(f_n, 6.0)
    f7 = float_add(f_n, 7.0)
    f8 = float_add(f_n, 8.0)
    f9 = float_add(f_n, 9.0)
    fa = float_add(f1, f2)
    fb = float_add(fa, f3)
    fc = float_add(fb, f4)
    fd = float_add(fc, f5)
    fe = float_add(fd, f6)
    ff = float_add(fe, f7)
    fg = float_add(ff, f8)
    fh = float_add(fg, f9)
    f_acc2 = float_add(f_acc, fh)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, f_acc2]
    jump(i_n2, f_acc2, descr=targettoken)
    """
    _check(src, _ns, [5, 0.0], ["i", "f"])


def test_guard_overflow_taken():
    import sys
    big = sys.maxint // 2 + 1
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_prod = int_mul_ovf(i_n, i_n)
    guard_no_overflow(descr=gdescr) [i_acc, i_n]
    i_acc2 = int_add(i_acc, i_prod)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [big, 0], ["i", "i"])


def test_fallback_on_unsupported_op():
    src = """
    [i_a, i_b]
    i_c = int_add(i_a, i_b)
    i_d = int_mul(i_c, i_c)
    finish(i_d, descr=findescr)
    """
    _check(src, _ns, [6, 7], ["i"], expected=[169])


def test_kitchen_sink():
    src = """
    [i_n, i_acc]
    label(i_n, i_acc, descr=targettoken)
    i_a = int_and(i_n, 7)
    i_o = int_or(i_a, 16)
    i_x = int_xor(i_o, 3)
    i_sl = int_lshift(i_x, 2)
    i_sr = int_rshift(i_sl, 1)
    i_ur = uint_rshift(i_sr, 1)
    i_m = int_mul(i_ur, 2)
    i_lt = int_lt(i_m, 1000)
    i_eq = int_eq(i_a, 0)
    i_ne = int_ne(i_o, 0)
    i_t1 = int_add(i_lt, i_eq)
    i_t2 = int_add(i_t1, i_ne)
    i_t3 = int_add(i_t2, i_m)
    i_acc2 = int_add(i_acc, i_t3)
    i_n2 = int_sub(i_n, 1)
    i_cond = int_gt(i_n2, 0)
    guard_true(i_cond, descr=fdescr) [i_n2, i_acc2]
    jump(i_n2, i_acc2, descr=targettoken)
    """
    _check(src, _ns, [20, 0], ["i", "i"])
