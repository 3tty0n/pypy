import py
from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem import lltype

from rpython.jit.codewriter.assembler import Assembler
from rpython.jit.codewriter.flatten import SSARepr, Register, Label, TLabel
from rpython.jit.codewriter.genextension import GenExtension
from rpython.jit.codewriter.genextrun import generate_run_function
from rpython.jit.metainterp.test.test_blackhole import getblackholeinterp


def _arith_jitcode():
    ssarepr = SSARepr("test")
    i0, i1, i2, i3 = [Register('int', i) for i in range(4)]
    ssarepr.insns = [
        ('int_add', i0, i1, '->', i2),
        ('int_sub', i2, Constant(3, lltype.Signed), '->', i3),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 4})
    return assembler, ssarepr, jitcode


def _run_function(assembler, ssarepr, jitcode):
    genext = GenExtension(assembler, ssarepr, jitcode)
    return generate_run_function(genext)


def test_arith_matches_blackhole():
    assembler, ssarepr, jitcode = _arith_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 40)
    bh.setarg_i(1, 2)
    jit_run(bh)
    assert bh.registers_i[2] == 42
    assert bh.registers_i[3] == 39


def _branch_jitcode():
    ssarepr = SSARepr("test")
    i0, i1, i2 = [Register('int', i) for i in range(3)]
    ssarepr.insns = [
        ('int_lt', i0, i1, '->', i2),
        ('goto_if_not', i2, TLabel('nope')),
        ('int_return', i0),
        (Label('nope'),),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})
    return assembler, ssarepr, jitcode


def _blackhole_result(assembler, jitcode, a, b):
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, a)
    bh.setarg_i(1, b)
    bh.run()
    return bh._final_result_anytype()


def test_goto_if_not_matches_blackhole():
    assembler, ssarepr, jitcode = _branch_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a, b in [(1, 2), (2, 1)]:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        bh.setarg_i(1, b)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a, b)


def _fused_branch_jitcode():
    ssarepr = SSARepr("test")
    i0, i1 = [Register('int', i) for i in range(2)]
    ssarepr.insns = [
        ('goto_if_not_int_lt', i0, i1, TLabel('nope')),
        ('int_return', i0),
        (Label('nope'),),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    return assembler, ssarepr, jitcode


def test_goto_if_not_int_lt_matches_blackhole():
    assembler, ssarepr, jitcode = _fused_branch_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a, b in [(1, 2), (2, 1), (3, 3)]:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        bh.setarg_i(1, b)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a, b)


def _ovf_jitcode():
    ssarepr = SSARepr("test")
    i0, i1, i2 = [Register('int', i) for i in range(3)]
    ssarepr.insns = [
        ('int_add_jump_if_ovf', TLabel('ovf'), i0, i1, '->', i2),
        ('int_return', i2),
        (Label('ovf'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})
    return assembler, ssarepr, jitcode


def test_int_add_jump_if_ovf_matches_blackhole():
    from rpython.rlib.rarithmetic import LONG_BIT
    assembler, ssarepr, jitcode = _ovf_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    big = (1 << (LONG_BIT - 1)) - 1
    for a, b in [(40, 2), (big, 1)]:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        bh.setarg_i(1, b)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a, b)


def test_strlen_uses_cpu():
    from rpython.rtyper.lltypesystem import llmemory, rstr
    ssarepr = SSARepr("test")
    r0 = Register('ref', 0)
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('strlen', r0, '->', i0),
        ('int_return', i0),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1, 'ref': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)

    class CPU(object):
        def bh_strlen(self, string):
            return len(lltype.cast_opaque_ptr(lltype.Ptr(rstr.STR),
                                              string).chars)

    bh = getblackholeinterp(assembler.insns)
    bh.cpu = CPU()
    bh.setposition(jitcode, 0)
    s = rstr.mallocstr(5)
    bh.setarg_r(0, lltype.cast_opaque_ptr(llmemory.GCREF, s))
    assert jit_run(bh) == -1
    assert bh._final_result_anytype() == 5


@py.test.mark.xfail(strict=True, raises=NotImplementedError,
                    reason="run mode lacks -live-, ref_return, "
                           "jit_merge_point, the inline_call_* and "
                           "getfield_* families, and the "
                           "goto_if_not_ptr_iszero variant")
def test_tla_loop():
    from rpython.jit.codewriter.test.test_genext_scaffold import (
        _tla_jitcodes)
    assembler, captured = _tla_jitcodes()
    ssarepr, jitcode, snap = [(s, j, a) for s, j, a in captured
                              if s.name == 'Frame.interp'][0]
    genext = GenExtension(snap, ssarepr, jitcode)
    genext.generate()
    generate_run_function(genext)


@py.test.mark.xfail(strict=True, raises=NotImplementedError,
                    reason="run mode lacks the fallback contract")
def test_fallback_returns_pc():
    ssarepr = SSARepr("test")
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('int_add', i0, i0, '->', i0),
        ('int_push', i0),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 1)
    assert jit_run(bh) == 4
