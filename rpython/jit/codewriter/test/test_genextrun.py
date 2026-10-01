
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


def test_large_constant_is_inlined_as_literal():
    ssarepr = SSARepr("test")
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('int_add', i0, Constant(100000, lltype.Signed), '->', i0),
        ('int_return', i0),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    try:
        while 1:
            if pc == 0: # int_add
                i0 = plain_int(glob0(i0, 100000))
                pc = 4
                continue
            elif pc == 4: # int_return
                bh.tmpreg_i = i0
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0"""
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 5)
    assert jit_run(bh) == -1
    assert bh._final_result_anytype() == 100005


def test_ref_return_matches_blackhole(_gcptr_cases):
    ssarepr = SSARepr("test")
    r0 = Register('ref', 0)
    ssarepr.insns = [('ref_return', r0)]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'ref': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)

    for value in _gcptr_cases:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_r(0, value)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, value)


def test_void_return_matches_blackhole():
    ssarepr = SSARepr("test")
    ssarepr.insns = [('void_return',)]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={})
    jit_run = _run_function(assembler, ssarepr, jitcode)

    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    assert jit_run(bh) == -1
    assert bh._final_result_anytype() == _blackhole_result(
        assembler, jitcode)


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


def _blackhole_result(assembler, jitcode, *args):
    def _set_blackhole_arg(bh, index, value):
        if lltype.typeOf(value) is lltype.Signed:
            bh.setarg_i(index, value)
        else:
            bh.setarg_r(index, value)

    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    for index, value in enumerate(args):
        _set_blackhole_arg(bh, index, value)
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


def _fused_branch_jitcode(comp_op):
    assert comp_op in ('lt', 'eq', 'gt'), "comp_op must be one of 'lt', 'eq', or 'gt'"
    ssarepr = SSARepr("test")
    i0, i1 = [Register('int', i) for i in range(2)]
    ssarepr.insns = [
        ('goto_if_not_int_'+comp_op, i0, i1, TLabel('nope')),
        ('int_return', i0),
        (Label('nope'),),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    return assembler, ssarepr, jitcode


@py.test.fixture
def _fused_branch_cases():
    return [(1, 2), (2, 1), (3, 3), (0, 0),
            (-2, 0), (0, -3), (-3, -2), (-1, -1)]


@py.test.fixture
def _gcptr_cases():
    # TODO: Is this right?
    from rpython.rtyper.lltypesystem import llmemory
    X = lltype.GcStruct('X')
    return [lltype.cast_opaque_ptr(llmemory.GCREF, value)
            for value in [lltype.nullptr(X), lltype.malloc(X)]]


def assert_fused_branch_matches_blackhole(comp_op, _fused_branch_cases):
    assembler, ssarepr, jitcode = _fused_branch_jitcode(comp_op)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a, b in _fused_branch_cases:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        bh.setarg_i(1, b)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a, b)

def test_goto_if_not_int_lt_matches_blackhole(_fused_branch_cases):
    assert_fused_branch_matches_blackhole('lt', _fused_branch_cases)

def test_goto_if_not_int_eq_matches_blackhole(_fused_branch_cases):
    assert_fused_branch_matches_blackhole('eq', _fused_branch_cases)

def test_goto_if_not_int_gt_matches_blackhole(_fused_branch_cases):
    assert_fused_branch_matches_blackhole('gt', _fused_branch_cases)

def test_goto_if_not_int_is_true_matches_blackhole():
    ssarepr = SSARepr("test")
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('goto_if_not_int_is_true', i0, TLabel('nope')),
        ('int_return', i0),
        (Label('nope'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a in [0, 1, -1, 42]:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a)

def test_goto_if_not_int_is_zero_matches_blackhole():
    ssarepr = SSARepr("test")
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('goto_if_not_int_is_zero', i0, TLabel('nope')),
        ('int_return', i0),
        (Label('nope'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a in [0, 1, -1, 42]:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, a)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a)


def test_goto_if_not_ptr_iszero_matches_blackhole(_gcptr_cases):
    ssarepr = SSARepr("test")
    r0 = Register('ref', 0)
    ssarepr.insns = [
        ('goto_if_not_ptr_iszero', r0, TLabel('nope')),
        ('int_return', Constant(1, lltype.Signed)),
        (Label('nope'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'ref': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a in _gcptr_cases:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_r(0, a)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a)

def test_goto_if_not_ptr_nonzero_matches_blackhole(_gcptr_cases):
    ssarepr = SSARepr("test")
    r0 = Register('ref', 0)
    ssarepr.insns = [
        ('goto_if_not_ptr_nonzero', r0, TLabel('nope')),
        ('int_return', Constant(1, lltype.Signed)),
        (Label('nope'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'ref': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for a in _gcptr_cases:
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, 0)
        bh.setarg_r(0, a)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == _blackhole_result(
            assembler, jitcode, a)


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

# ----------
# registers kept in Python locals

def _assemble(insns, **num_regs):
    ssarepr = SSARepr("test")
    ssarepr.insns = insns
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs=num_regs)
    return assembler, ssarepr, jitcode


def _check_matches_blackhole(assembler, ssarepr, jitcode, *int_args):
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    for index, value in enumerate(int_args):
        bh.setarg_i(index, value)
    assert jit_run(bh) == -1
    expected = _blackhole_result(assembler, jitcode, *int_args)
    assert bh._final_result_anytype() == expected
    return expected


def test_registers_are_python_locals():
    assembler, ssarepr, jitcode = _arith_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    source = jitcode._genext_run_source
    assert source ==  """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    i2 = bh.registers_i[2]
    i3 = bh.registers_i[3]
    try:
        while 1:
            if pc == 0: # int_add
                i2 = plain_int(glob0(i0, i1))
                pc = 4
                continue
            elif pc == 4: # int_sub
                i3 = plain_int(glob1(i2, 3))
                pc = 8
                continue
            else:
                return pc
    finally:
        bh.registers_i[2] = i2
        bh.registers_i[3] = i3"""

def test_only_written_registers_are_stored_back():
    assembler, ssarepr, jitcode = _assemble([
        ('int_add', Register('int', 0), Register('int', 1), '->',
         Register('int', 2)),
        ('int_return', Register('int', 2)),
        ], int=3)
    _run_function(assembler, ssarepr, jitcode)
    source = jitcode._genext_run_source
    assert source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    i2 = bh.registers_i[2]
    try:
        while 1:
            if pc == 0: # int_add
                i2 = plain_int(glob0(i0, i1))
                pc = 4
                continue
            elif pc == 4: # int_return
                bh.tmpreg_i = i2
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[2] = i2"""

def test_registers_flushed_when_an_exception_escapes():
    from rpython.rtyper.lltypesystem import llmemory, rstr
    assembler, ssarepr, jitcode = _assemble([
        ('int_add', Register('int', 0), Constant(7, lltype.Signed), '->',
         Register('int', 1)),
        ('strlen', Register('ref', 0), '->', Register('int', 2)),
        ('int_return', Register('int', 2)),
        ], int=3, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    class CPU(object):
        def bh_strlen(self, string):
            raise ValueError

    bh = getblackholeinterp(assembler.insns)
    bh.cpu = CPU()
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 5)
    s = rstr.mallocstr(1)
    bh.setarg_r(0, lltype.cast_opaque_ptr(llmemory.GCREF, s))
    py.test.raises(ValueError, jit_run, bh)
    assert bh.registers_i[1] == 12


def _sum_loop_jitcode():
    i0, i1 = Register('int', 0), Register('int', 1)
    return _assemble([
        ('int_copy', Constant(0, lltype.Signed), '->', i1),
        (Label('loop'),),
        ('goto_if_not_int_gt', i0, Constant(0, lltype.Signed),
         TLabel('done')),
        ('int_add', i1, i0, '->', i1),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('goto', TLabel('loop')),
        (Label('done'),),
        ('int_return', i1),
        ], int=2)


def test_loop_with_locals_matches_blackhole():
    for n in [0, 1, 2, 10, 100]:
        assembler, ssarepr, jitcode = _sum_loop_jitcode()
        result = _check_matches_blackhole(assembler, ssarepr, jitcode, n)
        assert result == n * (n + 1) // 2
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    try:
        while 1:
            if pc == 0: # int_copy
                i1 = plain_int(glob0(0))
                pc = 3
                continue
            elif pc == 3: # goto_if_not_int_gt
                if i0 > 0:
                    pc = 8
                else:
                    pc = 19
                continue
            elif pc == 8: # int_add
                i1 = plain_int(glob1(i1, i0))
                pc = 12
                continue
            elif pc == 12: # int_sub
                i0 = plain_int(glob2(i0, 1))
                pc = 16
                continue
            elif pc == 16: # goto
                pc = 3
                continue
            elif pc == 19: # int_return
                bh.tmpreg_i = i1
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0
        bh.registers_i[1] = i1"""


def _const_chain_jitcode():
    i0, i1 = Register('int', 0), Register('int', 1)
    return _assemble([
        ('int_copy', Constant(100000, lltype.Signed), '->', i0),
        ('int_add', i0, Constant(200000, lltype.Signed), '->', i1),
        ('int_return', i1),
        ], int=2)


def test_every_original_pc_is_a_valid_entry():
    assembler, ssarepr, jitcode = _const_chain_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    for start in sorted(set(ssarepr._insns_pos)):
        bh = getblackholeinterp(assembler.insns)
        bh.setposition(jitcode, start)
        bh.setarg_i(0, 100000)
        bh.setarg_i(1, 300000)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == 300000


def test_pc_after_live_is_the_next_instruction():
    i0, i1 = Register('int', 0), Register('int', 1)
    for cond in [0, 1]:
        a, s_, j = _assemble([
            ('goto_if_not', i0, TLabel('else')),
            ('int_copy', Constant(7, lltype.Signed), '->', i1),
            ('goto', TLabel('join')),
            (Label('else'),),
            ('int_copy', Constant(7, lltype.Signed), '->', i1),
            (Label('join'),),
            ('-live-', i1),
            ('int_return', i1),
            ], int=2)
        assert _check_matches_blackhole(a, s_, j, cond) == 7


def test_recursive_call():
    # TODO
    pass

def test_inline_call():
    # TODO
    pass

def test_portal_point():
    # TODO
    pass

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
