import py
from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem import llmemory, lltype

from rpython.jit.codewriter.assembler import Assembler
from rpython.jit.codewriter.flatten import (
    SSARepr, Register, Label, TLabel, ListOfKind)
from rpython.jit.codewriter.genextension import GenExtension
from rpython.jit.codewriter.genextrun import generate_run_function
from rpython.jit.codewriter.jitcode import JitCode
from rpython.jit.metainterp import jitexc
from rpython.jit.metainterp.history import AbstractDescr
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

def _fused_branch_jitcode_matches_expected(comp_op, jitcode):
    return jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    while 1:
        if pc == 0: # goto_if_not_int_%s
            if i0 %s i1:
                pc = 5
            else:
                pc = 7
            continue
        elif pc == 5: # int_return
            bh.tmpreg_i = i0
            bh._return_type = 'i'
            return -1
        elif pc == 7: # int_return
            bh.tmpreg_i = i1
            bh._return_type = 'i'
            return -1
        else:
            return pc""" % (comp_op, {'lt': '<', 'eq': '==', 'gt': '>'}[comp_op])


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
    assert _fused_branch_jitcode_matches_expected(comp_op, jitcode)
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
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    while 1:
        if pc == 0: # goto_if_not_int_is_true
            if i0:
                pc = 4
            else:
                pc = 6
            continue
        elif pc == 4: # int_return
            bh.tmpreg_i = i0
            bh._return_type = 'i'
            return -1
        elif pc == 6: # int_return
            bh.tmpreg_i = -1
            bh._return_type = 'i'
            return -1
        else:
            return pc"""


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
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    while 1:
        if pc == 0: # goto_if_not_int_is_zero
            if i0 == 0:
                pc = 4
            else:
                pc = 6
            continue
        elif pc == 4: # int_return
            bh.tmpreg_i = i0
            bh._return_type = 'i'
            return -1
        elif pc == 6: # int_return
            bh.tmpreg_i = -1
            bh._return_type = 'i'
            return -1
        else:
            return pc"""


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
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    r0 = bh.registers_r[0]
    while 1:
        if pc == 0: # goto_if_not_ptr_iszero
            if not r0:
                pc = 4
            else:
                pc = 6
            continue
        elif pc == 4: # int_return
            bh.tmpreg_i = 1
            bh._return_type = 'i'
            return -1
        elif pc == 6: # int_return
            bh.tmpreg_i = -1
            bh._return_type = 'i'
            return -1
        else:
            return pc"""

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
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    r0 = bh.registers_r[0]
    while 1:
        if pc == 0: # goto_if_not_ptr_nonzero
            if r0:
                pc = 4
            else:
                pc = 6
            continue
        elif pc == 4: # int_return
            bh.tmpreg_i = 1
            bh._return_type = 'i'
            return -1
        elif pc == 6: # int_return
            bh.tmpreg_i = -1
            bh._return_type = 'i'
            return -1
        else:
            return pc"""


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
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    i2 = bh.registers_i[2]
    try:
        while 1:
            if pc == 0: # int_add_jump_if_ovf
                try:
                    i2 = ovfcheck(i0 + i1)
                except OverflowError:
                    pc = 8
                    continue
                pc = 6
                continue
            elif pc == 6: # int_return
                bh.tmpreg_i = i2
                bh._return_type = 'i'
                return -1
            elif pc == 8: # int_return
                bh.tmpreg_i = -1
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[2] = i2"""


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
# registers kept in Python locals, and generation-time constant propagation

def _loop_body(jitcode):
    """The part of the generated source that runs inside the dispatch loop"""
    lines = jitcode._genext_run_source.split("\n")
    start = [i for i, l in enumerate(lines) if l.strip() == "while 1:"][0]
    end = len(lines)
    for i, l in enumerate(lines):
        if l.strip() == "finally:":
            end = i
    return "\n".join(lines[start + 1:end])


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
                i1 = 0
                pc = 3
                continue
            elif pc == 3: # goto_if_not_int_gt
                if i0 > 0:
                    pc = 8
                else:
                    pc = 19
                continue
            elif pc == 8: # int_add
                i1 = plain_int(glob0(i1, i0))
                pc = 12
                continue
            elif pc == 12: # int_sub
                i0 = plain_int(glob1(i0, 1))
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


def test_loop_carried_register_is_not_treated_as_constant():
    assembler, ssarepr, jitcode = _sum_loop_jitcode()
    _run_function(assembler, ssarepr, jitcode)
    body = _loop_body(jitcode).strip()
    assert body == """if pc == 0: # int_copy
                i1 = 0
                pc = 3
                continue
            elif pc == 3: # goto_if_not_int_gt
                if i0 > 0:
                    pc = 8
                else:
                    pc = 19
                continue
            elif pc == 8: # int_add
                i1 = plain_int(glob0(i1, i0))
                pc = 12
                continue
            elif pc == 12: # int_sub
                i0 = plain_int(glob1(i0, 1))
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
                return pc"""

def _const_chain_jitcode():
    i0, i1 = Register('int', 0), Register('int', 1)
    return _assemble([
        ('int_copy', Constant(100000, lltype.Signed), '->', i0),
        ('int_add', i0, Constant(200000, lltype.Signed), '->', i1),
        ('int_return', i1),
        ], int=2)


def test_constants_are_propagated_and_folded():
    assembler, ssarepr, jitcode = _const_chain_jitcode()
    result = _check_matches_blackhole(assembler, ssarepr, jitcode)
    assert result == 300000
    body = _loop_body(jitcode).strip()
    assert body == """if pc == 0: # int_copy
                i0 = 100000
                pc = 3
                continue
            elif pc == 3: # int_add
                i1 = 300000
                pc = 7
                continue
            elif pc == 7: # int_return
                bh.tmpreg_i = 300000
                bh._return_type = 'i'
                return -1
            else:
                return pc"""


def test_folding_wraps_around_like_blackhole():
    import sys
    i0 = Register('int', 0)
    assembler, ssarepr, jitcode = _assemble([
        ('int_copy', Constant(sys.maxint, lltype.Signed), '->', i0),
        ('int_add', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_return', i0),
        ], int=1)
    result = _check_matches_blackhole(assembler, ssarepr, jitcode)
    assert result == -sys.maxint - 1


def test_folding_of_comparisons_and_unary_ops():
    i0, i1, i2 = [Register('int', i) for i in range(3)]
    assembler, ssarepr, jitcode = _assemble([
        ('int_copy', Constant(5, lltype.Signed), '->', i0),
        ('int_lt', i0, Constant(9, lltype.Signed), '->', i1),
        ('int_neg', i0, '->', i2),
        ('int_add', i1, i2, '->', i2),
        ('int_return', i2),
        ], int=3)
    assert _check_matches_blackhole(assembler, ssarepr, jitcode) == -4
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    i2 = bh.registers_i[2]
    try:
        while 1:
            if pc == 0: # int_copy
                i0 = 5
                pc = 3
                continue
            elif pc == 3: # int_lt
                i1 = 1
                pc = 7
                continue
            elif pc == 7: # int_neg
                i2 = -5
                pc = 10
                continue
            elif pc == 10: # int_add
                i2 = -4
                pc = 14
                continue
            elif pc == 14: # int_return
                bh.tmpreg_i = -4
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0
        bh.registers_i[1] = i1
        bh.registers_i[2] = i2"""


def _folded_branch_jitcode(value):
    i0 = Register('int', 0)
    return _assemble([
        ('int_copy', Constant(value, lltype.Signed), '->', i0),
        ('goto_if_not_int_lt', i0, Constant(10, lltype.Signed),
         TLabel('nope')),
        ('int_return', Constant(1, lltype.Signed)),
        (Label('nope'),),
        ('int_return', Constant(2, lltype.Signed)),
        ], int=1)


def test_branch_on_known_values_is_resolved():
    for value, expected, target in [(5, 1, 8), (50, 2, 10)]:
        assembler, ssarepr, jitcode = _folded_branch_jitcode(value)
        assert _check_matches_blackhole(
            assembler, ssarepr, jitcode) == expected
        assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    try:
        while 1:
            if pc == 0: # int_copy
                i0 = %(value)d
                pc = 3
                continue
            elif pc == 3: # goto_if_not_int_lt
                pc = %(target)d
                continue
            elif pc == 8: # int_return
                bh.tmpreg_i = 1
                bh._return_type = 'i'
                return -1
            elif pc == 10: # int_return
                bh.tmpreg_i = 2
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0""" % dict(value=value, target=target)


def test_branch_on_unknown_value_is_kept():
    assembler, ssarepr, jitcode = _fused_branch_jitcode('lt')
    _run_function(assembler, ssarepr, jitcode)
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    while 1:
        if pc == 0: # goto_if_not_int_lt
            if i0 < i1:
                pc = 5
            else:
                pc = 7
            continue
        elif pc == 5: # int_return
            bh.tmpreg_i = i0
            bh._return_type = 'i'
            return -1
        elif pc == 7: # int_return
            bh.tmpreg_i = i1
            bh._return_type = 'i'
            return -1
        else:
            return pc"""


def _join_jitcode(v_then, v_else):
    i0, i1 = Register('int', 0), Register('int', 1)
    return _assemble([
        ('goto_if_not', i0, TLabel('else')),
        ('int_copy', Constant(v_then, lltype.Signed), '->', i1),
        ('goto', TLabel('join')),
        (Label('else'),),
        ('int_copy', Constant(v_else, lltype.Signed), '->', i1),
        (Label('join'),),
        ('int_return', i1),
        ], int=2)


def test_facts_do_not_cross_a_label():
    for v_then, v_else in [(1, 2), (7, 7)]:
        for cond, expected in [(0, v_else), (1, v_then)]:
            a, s_, j = _join_jitcode(v_then, v_else)
            assert _check_matches_blackhole(a, s_, j, cond) == expected
        assert j._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    try:
        while 1:
            if pc == 0: # goto_if_not
                if i0:
                    pc = 4
                else:
                    pc = 10
                continue
            elif pc == 4: # int_copy
                i1 = %d
                pc = 7
                continue
            elif pc == 7: # goto
                pc = 13
                continue
            elif pc == 10: # int_copy
                i1 = %d
                pc = 13
                continue
            elif pc == 13: # int_return
                bh.tmpreg_i = i1
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[1] = i1""" % (v_then, v_else)


def test_facts_survive_the_fallthrough_of_a_conditional_branch():
    i0, i1 = [Register('int', i) for i in range(2)]
    assembler, ssarepr, jitcode = _assemble([
        ('int_copy', Constant(5, lltype.Signed), '->', i1),
        ('goto_if_not', i0, TLabel('else')),
        ('int_return', i1),
        (Label('else'),),
        ('int_return', Constant(9, lltype.Signed)),
        ], int=2)
    for cond, expected in [(0, 9), (1, 5)]:
        assert _check_matches_blackhole(
            assembler, ssarepr, jitcode, cond) == expected
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    try:
        while 1:
            if pc == 0: # int_copy
                i1 = 5
                pc = 3
                continue
            elif pc == 3: # goto_if_not
                if i0:
                    pc = 7
                else:
                    pc = 9
                continue
            elif pc == 7: # int_return
                bh.tmpreg_i = 5
                bh._return_type = 'i'
                return -1
            elif pc == 9: # int_return
                bh.tmpreg_i = 9
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[1] = i1"""


def test_unfoldable_write_kills_the_constant():
    from rpython.rtyper.lltypesystem import llmemory, rstr
    i0 = Register('int', 0)
    assembler, ssarepr, jitcode = _assemble([
        ('int_copy', Constant(5, lltype.Signed), '->', i0),
        ('strlen', Register('ref', 0), '->', i0),
        ('int_return', i0),
        ], int=1, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    class CPU(object):
        def bh_strlen(self, string):
            return 3

    bh = getblackholeinterp(assembler.insns)
    bh.cpu = CPU()
    bh.setposition(jitcode, 0)
    s = rstr.mallocstr(3)
    bh.setarg_r(0, lltype.cast_opaque_ptr(llmemory.GCREF, s))
    assert jit_run(bh) == -1
    assert bh._final_result_anytype() == 3
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    r0 = bh.registers_r[0]
    try:
        while 1:
            if pc == 0: # int_copy
                i0 = 5
                pc = 3
                continue
            elif pc == 3: # strlen
                bh.position = 6
                i0 = plain_int(bh.cpu.bh_strlen(r0))
                pc = 6
                continue
            elif pc == 6: # int_return
                bh.tmpreg_i = i0
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0"""


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


def test_counting_loop_with_constants_is_not_unrolled():
    i0 = Register('int', 0)
    assembler, ssarepr, jitcode = _assemble([
        ('int_copy', Constant(0, lltype.Signed), '->', i0),
        (Label('loop'),),
        ('int_add', i0, Constant(1, lltype.Signed), '->', i0),
        ('goto_if_not_int_lt', i0, Constant(100, lltype.Signed),
         TLabel('done')),
        ('goto', TLabel('loop')),
        (Label('done'),),
        ('int_return', i0),
        ], int=1)
    assert _check_matches_blackhole(assembler, ssarepr, jitcode) == 100
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    try:
        while 1:
            if pc == 0: # int_copy
                i0 = 0
                pc = 3
                continue
            elif pc == 3: # int_add
                i0 = plain_int(glob0(i0, 1))
                pc = 7
                continue
            elif pc == 7: # goto_if_not_int_lt
                if i0 < 100:
                    pc = 12
                else:
                    pc = 15
                continue
            elif pc == 12: # goto
                pc = 3
                continue
            elif pc == 15: # int_return
                bh.tmpreg_i = i0
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0"""


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


def test_resuming_in_the_middle_sees_the_same_constants():
    assembler, ssarepr, jitcode = _const_chain_jitcode()
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, ssarepr._insns_pos[1])
    bh.setarg_i(0, 100000)
    assert jit_run(bh) == -1
    assert bh._final_result_anytype() == 300000


class FakeCallDescr(AbstractDescr):
    pass


class FakeFieldDescr(AbstractDescr):
    def __init__(self):
        self.cleared_structs = []

    def get_vinfo(self):
        return self

    def clear_vable_token(self, struct):
        self.cleared_structs.append(struct)


class FakeFieldCPU(object):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _getfield(self, kind, struct, descr):
        self.calls.append((kind, struct, descr))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def bh_getfield_gc_i(self, struct, descr):
        return self._getfield('gc_i', struct, descr)

    def bh_getfield_gc_r(self, struct, descr):
        return self._getfield('gc_r', struct, descr)

    def bh_getfield_raw_i(self, struct, descr):
        return self._getfield('raw_i', struct, descr)

    def _setfield(self, kind, struct, value, descr):
        self.calls.append((kind, struct, value, descr))
        if isinstance(self.result, Exception):
            raise self.result

    def bh_setfield_gc_i(self, struct, value, descr):
        self._setfield('gc_i', struct, value, descr)

    def bh_setfield_gc_r(self, struct, value, descr):
        self._setfield('gc_r', struct, value, descr)


@py.test.mark.parametrize('opname,kind,restype', [
    ('getfield_gc_i', 'r', 'i'),
    ('getfield_gc_i_pure', 'r', 'i'),
    ('getfield_gc_r', 'r', 'r'),
    ('getfield_gc_r_pure', 'r', 'r'),
    ('getfield_raw_i', 'i', 'i'),
    ('getfield_vable_i', 'r', 'i'),
    ('getfield_vable_r', 'r', 'r'),
])
def test_getfield_variants(opname, kind, restype):
    descr = FakeFieldDescr()
    result = 42 if restype == 'i' else _gcptr_value()
    value = _gcptr_value() if kind == 'r' else 12345
    arg = Register('ref' if kind == 'r' else 'int', 0)
    result_register = Register('int' if restype == 'i' else 'ref', 1)
    assembler, ssarepr, jitcode = _assemble([
        (opname, arg, descr, '->', result_register),
        (('int_return' if restype == 'i' else 'ref_return'),
         result_register),
        ], int=2, ref=2)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeFieldCPU(result)
        bh.setposition(jitcode, 0)
        if kind == 'r':
            bh.setarg_r(0, value)
        else:
            bh.setarg_i(0, value)
        if use_generated:
            jitcode.genext_run_function = jit_run
        bh.run()
        runs.append((cpu.calls, bh._final_result_anytype()))

    expected_call = ('raw_i' if 'raw' in opname else
                     'gc_' + restype, value, descr)
    assert runs[0] == runs[1] == ([expected_call], result)
    if 'vable' in opname:
        assert descr.cleared_structs == [value, value]


@py.test.mark.parametrize('opname,valuekind', [
    ('setfield_gc_i', 'i'),
    ('setfield_gc_r', 'r'),
    ('setfield_vable_i', 'i'),
])
def test_setfield_variants(opname, valuekind):
    descr = FakeFieldDescr()
    struct = _gcptr_value()
    value = 42 if valuekind == 'i' else _gcptr_value()
    value_register = Register('int' if valuekind == 'i' else 'ref',
                              0 if valuekind == 'i' else 1)
    assembler, ssarepr, jitcode = _assemble([
        (opname, Register('ref', 0), value_register, descr),
        ('void_return',),
        ], int=1, ref=1 if valuekind == 'i' else 2)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeFieldCPU(None)
        bh.setposition(jitcode, 0)
        bh.setarg_r(0, struct)
        if valuekind == 'i':
            bh.setarg_i(0, value)
        else:
            bh.setarg_r(1, value)
        if use_generated:
            assert jit_run(bh) == -1
        else:
            bh.run()
        runs.append((cpu.calls, bh._final_result_anytype()))

    for calls, result in runs:
        assert result is None
        assert len(calls) == 1
        kind, actual_struct, actual_value, actual_descr = calls[0]
        assert kind == 'gc_' + valuekind
        assert llmemory.cast_ptr_to_adr(actual_struct) == \
            llmemory.cast_ptr_to_adr(struct)
        if valuekind == 'i':
            assert actual_value == value
        else:
            assert llmemory.cast_ptr_to_adr(actual_value) == \
                llmemory.cast_ptr_to_adr(value)
        assert actual_descr is descr
    if 'vable' in opname:
        assert descr.cleared_structs == [struct, struct]


def test_setfield_exception_leaves_position_after_the_instruction():
    descr = FakeFieldDescr()
    assembler, ssarepr, jitcode = _assemble([
        ('setfield_gc_i', Register('ref', 0), Register('int', 0), descr),
        ('void_return',),
        ], int=1, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.cpu = FakeFieldCPU(ValueError())
    bh.setposition(jitcode, 0)
    bh.setarg_r(0, _gcptr_value())
    bh.setarg_i(0, 7)

    py.test.raises(ValueError, jit_run, bh)
    assert bh.position == ssarepr._insns_pos[1]


def _gcptr_value():
    from rpython.rtyper.lltypesystem import llmemory
    return lltype.cast_opaque_ptr(llmemory.GCREF,
                                  lltype.malloc(lltype.GcStruct('X')))


class FakeArrayDescr(AbstractDescr):
    pass


class FakeArrayCPU(object):
    def __init__(self, result, array):
        self.result = result
        self.array = array
        self.calls = []

    def bh_getfield_gc_r(self, vable, descr):
        self.calls.append(('getfield_gc_r', vable, descr))
        return self.array

    def bh_getarrayitem_gc_i(self, array, index, descr):
        self.calls.append(('getarrayitem_gc_i', array, index, descr))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def bh_getarrayitem_gc_r(self, array, index, descr):
        self.calls.append(('getarrayitem_gc_r', array, index, descr))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def _setarrayitem(self, kind, array, index, value, descr):
        self.calls.append(('setarrayitem_gc_' + kind,
                           array, index, value, descr))
        if isinstance(self.result, Exception):
            raise self.result

    def bh_setarrayitem_gc_i(self, array, index, value, descr):
        self._setarrayitem('i', array, index, value, descr)

    def bh_setarrayitem_gc_r(self, array, index, value, descr):
        self._setarrayitem('r', array, index, value, descr)


@py.test.mark.parametrize('opname,restype,vable', [
    ('getarrayitem_gc_i', 'i', False),
    ('getarrayitem_gc_i_pure', 'i', False),
    ('getarrayitem_gc_r', 'r', False),
    ('getarrayitem_gc_r_pure', 'r', False),
    ('getarrayitem_vable_r', 'r', True),
])
def test_getarrayitem_variants(opname, restype, vable):
    arraydescr = FakeArrayDescr()
    fielddescr = FakeFieldDescr()
    array = _gcptr_value()
    vable_value = _gcptr_value()
    result = 42 if restype == 'i' else _gcptr_value()
    index = Register('int', 0)
    args = [Register('ref', 0) if vable else Register('ref', 1), index]
    if vable:
        args.extend([fielddescr, arraydescr])
    else:
        args.append(arraydescr)
    result_register = Register('int' if restype == 'i' else 'ref', 1)
    insn = tuple([opname] + args + ['->', result_register])
    return_insn = (('int_return' if restype == 'i' else 'ref_return'),
                   result_register)
    assembler, ssarepr, jitcode = _assemble(
        [insn, return_insn], int=2, ref=2)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeArrayCPU(result, array)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, 7)
        if vable:
            bh.setarg_r(0, vable_value)
        else:
            bh.setarg_r(1, array)
        if use_generated:
            jitcode.genext_run_function = jit_run
        bh.run()
        runs.append((cpu.calls, bh._final_result_anytype()))

    if vable:
        expected_calls = [
            ('getfield_gc_r', vable_value, fielddescr),
            ('getarrayitem_gc_r', array, 7, arraydescr),
        ]
        assert fielddescr.cleared_structs == [vable_value, vable_value]
    else:
        expected_calls = [
            ('getarrayitem_gc_' + restype, array, 7, arraydescr)]
    assert runs[0] == runs[1] == (expected_calls, result)


def test_getarrayitem_exception_leaves_position_after_instruction():
    arraydescr = FakeArrayDescr()
    assembler, ssarepr, jitcode = _assemble([
        ('getarrayitem_gc_i', Register('ref', 0), Register('int', 0),
         arraydescr, '->', Register('int', 1)),
        ('int_return', Register('int', 1)),
        ], int=2, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.cpu = FakeArrayCPU(ValueError(), _gcptr_value())
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 7)
    bh.setarg_r(0, _gcptr_value())

    py.test.raises(ValueError, jit_run, bh)
    assert bh.position == ssarepr._insns_pos[1]


@py.test.mark.parametrize('opname,valuekind,vable', [
    ('setarrayitem_gc_i', 'i', False),
    ('setarrayitem_gc_r', 'r', False),
    ('setarrayitem_vable_i', 'i', True),
    ('setarrayitem_vable_r', 'r', True),
])
def test_setarrayitem_variants(opname, valuekind, vable):
    arraydescr = FakeArrayDescr()
    fielddescr = FakeFieldDescr()
    array = _gcptr_value()
    vable_value = _gcptr_value()
    value = 42 if valuekind == 'i' else _gcptr_value()
    value_register = Register('int' if valuekind == 'i' else 'ref',
                              1 if valuekind == 'i' else 2)
    args = [Register('ref', 0) if vable else Register('ref', 1),
            Register('int', 0), value_register]
    if vable:
        args.extend([fielddescr, arraydescr])
    else:
        args.append(arraydescr)
    assembler, ssarepr, jitcode = _assemble(
        [tuple([opname] + args), ('void_return',)],
        int=2, ref=3)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeArrayCPU(None, array)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, 7)
        if vable:
            bh.setarg_r(0, vable_value)
        else:
            bh.setarg_r(1, array)
        if valuekind == 'i':
            bh.setarg_i(1, value)
        else:
            bh.setarg_r(2, value)
        if use_generated:
            assert jit_run(bh) == -1
        else:
            bh.run()
        runs.append(cpu.calls)

    for calls in runs:
        if vable:
            expected_calls = [
                ('getfield_gc_r', vable_value, fielddescr),
                ('setarrayitem_gc_' + valuekind, array, 7, value, arraydescr),
            ]
            assert fielddescr.cleared_structs[-1] == vable_value
        else:
            expected_calls = [
                ('setarrayitem_gc_' + valuekind, array, 7, value, arraydescr)]
        assert len(calls) == len(expected_calls)
        for actual, expected in zip(calls, expected_calls):
            assert actual[0] == expected[0]
            if actual[0] == 'setarrayitem_gc_' + valuekind:
                assert llmemory.cast_ptr_to_adr(actual[1]) == \
                    llmemory.cast_ptr_to_adr(expected[1])
                assert actual[2] == expected[2]
                if valuekind == 'i':
                    assert actual[3] == expected[3]
                else:
                    assert llmemory.cast_ptr_to_adr(actual[3]) == \
                        llmemory.cast_ptr_to_adr(expected[3])
                assert actual[4] is expected[4]
            else:
                assert llmemory.cast_ptr_to_adr(actual[1]) == \
                    llmemory.cast_ptr_to_adr(expected[1])
                assert actual[2] is expected[2]


def test_setarrayitem_exception_leaves_position_after_instruction():
    arraydescr = FakeArrayDescr()
    assembler, ssarepr, jitcode = _assemble([
        ('setarrayitem_gc_i', Register('ref', 0), Register('int', 0),
         Register('int', 1), arraydescr),
        ('void_return',),
        ], int=2, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.cpu = FakeArrayCPU(ValueError(), _gcptr_value())
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 7)
    bh.setarg_i(1, 42)
    bh.setarg_r(0, _gcptr_value())

    py.test.raises(ValueError, jit_run, bh)
    assert bh.position == ssarepr._insns_pos[1]


def test_getfield_exception_leaves_position_after_the_instruction():
    descr = FakeFieldDescr()
    assembler, ssarepr, jitcode = _assemble([
        ('getfield_gc_i', Register('ref', 0), descr,
         '->', Register('int', 1)),
        ('int_return', Register('int', 1)),
        ], int=2, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.cpu = FakeFieldCPU(ValueError())
    bh.setposition(jitcode, 0)
    bh.setarg_r(0, _gcptr_value())

    py.test.raises(ValueError, jit_run, bh)
    assert bh.position == ssarepr._insns_pos[1]


class FakeCallCPU(object):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _call(self, kind, func, args_i, args_r, args_f, descr):
        self.calls.append((kind, func, args_i, args_r, args_f, descr))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def bh_call_i(self, *args):
        return self._call('i', *args)

    def bh_call_r(self, *args):
        return self._call('r', *args)

    def bh_call_v(self, *args):
        return self._call('v', *args)


def _call_insn(family, argtypes, restype, callee, descr):
    if family == 'residual':
        args = [Constant(12345, lltype.Signed)]
    else:
        args = [callee]
    for kind in argtypes:
        if kind == 'i':
            args.append(ListOfKind(
                'int', [Register('int', 0),
                        Constant(99999, lltype.Signed)]))
        else:
            assert kind == 'r'
            args.append(ListOfKind('ref', [Register('ref', 0)]))
    if family == 'residual':
        args.append(descr)
    result_register = None
    if restype != 'v':
        index = 1
        result_register = Register('int' if restype == 'i' else 'ref', index)
        args.extend(['->', result_register])
    return tuple([family + '_call_' + argtypes + '_' + restype] + args), \
        result_register


def _check_cpu_call(family, argtypes, restype, has_run_emitter):
    descr = FakeCallDescr()
    callee = JitCode("callee", llmemory.NULL, descr)
    insn, result_register = _call_insn(
        family, argtypes, restype, callee, descr)
    return_insn = (('int_return' if restype == 'i' else 'ref_return'),
                   result_register) if result_register else ('void_return',)
    assembler, ssarepr, jitcode = _assemble(
        [insn, return_insn], int=2, ref=2)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    ref_value = _gcptr_value()
    result = {'i': 42, 'r': ref_value, 'v': None}[restype]
    expected_args_i = [7, 99999] if 'i' in argtypes else None
    expected_args_r = [ref_value] if 'r' in argtypes else None
    func = (12345 if family == 'residual' else
            callee.get_fnaddr_as_int())
    expected_call = (restype, func, expected_args_i, expected_args_r,
                     None, descr)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeCallCPU(result)
        bh.setposition(jitcode, 0)
        if 'i' in argtypes:
            bh.setarg_i(0, 7)
        if 'r' in argtypes:
            bh.setarg_r(0, ref_value)
        if use_generated:
            jitcode.genext_run_function = jit_run
        bh.run()
        value = (bh._final_result_anytype() if result_register else None)
        runs.append((cpu.calls, value))
    assert runs[0] == runs[1] == ([expected_call], result)

    if has_run_emitter:
        assert 'bh.cpu.bh_call_%s(' % restype in jitcode._genext_run_source
    else:
        assert 'return 0' in jitcode._genext_run_source


@py.test.mark.parametrize('argtypes,restype,has_run_emitter', [
    ('r', 'i', True), ('r', 'r', True), ('r', 'v', False),
    ('ir', 'i', True), ('ir', 'r', False), ('ir', 'v', False),
])
def test_residual_call_variants(argtypes, restype, has_run_emitter):
    _check_cpu_call('residual', argtypes, restype, has_run_emitter)


def test_residual_call_exception_leaves_position_after_the_call():
    assembler, ssarepr, jitcode = _assemble([
        ('residual_call_r_i', Constant(12345, lltype.Signed),
         ListOfKind('ref', []), FakeCallDescr(), '->', Register('int', 0)),
        ('int_return', Register('int', 0)),
        ], int=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.cpu = FakeCallCPU(ValueError())
    bh.setposition(jitcode, 0)
    py.test.raises(ValueError, jit_run, bh)
    assert bh.position == ssarepr._insns_pos[1]
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    try:
        while 1:
            if pc == 0: # residual_call_r_i
                bh.position = 6
                i0 = plain_int(bh.cpu.bh_call_i(12345, None, [], None, glob0))
                pc = 6
                continue
            elif pc == 6: # int_return
                bh.tmpreg_i = i0
                bh._return_type = 'i'
                return -1
            else:
                return pc
    finally:
        bh.registers_i[0] = i0"""


@py.test.mark.parametrize('argtypes,restype', [
    ('r', 'i'), ('r', 'r'), ('r', 'v'),
    ('ir', 'i'), ('ir', 'r'), ('ir', 'v'),
])
def test_inline_call_variants(argtypes, restype):
    _check_cpu_call('inline', argtypes, restype, True)


@py.test.mark.parametrize('restype', ['i', 'r', 'v'])
def test_recursive_call_variants(restype):
    result = {'i': 42, 'r': _gcptr_value(), 'v': None}[restype]
    result_register = (Register('int' if restype == 'i' else 'ref', 2)
                       if restype != 'v' else None)
    insn = ('recursive_call_' + restype,
            Constant(12345, lltype.Signed),
            ListOfKind('int', [Register('int', 0)]),
            ListOfKind('ref', [Register('ref', 0)]),
            ListOfKind('float', []),
            ListOfKind('int', [Register('int', 1)]),
            ListOfKind('ref', []),
            ListOfKind('float', []))
    if result_register:
        insn += ('->', result_register)
        return_insn = (('int_return' if restype == 'i' else 'ref_return'),
                       result_register)
    else:
        return_insn = ('void_return',)
    assembler, ssarepr, jitcode = _assemble(
        [insn, return_insn], int=3, ref=3)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    bh = getblackholeinterp(assembler.insns, assembler.descrs)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 7)
    bh.setarg_i(1, 11)
    ref_value = _gcptr_value()
    bh.setarg_r(0, ref_value)
    calls = []

    def recursive_call(*args):
        calls.append(args)
        return result
    setattr(bh, 'bhimpl_recursive_call_' + restype, recursive_call)
    jitcode.genext_run_function = jit_run
    bh.run()

    assert calls == [
        (12345, [7], [ref_value], [], [11], [], [])]
    assert (bh._final_result_anytype() if result_register else None) == result
    assert 'bh.bhimpl_recursive_call_%s(' % restype in (
        jitcode._genext_run_source)

def test_portal_point():
    # TODO: @Yusuke: Check this
    ssarepr = SSARepr("test")
    i0 = Register('int', 0)
    ssarepr.insns = [
        ('jit_merge_point', Constant(7, lltype.Signed),
         ListOfKind('int', []), ListOfKind('ref', []),
         ListOfKind('float', []), ListOfKind('int', [i0]),
         ListOfKind('ref', []), ListOfKind('float', [])),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    jit_run = _run_function(assembler, ssarepr, jitcode)

    bh = getblackholeinterp(assembler.insns)
    bh.nextblackholeinterp = None
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 42)
    exc = py.test.raises(jitexc.ContinueRunningNormally, jit_run, bh)
    assert exc.value.green_int == []
    assert exc.value.green_ref == []
    assert exc.value.green_float == []
    assert exc.value.red_int == [42]
    assert exc.value.red_ref == []
    assert exc.value.red_float == []
    assert bh.position == len(jitcode.code)
    assert jitcode._genext_run_source == """def jit_run(bh): # test
    pc = bh.position
    i0 = bh.registers_i[0]
    while 1:
        if pc == 0: # jit_merge_point
            bh.position = 9
            try:
                bh.bhimpl_jit_merge_point(7, [], [], [], [i0], [], [])
            except LeaveFrame:
                return -1
        else:
            return pc"""

def test_tla_loop():
    from rpython.jit.codewriter.test.test_genext_scaffold import (
        _tla_jitcodes)
    assembler, captured = _tla_jitcodes()
    ssarepr, jitcode, snap = [(s, j, a) for s, j, a in captured
                              if s.name == 'Frame.interp'][0]
    genext = GenExtension(snap, ssarepr, jitcode)
    genext.generate()
    generate_run_function(genext)


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
    assert bh.registers_i[0] == 2


def test_run_single_steps_a_fallback_and_reenters_jit_run():
    i0 = Register('int', 0)
    assembler, ssarepr, jitcode = _assemble([
        ('int_add', i0, i0, '->', i0),
        ('int_push', i0),
        ('int_add', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_return', i0),
        ], int=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    entries = []

    def counting(bh):
        entries.append(bh.position)
        return jit_run(bh)

    jitcode.genext_run_function = counting
    bh = getblackholeinterp(assembler.insns)
    bh.setposition(jitcode, 0)
    bh.setarg_i(0, 1)
    bh.run()
    assert bh._final_result_anytype() == 3
    assert entries == [0, ssarepr._insns_pos[2]]


class FakeNewCPU(object):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _new(self, kind, args):
        self.calls.append((kind,) + args)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def bh_new(self, descr):
        return self._new('new', (descr,))

    def bh_new_with_vtable(self, descr):
        return self._new('new_with_vtable', (descr,))

    def bh_new_array(self, length, descr):
        return self._new('new_array', (length, descr))

    def bh_new_array_clear(self, length, descr):
        return self._new('new_array_clear', (length, descr))


@py.test.mark.parametrize('opname,has_length', [
    ('new', False),
    ('new_with_vtable', False),
    ('new_array', True),
    ('new_array_clear', True),
])
def test_new_variants(opname, has_length):
    descr = FakeFieldDescr()
    result = _gcptr_value()
    args = (Register('int', 0),) if has_length else ()
    assembler, ssarepr, jitcode = _assemble([
        (opname,) + args + (descr, '->', Register('ref', 0)),
        ('ref_return', Register('ref', 0)),
        ], int=1, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeNewCPU(result)
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, 7)
        if use_generated:
            assert jit_run(bh) == -1
            final = bh.tmpreg_r
        else:
            bh.run()
            final = bh._final_result_anytype()
        runs.append((cpu.calls, final))

    expected = ((opname, 7, descr) if has_length else (opname, descr))
    for calls, final in runs:
        assert calls == [expected]
        assert llmemory.cast_ptr_to_adr(final) == \
            llmemory.cast_ptr_to_adr(result)


class FakeStringCPU(object):
    """Records every bh_* call; returns self.result"""
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith('bh_'):
            raise AttributeError(name)
        def method(*args):
            self.calls.append((name,) + args)
            if isinstance(self.result, Exception):
                raise self.result
            return self.result
        return method


_STRING_OPS = [
    # opname, argument kinds, result kind
    ('newstr', 'i', 'r'),
    ('strlen', 'r', 'i'),
    ('strgetitem', 'ri', 'i'),
    ('strsetitem', 'rii', None),
    ('copystrcontent', 'rriii', None),
]


@py.test.mark.parametrize('opname,argkinds,resultkind', _STRING_OPS)
def test_string_ops_reach_cpu(opname, argkinds, resultkind):
    names = {'i': 'int', 'r': 'ref'}
    counts = {'i': 0, 'r': 0}
    regs = []
    values = []
    for kind in argkinds:
        regs.append(Register(names[kind], counts[kind]))
        values.append(_gcptr_value() if kind == 'r' else 10 + counts[kind])
        counts[kind] += 1
    insn = [opname] + regs
    if resultkind is None:
        instructions = [tuple(insn), ('void_return',)]
        result = None
    else:
        res = Register(names[resultkind], counts[resultkind])
        counts[resultkind] += 1
        insn += ['->', res]
        ret = 'int_return' if resultkind == 'i' else 'ref_return'
        instructions = [tuple(insn), (ret, res)]
        result = 99 if resultkind == 'i' else _gcptr_value()
    assembler, ssarepr, jitcode = _assemble(
        instructions, int=counts['i'], ref=counts['r'])
    jit_run = _run_function(assembler, ssarepr, jitcode)

    runs = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = cpu = FakeStringCPU(result)
        bh.setposition(jitcode, 0)
        ri = rr = 0
        for kind, value in zip(argkinds, values):
            if kind == 'i':
                bh.setarg_i(ri, value)
                ri += 1
            else:
                bh.setarg_r(rr, value)
                rr += 1
        if use_generated:
            assert jit_run(bh) == -1
            final = {'i': bh.tmpreg_i, 'r': bh.tmpreg_r,
                     None: None}[resultkind]
        else:
            bh.run()
            final = bh._final_result_anytype()
        runs.append((cpu.calls, final))

    for calls, final in runs:
        assert len(calls) == 1
        assert calls[0][0] == 'bh_' + opname
        for kind, actual, expected in zip(argkinds, calls[0][1:], values):
            if kind == 'r':
                assert llmemory.cast_ptr_to_adr(actual) == \
                    llmemory.cast_ptr_to_adr(expected)
            else:
                assert actual == expected
        if resultkind == 'r':
            assert llmemory.cast_ptr_to_adr(final) == \
                llmemory.cast_ptr_to_adr(result)
        else:
            assert final == result


def _exception_object(vtable=None):
    from rpython.rtyper import rclass
    obj = lltype.malloc(rclass.OBJECT, immortal=True)
    if vtable is not None:
        obj.typeptr = vtable
    return obj


def _vtable(minid, maxid):
    from rpython.rtyper import rclass
    vtable = lltype.malloc(rclass.OBJECT_VTABLE, immortal=True)
    vtable.subclassrange_min = minid
    vtable.subclassrange_max = maxid
    return vtable


def _run_both(assembler, jitcode, jit_run, setup):
    """Run via blackhole dispatch and via jit_run; returns both outcomes
    as (final result, or the exception raised out of the frame)"""
    from rpython.rtyper.llinterp import LLException
    outcomes = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.setposition(jitcode, 0)
        setup(bh)
        jitcode.genext_run_function = jit_run if use_generated else None
        try:
            bh.run()
            outcomes.append(bh._final_result_anytype())
        except LLException as e:
            outcomes.append(e.args[1])
    jitcode.genext_run_function = None
    return outcomes


def _catching_program(raising_insn, **num_regs):
    return _assemble([
        raising_insn,
        ('catch_exception', TLabel('handler')),
        ('int_return', Constant(0, lltype.Signed)),
        (Label('handler'),),
        ('int_return', Constant(42, lltype.Signed)),
        ], **num_regs)


def test_raise_is_caught_by_catch_exception():
    exc = _exception_object()
    assembler, ssarepr, jitcode = _catching_program(
        ('raise', Register('ref', 0)), ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert 'bhimpl_raise' in jitcode._genext_run_source

    def setup(bh):
        bh.setarg_r(0, lltype.cast_opaque_ptr(llmemory.GCREF, exc))
    assert _run_both(assembler, jitcode, jit_run, setup) == [42, 42]


def test_raise_escaping_the_frame():
    exc = _exception_object()
    assembler, ssarepr, jitcode = _assemble([
        ('raise', Register('ref', 0)),
        ], ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    def setup(bh):
        bh.setarg_r(0, lltype.cast_opaque_ptr(llmemory.GCREF, exc))
    outcomes = _run_both(assembler, jitcode, jit_run, setup)
    assert outcomes[0] == outcomes[1] == exc


def test_reraise():
    exc = _exception_object()
    assembler, ssarepr, jitcode = _catching_program(('reraise',))
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert 'bhimpl_reraise' in jitcode._genext_run_source

    def setup(bh):
        bh.exception_last_value = exc
    assert _run_both(assembler, jitcode, jit_run, setup) == [42, 42]


def test_exception_from_cpu_call_is_caught():
    from rpython.rtyper.llinterp import LLException
    exc = _exception_object()
    assembler, ssarepr, jitcode = _catching_program(
        ('setfield_gc_i', Register('ref', 0), Register('int', 0),
         FakeFieldDescr()), int=1, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)

    def run(result, use_generated):
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.cpu = FakeFieldCPU(result)
        bh.setposition(jitcode, 0)
        bh.setarg_r(0, _gcptr_value())
        bh.setarg_i(0, 1)
        jitcode.genext_run_function = jit_run if use_generated else None
        bh.run()
        return bh._final_result_anytype()
    for use_generated in [False, True]:
        assert run(None, use_generated) == 0
        assert run(LLException("etype", exc), use_generated) == 42
    jitcode.genext_run_function = None


@py.test.mark.parametrize('matches', [True, False])
def test_goto_if_exception_mismatch(matches):
    from rpython.jit.metainterp.support import ptr2int
    bounding = _vtable(1, 4)
    actual = _vtable(2, 3) if matches else _vtable(7, 8)
    exc = _exception_object(actual)
    assembler, ssarepr, jitcode = _assemble([
        ('goto_if_exception_mismatch', Register('int', 0),
         TLabel('mismatch')),
        ('int_return', Constant(1, lltype.Signed)),
        (Label('mismatch'),),
        ('int_return', Constant(2, lltype.Signed)),
        ], int=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert 'bhimpl_goto_if_exception_mismatch' in jitcode._genext_run_source

    def setup(bh):
        bh.exception_last_value = exc
        bh.setarg_i(0, ptr2int(bounding))
    assert _run_both(assembler, jitcode, jit_run, setup) == \
        [1 if matches else 2] * 2


def test_last_exception_and_last_exc_value():
    from rpython.jit.metainterp.support import ptr2int
    vtable = _vtable(1, 2)
    exc = _exception_object(vtable)
    assembler, ssarepr, jitcode = _assemble([
        ('last_exception', '->', Register('int', 0)),
        ('last_exc_value', '->', Register('ref', 0)),
        ('int_return', Register('int', 0)),
        ], int=1, ref=1)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    source = jitcode._genext_run_source
    assert 'bhimpl_last_exception' in source
    assert 'bhimpl_last_exc_value' in source

    results = []
    for use_generated in [False, True]:
        bh = getblackholeinterp(assembler.insns, assembler.descrs)
        bh.setposition(jitcode, 0)
        bh.exception_last_value = exc
        if use_generated:
            assert jit_run(bh) == -1
        else:
            bh.run()
        assert llmemory.cast_ptr_to_adr(bh.registers_r[0]) == \
            llmemory.cast_ptr_to_adr(lltype.cast_opaque_ptr(
                llmemory.GCREF, exc))
        results.append(bh.registers_i[0])
    assert results == [ptr2int(vtable)] * 2


def _switch_program(value):
    from rpython.jit.codewriter.jitcode import SwitchDictDescr
    descr = SwitchDictDescr()
    descr._labels = [(1, TLabel('one')), (5, TLabel('five'))]
    argument = Register('int', 0) if value is None else Constant(
        value, lltype.Signed)
    return _assemble([
        ('switch', argument, descr),
        ('int_return', Constant(0, lltype.Signed)),
        (Label('one'),),
        ('int_return', Constant(10, lltype.Signed)),
        (Label('five'),),
        ('int_return', Constant(50, lltype.Signed)),
        ], int=1)


@py.test.mark.parametrize('value,expected', [(1, 10), (5, 50), (3, 0)])
def test_switch_on_a_register(value, expected):
    assembler, ssarepr, jitcode = _switch_program(None)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert '.get(' in jitcode._genext_run_source

    def setup(bh):
        bh.setarg_i(0, value)
    assert _run_both(assembler, jitcode, jit_run, setup) == [expected] * 2


@py.test.mark.parametrize('value,expected', [(1, 10), (5, 50), (3, 0)])
def test_switch_on_a_constant_is_resolved_statically(value, expected):
    assembler, ssarepr, jitcode = _switch_program(value)
    jit_run = _run_function(assembler, ssarepr, jitcode)
    assert '.get(' not in jitcode._genext_run_source
    assert _run_both(assembler, jitcode, jit_run, lambda bh: None) == \
        [expected] * 2
