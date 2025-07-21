from rpython.flowspace.model import Constant
from rpython.jit.codewriter.jitcode import SwitchDictDescr
from rpython.jit.codewriter.flatten import SSARepr, Label, TLabel, Register
from rpython.jit.codewriter.assembler import Assembler, AssemblerError
from rpython.rtyper.lltypesystem import lltype, llmemory
from rpython.jit.codewriter.genextension import WorkList

import pytest


def test_assemble_loop():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0x16), Register('int', 0x17)
    ssarepr.insns = [
        (Label('L1'),),
        ('goto_if_not_int_gt', i0, Constant(4, lltype.Signed), TLabel('L2')),
        ('int_add', i1, i0, '->', i1),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('goto', TLabel('L1')),
        (Label('L2'),),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 0x18})
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    i22 = 0xcafedead
    i23 = 0xcafedead
    if pc == 0: pc = 0
    elif pc == 5: pc = 5
    elif pc == 9: pc = 9
    elif pc == 13: pc = 13
    elif pc == 16: pc = 16
    else: assert 0, 'unreachable'
    while 1:
        if pc == 0: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([])
            self.pc = 5
            ri22 = self.registers_i[22]
            if isinstance(ri22, ConstInt):
                i22 = ri22.getint()
                pc = 116
                continue
            condbox = self.opimpl_int_gt(ri22, ConstInt(4))
            self.opimpl_goto_if_not(condbox, 16, 0)
            pc = self.pc
            if pc == 16:
                pc = 16
            else:
                assert self.pc == 5
                pc = 5
            continue
        if pc == 5: # ('int_add', %i23, %i22, '->', %i23) frozenset([])
            self.pc = 9
            ri23 = self.registers_i[23]
            ri22 = self.registers_i[22]
            if isinstance(ri23, ConstInt) and isinstance(ri22, ConstInt):
                i23 = ri23.getint()
                i22 = ri22.getint()
                pc = 117
                continue
            else:
                self.registers_i[23] = self.opimpl_int_add(ri23, ri22)
            pc = 9
            continue
        if pc == 9: # ('int_sub', %i22, (1), '->', %i22) frozenset([])
            self.pc = 13
            ri22 = self.registers_i[22]
            if isinstance(ri22, ConstInt):
                i22 = ri22.getint()
                pc = 118
                continue
            else:
                self.registers_i[22] = self.opimpl_int_sub(ri22, ConstInt(1))
            pc = 0
            continue
        if pc == 13: # ('goto', TLabel('L1')) frozenset([])
            self.pc = 16
            pc = 0
            continue
        if pc == 16: # ('int_return', %i23) frozenset([])
            self.pc = 18
            ri23 = self.registers_i[23]
            try:
                self.opimpl_int_return(ri23)
            except ChangeFrame: return
        if pc == 116: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([%i22])
            self.pc = 5
            cond = i22 > 4
            if not cond:
                pc = 119
                continue
            pc = 120
            continue
        if pc == 117: # ('int_add', %i23, %i22, '->', %i23) frozenset([%i23, %i22])
            self.pc = 9
            i23 = i23 + i22
            pc = 121
            continue
        if pc == 118: # ('int_sub', %i22, (1), '->', %i22) frozenset([%i22])
            self.pc = 13
            i22 = i22 - 1
            pc = 116
            continue
        if pc == 119: # ('int_return', %i23) frozenset([%i22])
            self.pc = 18
            ri23 = self.registers_i[23]
            try:
                self.opimpl_int_return(ri23)
            except ChangeFrame: return
        if pc == 120: # ('int_add', %i23, %i22, '->', %i23) frozenset([%i22])
            self.pc = 9
            ri23 = self.registers_i[23]
            if isinstance(ri23, ConstInt):
                i23 = ri23.getint()
                pc = 117
                continue
            else:
                self.registers_i[23] = self.opimpl_int_add(ri23, ConstInt(i22))
            pc = 118
            continue
        if pc == 121: # ('int_sub', %i22, (1), '->', %i22) frozenset([%i23, %i22])
            self.pc = 13
            i22 = i22 - 1
            pc = 122
            continue
        if pc == 122: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([%i23, %i22])
            self.pc = 5
            cond = i22 > 4
            if not cond:
                pc = 123
                continue
            pc = 117
            continue
        if pc == 123: # ('int_return', %i23) frozenset([%i23, %i22])
            self.pc = 18
            ri23 = self.registers_i[23]
            try:
                self.opimpl_int_return(ConstInt(i23))
            except ChangeFrame: return
        assert 0 # unreachable"""

def test_integration_switch():
    ssarepr = SSARepr("test", genextension=True)
    i0 = Register('int', 0x16)
    switchdescr = SwitchDictDescr()
    switchdescr._labels = [(-5, Label("L1")), (2, Label("L2")),
                           (7, Label("L3"))]
    ssarepr.insns = [
        (Label("L0"),),
        ('-live-', i0),
        ('switch', i0, switchdescr),
        ('int_return', Constant(42, lltype.Signed)),
        ('---',),
        (Label("L1"),),
        ('-live-',),
        ('int_return', Constant(12, lltype.Signed)),
        ('---',),
        (Label("L2"),),
        ('-live-',),
        ('int_return', Constant(51, lltype.Signed)),
        ('---',),
        (Label("L3"),),
        ('-live-',),
        ('int_return', Constant(1212, lltype.Signed)),
        ('---',),
    ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 0x17})
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    i22 = 0xcafedead
    if pc == 0: pc = 0
    elif pc == 3: pc = 3
    elif pc == 7: pc = 7
    elif pc == 9: pc = 9
    elif pc == 12: pc = 12
    elif pc == 14: pc = 14
    elif pc == 17: pc = 17
    elif pc == 19: pc = 19
    elif pc == 22: pc = 22
    else: assert 0, 'unreachable'
    while 1:
        if pc == 0: # ('-live-', %i22) frozenset([])
            self.pc = 3
            pc = 3
            continue
        if pc == 3: # ('switch', %i22, <SwitchDictDescr {-5: 9, 2: 14, 7: 19}>) frozenset([])
            self.pc = 7
            ri22 = self.registers_i[22]
            if isinstance(ri22, ConstInt):
                i22 = ri22.getint()
                pc = 122
                continue
            self.opimpl_switch(ri22, glob2, 3)
            pc = self.pc
            if pc == 9: pc = 9
            elif pc == 14: pc = 14
            elif pc == 19: pc = 19
            elif pc == 7: pc = 7
            else: assert 0
            continue
        if pc == 7: # ('int_return', (42)) frozenset([])
            self.pc = 9
            try:
                self.opimpl_int_return(ConstInt(42))
            except ChangeFrame: return
        if pc == 9: # ('-live-',) frozenset([])
            self.pc = 12
            pc = 12
            continue
        if pc == 12: # ('int_return', (12)) frozenset([])
            self.pc = 14
            try:
                self.opimpl_int_return(ConstInt(12))
            except ChangeFrame: return
        if pc == 14: # ('-live-',) frozenset([])
            self.pc = 17
            pc = 17
            continue
        if pc == 17: # ('int_return', (51)) frozenset([])
            self.pc = 19
            try:
                self.opimpl_int_return(ConstInt(51))
            except ChangeFrame: return
        if pc == 19: # ('-live-',) frozenset([])
            self.pc = 22
            pc = 22
            continue
        if pc == 22: # ('int_return', (1212)) frozenset([])
            self.pc = 24
            try:
                self.opimpl_int_return(ConstInt(1212))
            except ChangeFrame: return
        if pc == 122: # ('switch', %i22, <SwitchDictDescr {-5: 9, 2: 14, 7: 19}>) frozenset([%i22])
            self.pc = 7
            if i22 == -5:
                pc = 12
                continue
            elif i22 == 2:
                pc = 17
                continue
            elif i22 == 7:
                pc = 22
                continue
            pc = 123
            continue
        if pc == 123: # ('int_return', (42)) frozenset([%i22])
            self.pc = 9
            try:
                self.opimpl_int_return(ConstInt(42))
            except ChangeFrame: return
        assert 0 # unreachable"""

@pytest.mark.xfail()
def test_skip_jump_to_live():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0x0), Register('int', 0x1)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_add', i1, i0, '->', i1),
        ('-live-', i1, i0), # goal: make int_add jump to 'goto_if_not_int_gt'
        ('goto_if_not_int_gt', i0, Constant(0, lltype.Signed), TLabel('L2')),
        ('goto', TLabel('L1')),
        ('---',),
        (Label('L2'),),
        ('int_return', i1),
        ('---',)]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    while 1:
        if pc == 0: # ('int_sub', %i0, (1), '->', %i0)
            self.pc = 4
            self._result_argcode = 'i'
            self.registers_i[0] = self.opimpl_int_sub(self.registers_i[0], ConstInt(1))
            pc = 4
            continue
        if pc == 4: # ('int_add', %i1, %i0, '->', %i1)
            self.pc = 8
            self._result_argcode = 'i'
            self.registers_i[1] = self.opimpl_int_add(self.registers_i[1], self.registers_i[0])
            pc = 11
            continue
        if pc == 8: # ('-live-', %i1, %i0)
            self.pc = 11
            pass # live
            pc = 11
            continue
        if pc == 11: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2'))
            self.pc = 16
            self._result_argcode = 'v'
            self.opimpl_goto_if_not_int_gt(self.registers_i[0], ConstInt(0), 19, 11)
            pc = self.pc
            if pc == 16: pc = 0
            elif pc == 19: pc = 19
            else:
                assert 0 # unreachable
            continue
        if pc == 16: # ('goto', TLabel('L1'))
            self.pc = 19
            pc = self.pc = 0 # goto
            continue
            pc = 0
            continue
        if pc == 19: # ('int_return', %i1)
            self.pc = 21
            try:
                self.opimpl_int_return(self.registers_i[1])
            except ChangeFrame: return
            assert 0 # unreachable
        assert 0 # unreachable"""


@pytest.mark.xfail()
def test_skip_conditional_jump():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0x0), Register('int', 0x1)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_add', i1, i0, '->', i1),
        ('-live-', i1, i0), # goal: make int_add jump to 'goto_if_not_int_gt'
        ('goto_if_not_int_gt', i0, Constant(0, lltype.Signed), TLabel('L2')),
        ('goto', TLabel('L1')),
        ('---',),
        (Label('L2'),),
        ('-live-', i1, i0),     # TODO
        (Label('L3'),),         # optimize -live- and goto L4 chan
        ('goto', TLabel('L4')), # here
        (Label('L4'),),
        ('int_return', i1),
        ('---',)]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    while 1:
        if pc == 0: # ('int_sub', %i0, (1), '->', %i0)
            self.pc = 4
            self._result_argcode = 'i'
            self.registers_i[0] = self.opimpl_int_sub(self.registers_i[0], ConstInt(1))
            pc = 4
            continue
        if pc == 4: # ('int_add', %i1, %i0, '->', %i1)
            self.pc = 8
            self._result_argcode = 'i'
            self.registers_i[1] = self.opimpl_int_add(self.registers_i[1], self.registers_i[0])
            pc = 11
            continue
        if pc == 8: # ('-live-', %i1, %i0)
            self.pc = 11
            pass # live
            pc = 11
            continue
        if pc == 11: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2'))
            self.pc = 16
            self._result_argcode = 'v'
            self.opimpl_goto_if_not_int_gt(self.registers_i[0], ConstInt(0), 19, 11)
            pc = self.pc
            if pc == 16: pc = 0
            elif pc == 19: pc = 22
            else:
                assert 0 # unreachable
            continue
        if pc == 16: # ('goto', TLabel('L1'))
            self.pc = 19
            pc = self.pc = 0 # goto
            continue
            pc = 0
            continue
        if pc == 19: # ('-live-', %i1, %i0)
            self.pc = 22
            pass # live
            pc = 25
            continue
        if pc == 22: # ('goto', TLabel('L4'))
            self.pc = 25
            pc = self.pc = 25 # goto
            continue
            pc = 25
            continue
        if pc == 25: # ('int_return', %i1)
            self.pc = 27
            try:
                self.opimpl_int_return(self.registers_i[1])
            except ChangeFrame: return
            assert 0 # unreachable
        assert 0 # unreachable"""


@pytest.mark.xfail()
def test_skip_chained_jump_1():
    ssarepr = SSARepr("test", genextension=True)
    i0, i1 = Register('int', 0x0), Register('int', 0x1)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_add', i1, i0, '->', i1),
        ('goto', TLabel('L2'),),
        (Label('L3'),),
        ('-live-', i1, i0),
        ('goto', TLabel('L1'),),
        (Label('L2'),),
        ('goto', TLabel('L3'),),
        ('int_return', i1),
        ('---',)]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    while 1:
        if pc == 0: # ('int_sub', %i0, (1), '->', %i0)
            self.pc = 4
            self._result_argcode = 'i'
            self.registers_i[0] = self.opimpl_int_sub(self.registers_i[0], ConstInt(1))
            pc = 4
            continue
        if pc == 4: # ('int_add', %i1, %i0, '->', %i1)
            self.pc = 8
            self._result_argcode = 'i'
            self.registers_i[1] = self.opimpl_int_add(self.registers_i[1], self.registers_i[0])
            pc = 0
            continue
        if pc == 8: # ('goto', TLabel('L2'))
            self.pc = 11
            pc = self.pc = 0 # goto
            continue
            pc = 0
            continue
        if pc == 11: # ('-live-', %i1, %i0)
            self.pc = 14
            pass # live
            pc = 0
            continue
        if pc == 14: # ('goto', TLabel('L1'))
            self.pc = 17
            pc = self.pc = 0 # goto
            continue
            pc = 0
            continue
        if pc == 17: # ('goto', TLabel('L3'))
            self.pc = 20
            pc = self.pc = 0 # goto
            continue
            pc = 0
            continue
        if pc == 20: # ('int_return', %i1)
            self.pc = 22
            try:
                self.opimpl_int_return(self.registers_i[1])
            except ChangeFrame: return
            assert 0 # unreachable
        assert 0 # unreachable"""


def test_specialize_int_add():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn1 = (
        'int_add', i1, i0, '->', i1
    )
    insn2 = (
        'int_add', i1, i0, '->', i2
    )
    work_list = WorkList({5: insn1, 6: ('int_return', i1), 7: insn2, 8: ('int_return', i1)}, pc_to_nextpc={5: 6, 7: 8})
    insn_specializer = work_list.specialize_pc({i0, i1}, 5) # i0 and i1 are unboxed in local variables already
    assert work_list.specialize_pc({i0, i1}, 5) is insn_specializer
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert s == """\
i1 = i1 + i0
pc = 109
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {i0, i1}

    insn_specializer = work_list.specialize_insn(insn2, {i0, i1}, 7) # i0 and i1 are unboxed in local variables already
    s = insn_specializer.make_code()
    assert s == """\
i2 = i1 + i0
pc = 111
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {i0, i1, i2}

    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
ri1 = self.registers_i[1]
ri0 = self.registers_i[0]
if isinstance(ri1, ConstInt) and isinstance(ri0, ConstInt):
    i1 = ri1.getint()
    i0 = ri0.getint()
    pc = 108
    continue
else:
    self.registers_i[1] = self.opimpl_int_add(ri1, ri0)
pc = 6
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == set()

    insn_specializer = work_list.specialize_insn(insn1, {i2}, 5) # i0 and i1 are unboxed in local variables already
    s = insn_specializer.make_code()
    assert s == """\
ri1 = self.registers_i[1]
ri0 = self.registers_i[0]
if isinstance(ri1, ConstInt) and isinstance(ri0, ConstInt):
    i1 = ri1.getint()
    i0 = ri0.getint()
    pc = 113
    continue
else:
    self.registers_i[1] = self.opimpl_int_add(ri1, ri0)
pc = 114
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {i2}

def test_int_add_const():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn1 = (
        'int_add', i0, Constant(1, lltype.Signed), '->', i1
    )
    work_list = WorkList({5: insn1, 7: ('int_return', i1)}, pc_to_nextpc={5:7})
    insn_specializer = work_list.specialize_insn(insn1, {i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 7
    s = insn_specializer.make_code()
    assert s == """i1 = i0 + 1
pc = 108
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {i0, i1}

    insn_specializer = work_list.specialize_insn(insn1, set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = %d
    continue
else:
    self.registers_i[1] = self.opimpl_int_add(ri0, ConstInt(1))
pc = 7
continue""" % (work_list.OFFSET + 7)
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == set()


def test_strgetitem():
    r0, i0, i1 = Register('ref', 0), Register('int', 0), Register('int', 1)
    insn1 = ('strgetitem', r0, i0, '->', i1)
    work_list = WorkList({5: insn1, 6: ('int_return', i1)}, pc_to_nextpc={5: 6})

    insn_specializer = work_list.specialize_insn(insn1, set(), 5) # i0 and i1 are unboxed in local variables already
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
ri0 = self.registers_i[0]
if isinstance(rr0, ConstPtr) and isinstance(ri0, ConstInt):
    r0 = rr0.getref_base()
    i0 = ri0.getint()
    pc = %d
    continue
else:
    self.registers_i[1] = self.opimpl_strgetitem(rr0, ri0)
pc = 6
continue""" % (work_list.OFFSET + 6)

    insn_specializer = work_list.specialize_insn(insn1, {i0, r0}, 5) # i0 and i1 are unboxed in local variables already
    assert work_list.specialize_insn(insn1, {i0, r0}, 5) is insn_specializer
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 6
    s = insn_specializer.make_code()
    assert s == """\
i1 = ord(lltype.cast_opaque_ptr(lltype.Ptr(rstr.STR), r0).chars[i0])
pc = 107
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {r0, i0, i1}

def test_goto_if_not_int_lt():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_int_lt', i0, i1, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i0)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
ri1 = self.registers_i[1]
if isinstance(ri0, ConstInt) and isinstance(ri1, ConstInt):
    i0 = ri0.getint()
    i1 = ri1.getint()
    pc = 117
    continue
condbox = self.opimpl_int_lt(ri0, ri1)
self.opimpl_goto_if_not(condbox, 17, 5)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
ri1 = self.registers_i[1]
if isinstance(ri0, ConstInt) and isinstance(ri1, ConstInt):
    i0 = ri0.getint()
    i1 = ri1.getint()
    pc = 119
    continue
condbox = self.opimpl_int_lt(ri0, ri1)
self.registers_i[2] = ConstInt(i2)
self.opimpl_goto_if_not(condbox, 17, 5)
pc = self.pc
if pc == 17:
    pc = 120
else:
    assert self.pc == 6
    pc = 121
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0, i1}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
cond = i0 < i1
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_int_is_true():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_int_is_true', i0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i0)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 117
    continue
self.opimpl_goto_if_not_int_is_true(ri0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 119
    continue
self.registers_i[2] = ConstInt(i2)
self.opimpl_goto_if_not_int_is_true(ri0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 120
else:
    assert self.pc == 6
    pc = 121
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    assert s == """\
cond = i0 != 0
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_int_is_zero():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_int_is_zero', i0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i0)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 117
    continue
self.opimpl_goto_if_not_int_is_zero(ri0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 119
    continue
self.registers_i[2] = ConstInt(i2)
self.opimpl_goto_if_not_int_is_zero(ri0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 120
else:
    assert self.pc == 6
    pc = 121
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    assert s == """\
cond = i0 == 0
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_ptr_nonzero():
    r0, i1, i2 = Register('ref', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_ptr_nonzero', r0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', r0, i1, '->', i2), 6: ('int_return', r0)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 117
    continue
self.opimpl_goto_if_not_ptr_nonzero(rr0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 119
    continue
self.registers_i[2] = ConstInt(i2)
self.opimpl_goto_if_not_ptr_nonzero(rr0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 120
else:
    assert self.pc == 6
    pc = 121
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({r0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    assert s == """\
cond = r0.nonnull()
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_ptr_zero():
    r0, i1, i2 = Register('ref', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_ptr_zero', r0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', r0, i1, '->', i2), 6: ('int_return', r0)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 117
    continue
self.opimpl_goto_if_not_ptr_zero(rr0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 119
    continue
self.registers_i[2] = ConstInt(i2)
self.opimpl_goto_if_not_ptr_zero(rr0, 17, 5)
pc = self.pc
if pc == 17:
    pc = 120
else:
    assert self.pc == 6
    pc = 121
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({r0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    assert s == """\
cond = not r0.nonnull()
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_int_between():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('int_between', i0, i1, i2, '->', i3)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i0)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={})
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
ri1 = self.registers_i[1]
ri2 = self.registers_i[2]
if isinstance(ri0, ConstInt) and isinstance(ri1, ConstInt) and isinstance(ri2, ConstInt):
    i0 = ri0.getint()
    i1 = ri1.getint()
    i2 = ri2.getint()
    pc = 117
    continue
self.opimpl_int_between(ri0, ri1, ri2)"""

    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
v0 = i2 - i0
if v0 == 1:
    return i0 == i1
else:
    v1 = i1 - i0
    return v1 < v0"""

def test_int_guard_value():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn = ('int_guard_value', i0)
    work_list = WorkList({5: insn, 6: ('int_return', 6)}, pc_to_nextpc={5: 6})

    # specialized case
    insn_specializer = work_list.specialize_insn(insn, {i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 6
    s = insn_specializer.make_code()
    assert s == """\
# guard_value, argument is already constant
pc = 107
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {i0}

    # unspecialized cases
    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = %d
    continue
self.opimpl_int_guard_value(ri0, 5)
ri0 = self.registers_i[0]
i0 = ri0.getint()
pc = 107
continue""" % (work_list.OFFSET + 6)
    next_constant_registers = insn_specializer.get_next_constant_registers()

    insn_specializer = work_list.specialize_insn(insn, {i1, i2}, 5)
    s = insn_specializer.make_code()
    # we need to sync the registers from the unboxed values to allow the guard to be created
    # TODO: only do this for registers that are alive at this point
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 109
    continue
self.registers_i[1] = ConstInt(i1)
self.registers_i[2] = ConstInt(i2)
self.opimpl_int_guard_value(ri0, 5)
ri0 = self.registers_i[0]
i0 = ri0.getint()
pc = 110
continue"""

def test_switch():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    switchdict = {-5: 9,  2: 14, 7: 19}
    descr = SwitchDictDescr()
    descr.attach(switchdict)
    insn = ('switch', i0, descr)
    dummy_insn = ('-live-')
    dummy_insn2 = ('int_add', i0, i1, '->', i2)
    dummy_insn3 = ('int_sub', i0, i1, '->', i2)
    insns = {5: insn, 9: dummy_insn, 14: dummy_insn2, 19: dummy_insn3, 21: ('int_return', i0)}
    max_used_pc = max(insns)
    work_list = WorkList(insns, pc_to_nextpc={5: 21})

    # specialized case
    insn_specializer = work_list.specialize_pc({i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max_used_pc
    s = insn_specializer.make_code()
    assert s == """\
if i0 == -5:
    pc = %d
    continue
elif i0 == 2:
    pc = %d
    continue
elif i0 == 7:
    pc = %d
    continue
pc = %s
continue""" % (
            max_used_pc + work_list.OFFSET + 1,
            max_used_pc + work_list.OFFSET + 2,
            max_used_pc + work_list.OFFSET + 3,
            max_used_pc + work_list.OFFSET + 4,
        )

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 121
    continue
self.opimpl_switch(ri0, glob0, 5)
pc = self.pc
if pc == 9: pc = 9
elif pc == 14: pc = 14
elif pc == 19: pc = 19
elif pc == 21: pc = 21
else: assert 0
continue"""

def test_goto():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto', L1)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 19: ('int_return', i2)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 17, 17: 19})

    # unspecialized case
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
pc = 17
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0, i1}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert s == """\
pc = 120
continue"""

def test_guard_class():
    i0, r0, i2 = Register('int', 0), Register('ref', 0), Register('int', 2)
    insn = ('guard_class', r0, '->', i0)
    work_list = WorkList({5: insn, 6: ('int_return', 6)}, pc_to_nextpc={5: 6})

    # specialized case
    insn_specializer = work_list.specialize_insn(insn, {r0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 6
    s = insn_specializer.make_code()
    assert s == """\
# guard_class, argument is already constant
i0 = support.ptr2int(lltype.cast_opaque_ptr(OBJECTPTR, r0).typeptr)
pc = 107
continue"""

    # unspecialized cases
    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
rr0 = self.registers_r[0]
if self.metainterp.heapcache.is_class_known(rr0):
    i0 = self.cls_of_box(rr0).getint()
    pc = 108
    continue
i0 = self.opimpl_guard_class(rr0, 5).getint()
pc = 108
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()

def test_int_copy():
    i0, r0, i2 = Register('int', 0), Register('ref', 0), Register('int', 2)
    insn = ('int_copy', i2, '->', i0)
    work_list = WorkList({5: insn, 6: ('int_return', i0)}, pc_to_nextpc={5: 6})

    # specialized case
    insn_specializer = work_list.specialize_insn(insn, {i2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
i0 = i2
pc = 107
continue"""

    # unspecialized cases
    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
ri2 = self.registers_i[2]
self.registers_i[0] = ri2
pc = 6
continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()


def test_live():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn0 = ('int_add', i0, i1, '->', i2)
    insn1 = ('-live-', i2)
    insn2 = ('int_return', i2)
    work_list = WorkList({0: insn0, 1: insn1, 2: insn2}, pc_to_nextpc={0: 1, 1: 2})

    insn_specializer = work_list.specialize_pc({i0, i1, i2}, 1)
    assert insn_specializer.orig_pc == 1
    assert insn_specializer.constant_registers == frozenset([i2])
    s = insn_specializer.make_code()
    assert s == """\
pc = 103
continue"""
