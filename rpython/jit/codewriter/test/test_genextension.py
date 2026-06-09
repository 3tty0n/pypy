import re

from rpython.flowspace.model import Constant
from rpython.jit.codewriter.jitcode import JitCode, SwitchDictDescr
from rpython.jit.codewriter.flatten import (
    SSARepr, Label, TLabel, Register, ListOfKind)
from rpython.jit.codewriter.assembler import Assembler, AssemblerError
from rpython.jit.codewriter.effectinfo import EffectInfo
from rpython.rtyper.lltypesystem import lltype, llmemory
from rpython.jit.metainterp.history import AbstractDescr
from rpython.jit.codewriter.genextension import WorkList, GenExtension
from rpython.config import translationoption
from rpython.config.translationoption import get_combined_translation_config

import pytest


@pytest.fixture
def enable_genextension(request):
    config = get_combined_translation_config(translating=True)
    config.translation.genextension = True
    old_config = translationoption._GLOBAL_TRANSLATIONCONFIG
    translationoption._GLOBAL_TRANSLATIONCONFIG = config
    def cleanup():
        translationoption._GLOBAL_TRANSLATIONCONFIG = old_config
    request.addfinalizer(cleanup)
    return config


def test_thread_blocks_inlines_chains(enable_genextension):
    import rpython.jit.codewriter.genextension as genext

    def build():
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
        return jitcode._genext_source

    old = genext.THREAD_BLOCKS
    try:
        genext.THREAD_BLOCKS = False
        base = build()
        genext.THREAD_BLOCKS = True
        threaded = build()
    finally:
        genext.THREAD_BLOCKS = old

    n_entries_base = len(re.findall(r'if pc == \d+:', base))
    n_entries_thr = len(re.findall(r'if pc == \d+:', threaded))
    n_selfpc_base = len(re.findall(r'self\.pc = \d+', base))
    n_selfpc_thr = len(re.findall(r'self\.pc = \d+', threaded))

    assert n_entries_thr < n_entries_base
    assert n_selfpc_thr == n_selfpc_base
    assert "def jit_shortcut(self): # test" in threaded


def test_assemble_loop(enable_genextension):
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
    source = jitcode._genext_source
    assert "def jit_shortcut(self): # test" in source
    assert "if pc <" in source
    assert "execute_and_record(rop.INT_GT" in source
    assert "record2_int(rop.INT_ADD" in source
    assert "record2_int(rop.INT_SUB" in source
    assert "self.opimpl_goto_if_not(condbox, 16, 0, replace=False)" in source
    assert "pc == 0: # ('goto_if_not_int_gt'" in source
    first_branch = source.split("pc == 0: # ('goto_if_not_int_gt'", 1)[1]
    first_branch = first_branch.split("pc == 5: # ('int_add'", 1)[0]
    assert "if isinstance(ri22, ConstInt)" not in first_branch
    assert "condbox = self.metainterp.execute_and_record(rop.INT_GT" in first_branch
    return
    assert jitcode._genext_source == """\
def jit_shortcut(self): # test
    pc = self.pc
    i22 = 0xcafedead
    i23 = 0xcafedead
    if pc == 0: pc = 0
    else: assert 0, 'unreachable'
    while 1:
        if pc < 117:
            if pc < 13:
                if pc < 5:
                    if pc == 0: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([])
                        self.pc = 5
                        ri22 = self.registers_i[22]
                        if isinstance(ri22, ConstInt):
                            i22 = ri22.getint()
                            pc = 116
                            continue
                        _b0 = self.registers_i[22]
                        _b1 = const_int(4)
                        if _b0 is _b1:
                            pc = 16
                            continue
                        _v0 = ri22.getint()
                        _v1 = 4
                        _cond = int(_v0 > _v1)
                        condbox = self.metainterp.history.record2_int(rop.INT_GT, _b0, _b1, _cond)
                        self.opimpl_goto_if_not(condbox, 16, 0, replace=False)
                        pc = self.pc
                        if pc == 16:
                            pc = 16
                        else:
                            assert self.pc == 5
                            pc = 5
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 9:
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
                                _v0 = ri23.getint()
                                _v1 = ri22.getint()
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri23, ri22, _res)
                                self.registers_i[23] = _op
                                i23 = _res
                                pc = 9
                                continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 9: # ('int_sub', %i22, (1), '->', %i22) frozenset([])
                            self.pc = 13
                            ri22 = self.registers_i[22]
                            if isinstance(ri22, ConstInt):
                                i22 = ri22.getint()
                                pc = 116
                                continue
                            else:
                                _v0 = ri22.getint()
                                _v1 = 1
                                _res = _v0 - _v1
                                _op = self.metainterp.history.record2_int(rop.INT_SUB, ri22, const_int(1), _res)
                                self.registers_i[22] = _op
                                i22 = _res
                                pc = 0
                                continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 16:
                    if pc == 13: # ('goto', TLabel('L1')) frozenset([])
                        self.pc = 16
                        pc = 0
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 116:
                        if pc == 16: # ('int_return', %i23) frozenset([])
                            self.pc = 18
                            ri23 = self.registers_i[23]
                            try:
                                self.opimpl_int_return(ri23)
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 116: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([%i22])
                            self.pc = 5
                            cond = i22 > 4
                            if not cond:
                                pc = 118
                                continue
                            pc = 119
                            continue
                        else:
                            assert 0 # unreachable
        else:
            if pc < 120:
                if pc < 118:
                    if pc == 117: # ('int_sub', %i22, (1), '->', %i22) frozenset([%i23, %i22])
                        self.pc = 13
                        i22 = i22 - 1
                        pc = 120
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 119:
                        if pc == 118: # ('int_return', %i23) frozenset([%i22])
                            self.pc = 18
                            ri23 = self.registers_i[23]
                            try:
                                self.opimpl_int_return(ri23)
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 119: # ('int_add', %i23, %i22, '->', %i23) frozenset([%i22])
                            self.pc = 9
                            ri23 = self.registers_i[23]
                            if isinstance(ri23, ConstInt):
                                i23 = ri23.getint()
                                pc = 117
                                continue
                            else:
                                _v0 = ri23.getint()
                                _v1 = i22
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri23, const_int(i22), _res)
                                self.registers_i[23] = _op
                                i23 = _res
                                pc = 121
                                continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 122:
                    if pc < 121:
                        if pc == 120: # ('goto_if_not_int_gt', %i22, (4), TLabel('L2')) frozenset([%i23, %i22])
                            self.pc = 5
                            cond = i22 > 4
                            if not cond:
                                pc = 122
                                continue
                            pc = 123
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 121: # ('int_sub', %i22, (1), '->', %i22) frozenset([%i22])
                            self.pc = 13
                            i22 = i22 - 1
                            pc = 116
                            continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 123:
                        if pc == 122: # ('int_return', %i23) frozenset([%i23, %i22])
                            self.pc = 18
                            try:
                                self.opimpl_int_return(const_int(i23))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 123: # ('int_add', %i23, %i22, '->', %i23) frozenset([%i23, %i22])
                            self.pc = 9
                            i23 = i23 + i22
                            pc = 117
                            continue
                        else:
                            assert 0 # unreachable"""

def test_integration_switch(enable_genextension):
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
    elif pc == 12: pc = 12
    elif pc == 17: pc = 17
    elif pc == 22: pc = 22
    else: assert 0, 'unreachable'
    while 1:
        if pc < 14:
            if pc < 7:
                if pc < 3:
                    if pc == 0: # ('-live-', %i22) frozenset([])
                        self.pc = 3
                        pc = 3
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc == 3: # ('switch', %i22, <SwitchDictDescr {-5: 9, 2: 14, 7: 19}>) frozenset([])
                        self.pc = 7
                        ri22 = self.registers_i[22]
                        if isinstance(ri22, ConstInt):
                            i22 = ri22.getint()
                            pc = 122
                            continue
                        self.opimpl_switch(ri22, glob0, 3)
                        pc = self.pc
                        if pc == 9: pc = 9
                        elif pc == 14: pc = 14
                        elif pc == 19: pc = 19
                        elif pc == 7: pc = 7
                        else: assert 0
                        continue
                    else:
                        assert 0 # unreachable
            else:
                if pc < 9:
                    if pc == 7: # ('int_return', (42)) frozenset([])
                        self.pc = 9
                        try:
                            self.opimpl_int_return(const_int(42))
                        except ChangeFrame: return
                        assert 0, 'unreachable'
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 12:
                        if pc == 9: # ('-live-',) frozenset([])
                            self.pc = 12
                            pc = 12
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 12: # ('int_return', (12)) frozenset([])
                            self.pc = 14
                            try:
                                self.opimpl_int_return(const_int(12))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
        else:
            if pc < 22:
                if pc < 17:
                    if pc == 14: # ('-live-',) frozenset([])
                        self.pc = 17
                        pc = 17
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 19:
                        if pc == 17: # ('int_return', (51)) frozenset([])
                            self.pc = 19
                            try:
                                self.opimpl_int_return(const_int(51))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 19: # ('-live-',) frozenset([])
                            self.pc = 22
                            pc = 22
                            continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 122:
                    if pc == 22: # ('int_return', (1212)) frozenset([])
                        self.pc = 24
                        try:
                            self.opimpl_int_return(const_int(1212))
                        except ChangeFrame: return
                        assert 0, 'unreachable'
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 123:
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
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 123: # ('int_return', (42)) frozenset([%i22])
                            self.pc = 9
                            try:
                                self.opimpl_int_return(const_int(42))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable"""


@pytest.mark.parametrize("opname, ropname, pyop", [
    ("int_add_jump_if_ovf", "INT_ADD_OVF", "+"),
    ("int_sub_jump_if_ovf", "INT_SUB_OVF", "-"),
    ("int_mul_jump_if_ovf", "INT_MUL_OVF", "*"),
])
def test_int_binop_jump_if_ovf_fast_path(enable_genextension, opname, ropname, pyop):
    ssarepr = SSARepr("ovf_test", genextension=True)
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    ssarepr.insns = [
        (Label('L0'),),
        (opname, TLabel('L1'), i0, i1, '->', i2),
        ('int_return', i2),
        ('---',),
        (Label('L1'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ('---',),
    ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})
    source = jitcode._genext_source
    assert "self.metainterp.ovf_flag = False" in source
    assert "_res = ovfcheck(_v0 %s _v1)" % pyop in source
    assert "_op = self.metainterp.history.record2_int(rop.%s" % ropname in source
    assert "self.handle_possible_overflow_error(" in source
    fast_path = source.split("pc == 0: # ('%s'" % opname, 1)[1]
    fast_path = fast_path.split("pc == 1: # ('int_return'", 1)[0]
    assert "if isinstance(ri0, ConstInt)" not in fast_path
    assert "_v0 = self.registers_i[0].getint()" in fast_path
    assert "_v1 = self.registers_i[1].getint()" in fast_path

@pytest.mark.xfail()
def test_skip_jump_to_live(enable_genextension):
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
    i0 = 0xcafedead
    i1 = 0xcafedead
    if pc == 0: pc = 0
    elif pc == 11: pc = 11
    else: assert 0, 'unreachable'
    while 1:
        if pc < 120:
            if pc < 11:
                if pc < 4:
                    if pc == 0: # ('int_sub', %i0, (1), '->', %i0) frozenset([])
                        self.pc = 4
                        ri0 = self.registers_i[0]
                        if isinstance(ri0, ConstInt):
                            i0 = ri0.getint()
                            pc = 119
                            continue
                        else:
                            _v0 = ri0.getint()
                            _v1 = 1
                            _res = _v0 - _v1
                            _op = self.metainterp.history.record2_int(rop.INT_SUB, ri0, const_int(1), _res)
                            self.registers_i[0] = _op
                            i0 = _res
                            pc = 4
                            continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 8:
                        if pc == 4: # ('int_add', %i1, %i0, '->', %i1) frozenset([])
                            self.pc = 8
                            ri1 = self.registers_i[1]
                            ri0 = self.registers_i[0]
                            if isinstance(ri1, ConstInt) and isinstance(ri0, ConstInt):
                                i1 = ri1.getint()
                                i0 = ri0.getint()
                                pc = 120
                                continue
                            else:
                                _v0 = ri1.getint()
                                _v1 = ri0.getint()
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, ri0, _res)
                                self.registers_i[1] = _op
                                i1 = _res
                                pc = 11
                                continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 8: # ('-live-', %i1, %i0) frozenset([])
                            self.pc = 11
                            pc = 11
                            continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 19:
                    if pc < 16:
                        if pc == 11: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([])
                            self.pc = 16
                            ri0 = self.registers_i[0]
                            if isinstance(ri0, ConstInt):
                                i0 = ri0.getint()
                                pc = 121
                                continue
                            _v0 = ri0.getint()
                            _v1 = 0
                            _cond = int(_v0 > _v1)
                            condbox = self.metainterp.history.record2_int(rop.INT_GT, self.registers_i[0], const_int(0), _cond)
                            self.opimpl_goto_if_not(condbox, 19, 11, replace=False)
                            pc = self.pc
                            if pc == 19:
                                pc = 19
                            else:
                                assert self.pc == 16
                                pc = 16
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 16: # ('goto', TLabel('L1')) frozenset([])
                            self.pc = 19
                            pc = 0
                            continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 119:
                        if pc == 19: # ('int_return', %i1) frozenset([])
                            self.pc = 21
                            ri1 = self.registers_i[1]
                            try:
                                self.opimpl_int_return(ri1)
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 119: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i0])
                            self.pc = 8
                            ri1 = self.registers_i[1]
                            if isinstance(ri1, ConstInt):
                                i1 = ri1.getint()
                                pc = 120
                                continue
                            else:
                                _v0 = ri1.getint()
                                _v1 = i0
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, const_int(i0), _res)
                                self.registers_i[1] = _op
                                i1 = _res
                                pc = 121
                                continue
                        else:
                            assert 0 # unreachable
        else:
            if pc < 123:
                if pc < 121:
                    if pc == 120: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([%i1, %i0])
                        self.pc = 16
                        cond = i0 > 0
                        if not cond:
                            pc = 122
                            continue
                        pc = 123
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 122:
                        if pc == 121: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([%i0])
                            self.pc = 16
                            cond = i0 > 0
                            if not cond:
                                pc = 124
                                continue
                            pc = 125
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 122: # ('int_return', %i1) frozenset([%i1, %i0])
                            self.pc = 21
                            try:
                                self.opimpl_int_return(const_int(i1))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
            else:
                if pc < 125:
                    if pc < 124:
                        if pc == 123: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i1, %i0])
                            self.pc = 4
                            i0 = i0 - 1
                            pc = 126
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 124: # ('int_return', %i1) frozenset([%i0])
                            self.pc = 21
                            ri1 = self.registers_i[1]
                            try:
                                self.opimpl_int_return(ri1)
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 126:
                        if pc == 125: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i0])
                            self.pc = 4
                            i0 = i0 - 1
                            pc = 119
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 126: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i1, %i0])
                            self.pc = 8
                            i1 = i1 + i0
                            pc = 120
                            continue
                        else:
                            assert 0 # unreachable"""


@pytest.mark.xfail()
def test_skip_conditional_jump(enable_genextension):
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
    i0 = 0xcafedead
    i1 = 0xcafedead
    if pc == 0: pc = 0
    elif pc == 11: pc = 11
    elif pc == 22: pc = 22
    else: assert 0, 'unreachable'
    while 1:
        if pc < 126:
            if pc < 16:
                if pc < 8:
                    if pc < 4:
                        if pc == 0: # ('int_sub', %i0, (1), '->', %i0) frozenset([])
                            self.pc = 4
                            ri0 = self.registers_i[0]
                            if isinstance(ri0, ConstInt):
                                i0 = ri0.getint()
                                pc = 125
                                continue
                            else:
                                _v0 = ri0.getint()
                                _v1 = 1
                                _res = _v0 - _v1
                                _op = self.metainterp.history.record2_int(rop.INT_SUB, ri0, const_int(1), _res)
                                self.registers_i[0] = _op
                                i0 = _res
                                pc = 4
                                continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 4: # ('int_add', %i1, %i0, '->', %i1) frozenset([])
                            self.pc = 8
                            ri1 = self.registers_i[1]
                            ri0 = self.registers_i[0]
                            if isinstance(ri1, ConstInt) and isinstance(ri0, ConstInt):
                                i1 = ri1.getint()
                                i0 = ri0.getint()
                                pc = 126
                                continue
                            else:
                                _v0 = ri1.getint()
                                _v1 = ri0.getint()
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, ri0, _res)
                                self.registers_i[1] = _op
                                i1 = _res
                                pc = 11
                                continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 11:
                        if pc == 8: # ('-live-', %i1, %i0) frozenset([])
                            self.pc = 11
                            pc = 11
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 11: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([])
                            self.pc = 16
                            ri0 = self.registers_i[0]
                            if isinstance(ri0, ConstInt):
                                i0 = ri0.getint()
                                pc = 127
                                continue
                            _v0 = ri0.getint()
                            _v1 = 0
                            _cond = int(_v0 > _v1)
                            condbox = self.metainterp.history.record2_int(rop.INT_GT, self.registers_i[0], const_int(0), _cond)
                            self.opimpl_goto_if_not(condbox, 19, 11, replace=False)
                            pc = self.pc
                            if pc == 19:
                                pc = 19
                            else:
                                assert self.pc == 16
                                pc = 16
                            continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 22:
                    if pc < 19:
                        if pc == 16: # ('goto', TLabel('L1')) frozenset([])
                            self.pc = 19
                            pc = 0
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 19: # ('-live-', %i1, %i0) frozenset([])
                            self.pc = 22
                            pc = 25
                            continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 25:
                        if pc == 22: # ('goto', TLabel('L4')) frozenset([])
                            self.pc = 25
                            pc = 25
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc < 125:
                            if pc == 25: # ('int_return', %i1) frozenset([])
                                self.pc = 27
                                ri1 = self.registers_i[1]
                                try:
                                    self.opimpl_int_return(ri1)
                                except ChangeFrame: return
                                assert 0, 'unreachable'
                            else:
                                assert 0 # unreachable
                        else:
                            if pc == 125: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i0])
                                self.pc = 8
                                ri1 = self.registers_i[1]
                                if isinstance(ri1, ConstInt):
                                    i1 = ri1.getint()
                                    pc = 126
                                    continue
                                else:
                                    _v0 = ri1.getint()
                                    _v1 = i0
                                    _res = _v0 + _v1
                                    _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, const_int(i0), _res)
                                    self.registers_i[1] = _op
                                    i1 = _res
                                    pc = 127
                                    continue
                            else:
                                assert 0 # unreachable
        else:
            if pc < 130:
                if pc < 128:
                    if pc < 127:
                        if pc == 126: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([%i1, %i0])
                            self.pc = 16
                            cond = i0 > 0
                            if not cond:
                                pc = 128
                                continue
                            pc = 129
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 127: # ('goto_if_not_int_gt', %i0, (0), TLabel('L2')) frozenset([%i0])
                            self.pc = 16
                            cond = i0 > 0
                            if not cond:
                                pc = 130
                                continue
                            pc = 131
                            continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 129:
                        if pc == 128: # ('-live-', %i1, %i0) frozenset([%i1, %i0])
                            self.pc = 22
                            pc = 132
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 129: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i1, %i0])
                            self.pc = 4
                            i0 = i0 - 1
                            pc = 133
                            continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 132:
                    if pc < 131:
                        if pc == 130: # ('-live-', %i1, %i0) frozenset([%i0])
                            self.pc = 22
                            pc = 134
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 131: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i0])
                            self.pc = 4
                            i0 = i0 - 1
                            pc = 125
                            continue
                        else:
                            assert 0 # unreachable
                else:
                    if pc < 133:
                        if pc == 132: # ('int_return', %i1) frozenset([%i1, %i0])
                            self.pc = 27
                            try:
                                self.opimpl_int_return(const_int(i1))
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc < 134:
                            if pc == 133: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i1, %i0])
                                self.pc = 8
                                i1 = i1 + i0
                                pc = 126
                                continue
                            else:
                                assert 0 # unreachable
                        else:
                            if pc == 134: # ('int_return', %i1) frozenset([%i0])
                                self.pc = 27
                                ri1 = self.registers_i[1]
                                try:
                                    self.opimpl_int_return(ri1)
                                except ChangeFrame: return
                                assert 0, 'unreachable'
                            else:
                                assert 0 # unreachable"""


@pytest.mark.xfail()
def test_skip_chained_jump_1(enable_genextension):
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
    i0 = 0xcafedead
    i1 = 0xcafedead
    if pc == 0: pc = 0
    elif pc == 14: pc = 14
    else: assert 0, 'unreachable'
    while 1:
        if pc < 17:
            if pc < 8:
                if pc < 4:
                    if pc == 0: # ('int_sub', %i0, (1), '->', %i0) frozenset([])
                        self.pc = 4
                        ri0 = self.registers_i[0]
                        if isinstance(ri0, ConstInt):
                            i0 = ri0.getint()
                            pc = 120
                            continue
                        else:
                            _v0 = ri0.getint()
                            _v1 = 1
                            _res = _v0 - _v1
                            _op = self.metainterp.history.record2_int(rop.INT_SUB, ri0, const_int(1), _res)
                            self.registers_i[0] = _op
                            i0 = _res
                            pc = 4
                            continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc == 4: # ('int_add', %i1, %i0, '->', %i1) frozenset([])
                        self.pc = 8
                        ri1 = self.registers_i[1]
                        ri0 = self.registers_i[0]
                        if isinstance(ri1, ConstInt) and isinstance(ri0, ConstInt):
                            i1 = ri1.getint()
                            i0 = ri0.getint()
                            pc = 121
                            continue
                        else:
                            _v0 = ri1.getint()
                            _v1 = ri0.getint()
                            _res = _v0 + _v1
                            _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, ri0, _res)
                            self.registers_i[1] = _op
                            i1 = _res
                            pc = 0
                            continue
                    else:
                        assert 0 # unreachable
            else:
                if pc < 11:
                    if pc == 8: # ('goto', TLabel('L2')) frozenset([])
                        self.pc = 11
                        pc = 0
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 14:
                        if pc == 11: # ('-live-', %i1, %i0) frozenset([])
                            self.pc = 14
                            pc = 0
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 14: # ('goto', TLabel('L1')) frozenset([])
                            self.pc = 17
                            pc = 0
                            continue
                        else:
                            assert 0 # unreachable
        else:
            if pc < 121:
                if pc < 20:
                    if pc == 17: # ('goto', TLabel('L3')) frozenset([])
                        self.pc = 20
                        pc = 0
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 120:
                        if pc == 20: # ('int_return', %i1) frozenset([])
                            self.pc = 22
                            ri1 = self.registers_i[1]
                            try:
                                self.opimpl_int_return(ri1)
                            except ChangeFrame: return
                            assert 0, 'unreachable'
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 120: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i0])
                            self.pc = 8
                            ri1 = self.registers_i[1]
                            if isinstance(ri1, ConstInt):
                                i1 = ri1.getint()
                                pc = 121
                                continue
                            else:
                                _v0 = ri1.getint()
                                _v1 = i0
                                _res = _v0 + _v1
                                _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, const_int(i0), _res)
                                self.registers_i[1] = _op
                                i1 = _res
                                pc = 122
                                continue
                        else:
                            assert 0 # unreachable
            else:
                if pc < 122:
                    if pc == 121: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i1, %i0])
                        self.pc = 4
                        i0 = i0 - 1
                        pc = 123
                        continue
                    else:
                        assert 0 # unreachable
                else:
                    if pc < 123:
                        if pc == 122: # ('int_sub', %i0, (1), '->', %i0) frozenset([%i0])
                            self.pc = 4
                            i0 = i0 - 1
                            pc = 120
                            continue
                        else:
                            assert 0 # unreachable
                    else:
                        if pc == 123: # ('int_add', %i1, %i0, '->', %i1) frozenset([%i1, %i0])
                            self.pc = 8
                            i1 = i1 + i0
                            pc = 121
                            continue
                        else:
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
    pc = 109
    continue
else:
    _v0 = ri1.getint()
    _v1 = ri0.getint()
    _res = _v0 + _v1
    _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, ri0, _res)
    self.registers_i[1] = _op
    i1 = _res
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
    _v0 = ri1.getint()
    _v1 = ri0.getint()
    _res = _v0 + _v1
    _op = self.metainterp.history.record2_int(rop.INT_ADD, ri1, ri0, _res)
    self.registers_i[1] = _op
    i1 = _res
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
    pc = 109
    continue
else:
    _v0 = ri0.getint()
    _v1 = 1
    _res = _v0 + _v1
    _op = self.metainterp.history.record2_int(rop.INT_ADD, ri0, const_int(1), _res)
    self.registers_i[1] = _op
    i1 = _res
    pc = 7
    continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == set()

def test_int_rshift_const():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn1 = (
        'int_rshift', i0, Constant(1, lltype.Signed), '->', i1
    )
    work_list = WorkList({5: insn1, 7: ('int_return', i1)}, pc_to_nextpc={5:7})
    insn_specializer = work_list.specialize_insn(insn1, {i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 7
    s = insn_specializer.make_code()
    assert s == """i1 = i0 >> 1
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
    pc = 109
    continue
else:
    _v0 = ri0.getint()
    _v1 = 1
    _res = _v0 >> _v1
    _op = self.metainterp.history.record2_int(rop.INT_RSHIFT, ri0, const_int(1), _res)
    self.registers_i[1] = _op
    i1 = _res
    pc = 7
    continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == set()

def test_int_invert():
    i0, i1 = Register('int', 0), Register('int', 1)
    insn1 = (
        'int_invert', i0, '->', i1
    )
    work_list = WorkList({5: insn1, 7: ('int_return', i1)}, pc_to_nextpc={5:7})
    insn_specializer = work_list.specialize_insn(insn1, {i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 7
    s = insn_specializer.make_code()
    assert s == """i1 = ~i0
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
    pc = 109
    continue
else:
    _v0 = ri0.getint()
    _res = ~_v0
    _op = self.metainterp.history.record1_int(rop.INT_INVERT, ri0, _res)
    self.registers_i[1] = _op
    i1 = _res
    pc = 7
    continue"""
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
    # fast-path: compute the condition through execute_and_record so constant
    # boxes fold away without recording a condition op or guard.
    assert s == """\
_b0 = self.registers_i[0]
_b1 = self.registers_i[1]
if _b0 is _b1:
    pc = 17
    continue
condbox = self.metainterp.execute_and_record(rop.INT_LT, None, _b0, _b1)
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
pc = self.pc
if pc == 17:
    pc = 17
else:
    assert self.pc == 6
    pc = 6
continue"""

    # unspecialized case with constant register
    insn_specializer = work_list.specialize_pc({i2}, 5)
    s = insn_specializer.make_code()
    # fast-path with deferred register sync before guard
    assert s == """\
_b0 = self.registers_i[0]
_b1 = self.registers_i[1]
if _b0 is _b1:
    pc = 118
    continue
condbox = self.metainterp.execute_and_record(rop.INT_LT, None, _b0, _b1)
glob0(self, i2) # jit_sync_regs_i2
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
pc = self.pc
if pc == 17:
    pc = 118
else:
    assert self.pc == 6
    pc = 119
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0, i1}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 120
    s = insn_specializer.make_code()
    assert s == """\
cond = i0 < i1
if not cond:
    pc = 121
    continue
pc = 122
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
_b0 = self.registers_i[0]
condbox = self.metainterp.execute_and_record(rop.INT_IS_TRUE, None, _b0)
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
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
_b0 = self.registers_i[0]
condbox = self.metainterp.execute_and_record(rop.INT_IS_TRUE, None, _b0)
glob0(self, i2) # jit_sync_regs_i2
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
pc = self.pc
if pc == 17:
    pc = 118
else:
    assert self.pc == 6
    pc = 119
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == 120
    assert s == """\
cond = i0 != 0
if not cond:
    pc = 121
    continue
pc = 122
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
_b0 = self.registers_i[0]
condbox = self.metainterp.execute_and_record(rop.INT_IS_ZERO, None, _b0)
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
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
_b0 = self.registers_i[0]
condbox = self.metainterp.execute_and_record(rop.INT_IS_ZERO, None, _b0)
glob0(self, i2) # jit_sync_regs_i2
self.opimpl_goto_if_not(condbox, 17, 5, replace=False)
pc = self.pc
if pc == 17:
    pc = 118
else:
    assert self.pc == 6
    pc = 119
continue"""

    # specialized case
    insn_specializer = work_list.specialize_pc({i0}, 5)
    newpc = insn_specializer.get_pc()
    s = insn_specializer.make_code()
    assert newpc == 120
    assert s == """\
cond = i0 == 0
if not cond:
    pc = 121
    continue
pc = 122
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
glob0(self, i2) # jit_sync_regs_i2
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
cond = r0
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_ptr_iszero():
    r0, i1, i2 = Register('ref', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_ptr_iszero', r0, L1)
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
self.opimpl_goto_if_not_ptr_iszero(rr0, 17, 5)
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
glob0(self, i2) # jit_sync_regs_i2
self.opimpl_goto_if_not_ptr_iszero(rr0, 17, 5)
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
cond = not r0
if not cond:
    pc = 122
    continue
pc = 123
continue"""


def test_goto_if_not_ptr_ne_const_path_falls_through_on_true():
    r0, r1, i2 = Register('ref', 0), Register('ref', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_ptr_ne', r0, r1, L1)
    pc_to_insn = {5: insn, 17: ('int_return', i2), 6: ('int_return', i2)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17},
                         pc_to_nextpc={5: 6})

    s = work_list.specialize_pc(set(), 5).make_code()
    assert "if isinstance(_b0, Const) and isinstance(_b1, Const):" in s
    assert """\
    if _cond:
        pc = 6
    else:
        pc = 17""" in s


def test_int_between():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('int_between', i0, i1, i2, '->', i3)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i3)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
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
self.registers_i[3] = self.opimpl_int_between(ri0, ri1, ri2)
pc = 6
continue"""

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
i3 = i0 <= i1 < i2
pc = 118
continue"""

def test_int_xor():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('int_xor', i0, i1, '->', i2)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i3)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
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
else:
    _v0 = ri0.getint()
    _v1 = ri1.getint()
    _res = _v0 ^ _v1
    _op = self.metainterp.history.record2_int(rop.INT_XOR, ri0, ri1, _res)
    self.registers_i[2] = _op
    i2 = _res
    pc = 6
    continue"""

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
i2 = i0 ^ i1
pc = 118
continue"""

def test_int_mod():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('int_mod', i0, i1, '->', i2)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i3)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
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
else:
    self.registers_i[2] = self.opimpl_int_mod(ri0, ri1)
pc = 6
continue"""

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
i2 = i0 % i1
pc = 118
continue"""

def test_int_floordiv():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('int_floordiv', i0, i1, '->', i2)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i3)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
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
else:
    self.registers_i[2] = self.opimpl_int_floordiv(ri0, ri1)
pc = 6
continue"""

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + max(pc_to_insn)
    s = insn_specializer.make_code()
    assert s == """\
i2 = i0 // i1
pc = 118
continue"""

def test_uint_lt():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('uint_lt', i0, i1, '->', i2)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 6: ('int_return', i3)}

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert '_res = int(r_uint(_v0) < r_uint(_v1))' in s
    assert 'rop.UINT_LT' in s

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    s = insn_specializer.make_code()
    assert s == """\
i2 = int(r_uint(i0) < r_uint(i1))
pc = 118
continue"""


def test_uint_mul_high_masks_to_signed_int():
    i0, i1, i2, i3 = Register('int', 0), Register('int', 1), Register('int', 2), Register('int', 3)
    insn = ('uint_mul_high', i0, i1, '->', i2)
    pc_to_insn = {5: insn, 6: ('int_return', i3)}

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert '_res = intmask(uint_mul_high(_v0, _v1))' in s
    assert 'record2_int(rop.UINT_MUL_HIGH' in s

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({i0, i1, i2, i3}, 5)
    s = insn_specializer.make_code()
    assert s == """\
i2 = intmask(uint_mul_high(i0, i1))
pc = 107
continue"""


def test_goto_if_not_float_lt():
    f0, f1, i2 = Register('float', 0), Register('float', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_float_lt', f0, f1, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i2, i2, '->', i2), 6: ('int_return', i2)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert 'rop.FLOAT_LT' in s
    assert 'execute_and_record' in s
    assert 'self.opimpl_goto_if_not(condbox' in s


def test_goto_if_not_float_comparison_keeps_same_box_recording():
    f0, i2 = Register('float', 0), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not_float_eq', f0, f0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i2, i2, '->', i2), 6: ('int_return', i2)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 17}, pc_to_nextpc={5: 6})

    insn_specializer = work_list.specialize_pc(set(), 5)
    s = insn_specializer.make_code()
    assert "if _b0 is _b1:" not in s
    assert "execute_and_record(rop.FLOAT_EQ" in s
    assert "self.opimpl_goto_if_not(condbox, 17, 5, replace=False)" in s


def test_int_guard_value():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn = ('int_guard_value', i0)
    work_list = WorkList({5: insn, 6: ('int_xor', 6)}, pc_to_nextpc={5: 6})

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
glob0(self, i1, i2) # jit_sync_regs_i1_i2
self.opimpl_int_guard_value(ri0, 5)
ri0 = self.registers_i[0]
i0 = ri0.getint()
pc = 110
continue"""

def test_instance_ptr_eq():
    r0, r1, i2 = Register('ref', 0), Register('ref', 1), Register('int', 2)
    insn = ('instance_ptr_eq', r0, r1, '->', i2)
    pc_to_insn = {5: insn, 6: ('int_return', i2)}

    # unspecialized case
    # every register is unconstant
    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc(set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
_b0 = self.registers_r[0]
_b1 = self.registers_r[1]
_v0 = self.registers_r[0].getref_base()
_v1 = self.registers_r[1].getref_base()
_res = int(_v0 == _v1)
if isinstance(_b0, Const) and isinstance(_b1, Const):
    self.registers_i[2] = const_int(_res)
    i2 = _res
    pc = %d
    continue
# fast-path recording, skip heapcache
_op = self.metainterp.history.record2_int(rop.INSTANCE_PTR_EQ, _b0, _b1, _res)
self.registers_i[2] = _op
i2 = _res
pc = 6
continue""" % (work_list.OFFSET + max(pc_to_insn))

    work_list = WorkList(pc_to_insn, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_pc({r0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 106
    s = insn_specializer.make_code()
    assert s == """\
glob0(self, r0) # jit_sync_regs_r0
_b0 = self.registers_r[0]
_b1 = self.registers_r[1]
_v0 = r0
_v1 = self.registers_r[1].getref_base()
_res = int(_v0 == _v1)
if isinstance(_b0, Const) and isinstance(_b1, Const):
    self.registers_i[2] = const_int(_res)
    i2 = _res
    pc = %d
    continue
# fast-path recording, skip heapcache
_op = self.metainterp.history.record2_int(rop.INSTANCE_PTR_EQ, _b0, _b1, _res)
self.registers_i[2] = _op
i2 = _res
pc = %d
continue""" % (work_list.OFFSET + max(pc_to_insn) + 1,
               work_list.OFFSET + max(pc_to_insn) + 2)

    # specialized case
    work_list = WorkList({5: insn, 6: ('int_return', 6)}, pc_to_nextpc={5: 6})
    insn_specializer = work_list.specialize_insn(insn, {r0, r1}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 6
    s = insn_specializer.make_code()
    assert s == """\
i2 = r0 == r1
pc = %d
continue""" % (work_list.OFFSET + max(pc_to_insn) + 1)
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == {r0, r1, i2}

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
    i0 = support.ptr2int(lltype.cast_opaque_ptr(OBJECTPTR, rr0.getref_base()).typeptr)
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


def test_ref_copy():
    r0, r2 = Register('ref', 0), Register('ref', 2)
    insn = ('ref_copy', r2, '->', r0)
    work_list = WorkList({5: insn, 6: ('ref_return', r0)}, pc_to_nextpc={5: 6})

    # specialized case
    insn_specializer = work_list.specialize_insn(insn, {r2}, 5)
    s = insn_specializer.make_code()
    assert s == """\
r0 = r2
pc = 107
continue"""

    # unspecialized cases
    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """\
rr2 = self.registers_r[2]
self.registers_r[0] = rr2
pc = 6
continue"""


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


def test_goto_if_not():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    L1 = TLabel('L1')
    insn = ('goto_if_not', i0, L1)
    pc_to_insn = {5: insn, 17: ('int_add', i0, i1, '->', i2), 19: ('int_return', i2)}
    work_list = WorkList(pc_to_insn, label_to_pc={'L1': 19}, pc_to_nextpc={5: 17, 17: 19})

    insn_specializer = work_list.specialize_pc({i0}, 5)
    s = insn_specializer.make_code()
    assert s == """\
cond = i0
if not cond:
    pc = 120
    continue
pc = 121
continue"""

    insn_specializer = work_list.specialize_pc({}, 5)
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 119
    continue
self.opimpl_goto_if_not(ri0, 19, 5)
pc = self.pc
if pc == 19:
    pc = 19
else:
    assert self.pc == 17
    pc = 17
continue"""


def test_int_is_true():
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    insn1 = ('int_is_true', i0, '->', i1)
    work_list = WorkList({5: insn1, 7: ('int_return', i1)}, pc_to_nextpc={5:7})
    insn_specializer = work_list.specialize_insn(insn1, {i0}, 5)
    newpc = insn_specializer.get_pc()
    assert newpc == work_list.OFFSET + 7
    s = insn_specializer.make_code()
    assert s == """i1 = int(bool(i0))
pc = 108
continue"""

    insn_specializer = work_list.specialize_insn(insn1, set(), 5)
    newpc = insn_specializer.get_pc()
    assert newpc == 5
    s = insn_specializer.make_code()
    assert s == """\
ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 109
    continue
else:
    _v0 = ri0.getint()
    _res = int(bool(_v0))
    _op = self.metainterp.history.record1_int(rop.INT_IS_TRUE, ri0, _res)
    self.registers_i[1] = _op
    i1 = _res
    pc = 7
    continue"""
    next_constant_registers = insn_specializer.get_next_constant_registers()
    assert next_constant_registers == set()

def test_assert_not_none():
    i0, i1 = Register('int', 0), Register('int', 1)
    insn = ('assert_not_none', i0)
    work_list = WorkList({5: insn, 7: ('int_return', i1)}, pc_to_nextpc={5:7})
    insn_specializer = work_list.specialize_insn(insn, {i0}, 5)
    assert insn_specializer.make_code() == """\
assert bool(i0)
pc = 108
continue"""

def test_raise_generates_standard_fallback(enable_genextension):
    ssarepr = SSARepr("raise_test", genextension=True)
    r0 = Register('ref', 0)
    ssarepr.insns = [
        ('raise', r0),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'ref': 1})
    source = jitcode._genext_source
    assert jitcode.genext_function is not None
    assert "def jit_shortcut(self): # raise_test" in source
    assert "self.pc = 0" in source
    assert "return self._run_one_step_standard()" in source


def test_inline_call_disables_genextension(enable_genextension):
    ssarepr = SSARepr("inline_call_test", genextension=True)
    i0, r0, i1 = Register('int', 0), Register('ref', 0), Register('int', 1)
    callee = JitCode("callee")
    ssarepr.insns = [
        ('inline_call_ir_i', callee, ListOfKind('int', [i0]),
         ListOfKind('ref', [r0]), '->', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2, 'ref': 1})
    assert jitcode.genext_function is None
    assert not hasattr(jitcode, '_genext_source')


def test_arrayitem_vable_disables_genextension(enable_genextension):
    ssarepr = SSARepr("arrayitem_vable_test", genextension=True)
    r0, r1 = Register('ref', 0), Register('ref', 1)
    i0 = Register('int', 0)
    fdescr = AbstractDescr()
    adescr = AbstractDescr()
    ssarepr.insns = [
        ('getarrayitem_vable_r', r0, i0, fdescr, adescr, '->', r1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1, 'ref': 2})
    assert jitcode.genext_function is None
    assert not hasattr(jitcode, '_genext_source')


def test_setarrayitem_vable_disables_genextension(enable_genextension):
    ssarepr = SSARepr("setarrayitem_vable_test", genextension=True)
    r0, r1 = Register('ref', 0), Register('ref', 1)
    i0 = Register('int', 0)
    fdescr = AbstractDescr()
    adescr = AbstractDescr()
    ssarepr.insns = [
        ('setarrayitem_vable_r', r0, i0, r1, fdescr, adescr),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1, 'ref': 2})
    assert jitcode.genext_function is None
    assert not hasattr(jitcode, '_genext_source')


def test_dispatch_prefix_shortcut_is_generated():
    ssarepr = SSARepr("dispatch_bytecode__AccessDirect_None",
                      genextension=True)
    r0, r1, r4 = Register('ref', 0), Register('ref', 1), Register('ref', 4)
    i0, i2 = Register('int', 0), Register('int', 2)
    last_instr_descr = AbstractDescr()
    debugdata_descr = AbstractDescr()
    switch_descr = SwitchDictDescr()
    switch_descr.attach({
        23: 200,
        100: 300,
        124: 400,
    })
    ssarepr.insns = [
        ('setfield_vable_i', r0, i0, last_instr_descr),
        ('getfield_vable_r', r0, debugdata_descr, '->', r4),
        ('goto_if_not_ptr_nonzero', r4, TLabel('debug_zero')),
        ('goto_if_not_int_ge', i2, Constant(90, lltype.Signed),
         TLabel('no_longarg')),
        ('switch', i2, switch_descr),
    ]
    ssarepr._insns_pos = [3, 14, 22, 52, 84]

    class FakeAssembler(object):
        insns = {}
        label_positions = {
            'debug_zero': 43,
            'no_longarg': 111,
        }

    jitcode = JitCode("dispatch_bytecode__AccessDirect_None")
    genext = GenExtension(FakeAssembler(), ssarepr, jitcode)

    assert genext._try_generate_dispatch_prefix_shortcut()
    source = jitcode._genext_source
    assert "def jit_shortcut(self): # dispatch_bytecode__AccessDirect_None prefix" in source
    assert source.count("metainterp.execute_and_record(rop.INT_ADD") == 3
    assert "metainterp.execute_and_record(rop.INT_MUL" in source
    assert "metainterp.execute_and_record(rop.INT_OR" in source
    assert "elif op == 124:" in source


def test_float_add():
    f0, f1, f2 = Register('float', 0), Register('float', 1), Register('float', 2)
    insn = ('float_add', f0, f1, '->', f2)
    work_list = WorkList({5: insn, 7: ('void_return',)}, pc_to_nextpc={5: 7})
    insn_specializer = work_list.specialize_insn(insn, {f0, f1}, 5)
    s = insn_specializer.make_code()
    assert s == """f2 = f0 + f1
pc = 108
continue"""


def test_strlen_unicodelen():
    r0, i1 = Register('ref', 0), Register('int', 1)
    insn = ('strlen', r0, '->', i1)
    work_list = WorkList({5: insn, 7: ('void_return',)}, pc_to_nextpc={5: 7})
    insn_specializer = work_list.specialize_insn(insn, {r0, i1}, 5)
    s = insn_specializer.make_code()
    assert s == """i1 = len(lltype.cast_opaque_ptr(lltype.Ptr(rstr.STR), r0).chars)
pc = 108
continue"""

    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 109
    continue
else:
    self.registers_i[1] = self.opimpl_strlen(rr0)
pc = 7
continue"""

    insn = ('unicodelen', r0, '->', i1)
    work_list = WorkList({5: insn, 7: ('void_return',)}, pc_to_nextpc={5: 7})
    insn_specializer = work_list.specialize_insn(insn, {r0, i1}, 5)
    s = insn_specializer.make_code()
    assert s == """i1 = len(lltype.cast_opaque_ptr(lltype.Ptr(rstr.UNICODE), r0).chars)
pc = 108
continue"""

    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """rr0 = self.registers_r[0]
if isinstance(rr0, ConstPtr):
    r0 = rr0.getref_base()
    pc = 109
    continue
else:
    self.registers_i[1] = self.opimpl_unicodelen(rr0)
pc = 7
continue"""


def test_residual_call():
    class FakeCallDescr(AbstractDescr):
        def __init__(self, effectinfo):
            self.effectinfo = effectinfo

        def get_extra_info(self):
            return self.effectinfo

    effectinfo = EffectInfo([], [], [], [], [], [],
                            extraeffect=EffectInfo.EF_ELIDABLE_CAN_RAISE)
    descr = FakeCallDescr(effectinfo)
    i0, i1 = Register('int', 0), Register('int', 1)
    func = Constant(llmemory.NULL, llmemory.GCREF)
    insn = ('residual_call_ir_i', func,
            ListOfKind('int', [i0]), ListOfKind('ref', []), descr, '->', i1)
    work_list = WorkList({5: insn, 7: ('int_return', i1)}, pc_to_nextpc={5: 7})
    insn_specializer = work_list.specialize_insn(insn, {i0, i1}, 5)
    s = insn_specializer.make_code()
    assert s == """v0 = self.do_residual_call(ConstPtr(lltype.cast_opaque_ptr(llmemory.GCREF, glob1)), [const_int(i0)], glob0, 5)
i1 = v0.getint()
pc = 108
continue"""

    insn_specializer = work_list.specialize_insn(insn, set(), 5)
    s = insn_specializer.make_code()
    assert s == """ri0 = self.registers_i[0]
if isinstance(ri0, ConstInt):
    i0 = ri0.getint()
    pc = 109
    continue
else:
    ri0 = self.registers_i[0]
    v0 = self.do_residual_call(ConstPtr(lltype.cast_opaque_ptr(llmemory.GCREF, glob3)), [self.registers_i[0]], glob2, 5)
    i1 = v0.getint()
    self.registers_i[1] = v0
    pc = 7
    continue"""


def test_hbp_signals_dispatch_heavy(enable_genextension):
    ssarepr = SSARepr("dispatchy", genextension=True)
    i0 = Register('int', 0)
    ssarepr.insns = [
        (Label('L1'),),
        ('goto_if_not_int_gt', i0, Constant(1, lltype.Signed), TLabel('L2')),
        ('goto_if_not_int_gt', i0, Constant(5, lltype.Signed), TLabel('L2')),
        ('goto_if_not_int_gt', i0, Constant(9, lltype.Signed), TLabel('L2')),
        ('int_return', i0),
        (Label('L2'),),
        ('int_return', i0),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 1})
    # 5 real ops, 3 goto_if_not_ -> gbd=0.6, score=0.42 > threshold.
    assert jitcode.genext_hbp_candidate is True
    assert jitcode.genext_hbp_score > 0.20


def test_hbp_signals_arithmetic_only(enable_genextension):
    ssarepr = SSARepr("arith", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_add', i1, i0, '->', i1),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_mul', i1, i0, '->', i1),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode.genext_hbp_candidate is False
    assert jitcode.genext_hbp_score == 0.0


def test_genext_compile_function_target_gated(enable_genextension):
    # x86-64: genext_compile_function is the x86 emitter (not None).
    # aarch64: T1 stage 1a installs a SCAFFOLD closure (not None) whose
    # probe() is hard-False -> the assembler always falls back to the
    # normal backend (provable no-op).  Other targets: still None.
    # genext_is_pure_arithmetic / genext_function are unaffected.
    from rpython.jit.backend import detect_cpu
    ssarepr = SSARepr("arith_gate", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_add', i1, i0, '->', i1),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('int_return', i1),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode.genext_is_pure_arithmetic is True
    try:
        target = detect_cpu.autodetect()
    except Exception:
        target = None
    if target == detect_cpu.MODEL_X86_64:
        assert jitcode.genext_compile_function is not None
    elif target == detect_cpu.MODEL_ARM64:
        # stage 1a scaffold installed, but probe is hard-False
        assert jitcode.genext_compile_function is not None
        assert jitcode.genext_compile_function(None, None, None, True) is False
    else:
        assert jitcode.genext_compile_function is None


def test_genext_compile_function_accepts_checked_int_arithmetic(enable_genextension):
    from rpython.jit.backend import detect_cpu
    ssarepr = SSARepr("checked_arith_gate", genextension=True)
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    ssarepr.insns = [
        (Label('L1'),),
        ('int_add_jump_if_ovf', TLabel('L2'), i0, i1, '->', i2),
        ('int_mul_jump_if_ovf', TLabel('L2'), i2, i1, '->', i2),
        ('int_return', i2),
        ('---',),
        (Label('L2'),),
        ('int_return', Constant(-1, lltype.Signed)),
        ('---',),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})
    assert jitcode.genext_is_pure_arithmetic is True
    try:
        target = detect_cpu.autodetect()
    except Exception:
        target = None
    if target == detect_cpu.MODEL_X86_64:
        assert jitcode.genext_compile_function is not None
    elif target == detect_cpu.MODEL_ARM64:
        assert jitcode.genext_compile_function is not None
        assert jitcode.genext_compile_function(None, None, None, True) is False
    else:
        assert jitcode.genext_compile_function is None


def test_genext_compile_function_accepts_int_comparison_loop(enable_genextension):
    from rpython.jit.backend import detect_cpu
    ssarepr = SSARepr("comparison_arith_gate", genextension=True)
    i0, i1 = Register('int', 0), Register('int', 1)
    ssarepr.insns = [
        (Label('L1'),),
        ('goto_if_not_int_gt', i0, Constant(0, lltype.Signed), TLabel('L2')),
        ('int_add', i1, i0, '->', i1),
        ('int_sub', i0, Constant(1, lltype.Signed), '->', i0),
        ('goto', TLabel('L1')),
        ('---',),
        (Label('L2'),),
        ('int_return', i1),
        ('---',),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 2})
    assert jitcode.genext_is_pure_arithmetic is True
    try:
        target = detect_cpu.autodetect()
    except Exception:
        target = None
    if target == detect_cpu.MODEL_X86_64:
        assert jitcode.genext_compile_function is not None
    elif target == detect_cpu.MODEL_ARM64:
        assert jitcode.genext_compile_function is not None
        assert jitcode.genext_compile_function(None, None, None, True) is False
    else:
        assert jitcode.genext_compile_function is None
