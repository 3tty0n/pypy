from rpython.jit.codewriter.assembler import Assembler
from rpython.jit.codewriter.flatten import SSARepr, Register, Label, TLabel
from rpython.jit.codewriter import jitcode as jitcode_mod
from rpython.flowspace.model import Constant
from rpython.rtyper.lltypesystem import lltype


def test_dump_jitcode(monkeypatch):
    ssarepr = SSARepr("test")
    i0, i1, i2 = Register('int', 0), Register('int', 1), Register('int', 2)
    ssarepr.insns = [
        (Label('top'),),
        ('int_add', i0, Constant(300, lltype.Signed), '->', i2),
        ('int_sub', i2, Constant(7, lltype.Signed), '->', i1),
        ('goto_if_not_int_lt', i1, i0, TLabel('top')),
        ('int_return', i2),
        ]
    assembler = Assembler()
    jitcode = assembler.assemble(ssarepr, num_regs={'int': 3})

    class FakeSD:
        opcode_names = ['?'] * len(assembler.insns)
        opcode_descrs = assembler.descrs
        op_live = assembler.insns.get('live/', -1)
    for key, value in assembler.insns.items():
        FakeSD.opcode_names[value] = key
    lines = []
    monkeypatch.setattr(jitcode_mod, "debug_print", lines.append,
                        raising=False)
    import rpython.rlib.debug
    monkeypatch.setattr(rpython.rlib.debug, "debug_print", lines.append)
    jitcode_mod._dump_jitcode(jitcode, FakeSD)
    assert lines[0].startswith("jitcode test: ")
    assert lines[1] == "    0: int_add %i0 $300 -> %i2"
    assert lines[2] == "    4: int_sub %i2 7 -> %i1"
    assert lines[3] == "    8: goto_if_not_int_lt %i1 %i0 L0"
    assert lines[4] == "   13: int_return %i2"
