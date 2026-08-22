"""fold_constant_loads substitutes a constant only where the argcode holds."""

from rpython.translator.backendopt.native_fragments import (
    NativeInsn, NReg, NIntConst, NRefConst)
from rpython.translator.backendopt.native_pipeline import fold_constant_loads


def reg(kind, index):
    return NReg(kind, index)


def test_a_wide_constant_reaches_its_reader():
    big = NIntConst(1 << 40)
    insns = [
        NativeInsn("int_copy", [big], reg("int", 5)),
        NativeInsn("int_add", [reg("int", 5), reg("int", 2)], reg("int", 6)),
    ]
    result = fold_constant_loads(insns)
    assert [i.opcode for i in result] == ["int_add"]
    assert result[0].operands[0] is big


def test_a_short_constant_stays_in_its_register():
    """Folding it would turn the reader's argcode from "i" into "c"."""
    insns = [
        NativeInsn("int_copy", [NIntConst(3)], reg("int", 5)),
        NativeInsn("int_add", [reg("int", 5)], reg("int", 6)),
    ]
    result = fold_constant_loads(insns)
    assert len(result) == 2
    assert isinstance(result[1].operands[0], NReg)


def test_a_ref_constant_is_folded():
    ref = NRefConst(None)
    insns = [
        NativeInsn("ref_copy", [ref], reg("ref", 0)),
        NativeInsn("setfield_gc_r", [reg("ref", 1), reg("ref", 0)], None),
        NativeInsn("ref_copy", [NRefConst(None)], reg("ref", 0)),
    ]
    result = fold_constant_loads(insns)
    # The trailing load has no reader left either, so both go.
    assert len(result) == 1
    assert result[0].operands[1] is ref


def test_a_load_and_its_only_reader_collapse():
    ref = NRefConst(None)
    insns = [
        NativeInsn("ref_copy", [ref], reg("ref", 0)),
        NativeInsn("setfield_gc_r", [reg("ref", 0)], None),
    ]
    result = fold_constant_loads(insns)
    assert len(result) == 1
    assert result[0].operands[0] is ref


def test_a_register_a_region_reads_first_is_left_alone():
    insns = [
        NativeInsn("setfield_gc_r", [reg("ref", 0)], None),
        NativeInsn("@label", [], None),
        NativeInsn("ref_copy", [NRefConst(None)], reg("ref", 0)),
        NativeInsn("setfield_gc_r", [reg("ref", 0)], None),
    ]
    assert len(fold_constant_loads(insns)) == 4


def test_a_live_record_naming_others_proves_the_target_dead():
    ref = NRefConst(None)
    insns = [
        NativeInsn("ref_copy", [ref], reg("ref", 0)),
        NativeInsn("-live-", [reg("ref", 9)], None),
        NativeInsn("setfield_gc_r", [reg("ref", 0)], None),
    ]
    result = fold_constant_loads(insns)
    assert len(result) == 2
    assert result[1].operands[0] is ref


def test_a_register_move_is_not_a_constant_load():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("int_copy", [reg("int", 1)], reg("int", 5)),
    ]
    assert fold_constant_loads(insns) is insns


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print "ok"
