"""optimise_copies must drop only copies the block itself overwrites unread."""

from rpython.translator.backendopt.native_fragments import NativeInsn, NReg
from rpython.translator.backendopt.native_pipeline import optimise_copies


def reg(kind, index):
    return NReg(kind, index)


def opcodes(insns):
    return [insn.opcode for insn in insns]


def test_dead_copy_is_dropped():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("int_copy", [reg("int", 1)], reg("int", 5)),
        NativeInsn("int_add", [reg("int", 5), reg("int", 2)], reg("int", 6)),
    ]
    assert opcodes(optimise_copies(insns)) == ["int_copy", "int_add"]


def test_live_copy_is_kept():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("int_add", [reg("int", 5), reg("int", 2)], reg("int", 6)),
    ]
    assert opcodes(optimise_copies(insns)) == ["int_copy", "int_add"]


def test_reads_are_rewritten_to_the_copy_source():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("int_add", [reg("int", 5), reg("int", 2)], reg("int", 6)),
    ]
    result = optimise_copies(insns)
    assert (result[1].operands[0].kind, result[1].operands[0].index) == \
        ("int", 0)


def test_a_barrier_forgets_what_was_known():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("---", [], None),
        NativeInsn("int_add", [reg("int", 5), reg("int", 2)], reg("int", 6)),
    ]
    result = optimise_copies(insns)
    assert len(result) == 3
    assert (result[2].operands[0].kind, result[2].operands[0].index) == \
        ("int", 5)


def test_a_barrier_stops_the_dead_copy_search():
    insns = [
        NativeInsn("int_copy", [reg("int", 0)], reg("int", 5)),
        NativeInsn("---", [], None),
        NativeInsn("int_copy", [reg("int", 1)], reg("int", 5)),
    ]
    assert len(optimise_copies(insns)) == 3


if __name__ == "__main__":
    test_dead_copy_is_dropped()
    test_live_copy_is_kept()
    test_reads_are_rewritten_to_the_copy_source()
    test_a_barrier_forgets_what_was_known()
    test_a_barrier_stops_the_dead_copy_search()
    print "ok"
