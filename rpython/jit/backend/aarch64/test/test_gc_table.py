from rpython.jit.backend.aarch64.arch import WORD
from rpython.jit.backend.aarch64.assembler import AssemblerARM64
from rpython.jit.backend.aarch64.codebuilder import InstrBuilder
from rpython.jit.backend.aarch64 import registers as r
from rpython.rtyper.lltypesystem import lltype, rffi

load_from_gc_table = AssemblerARM64.load_from_gc_table.__func__
write_far_gc_table_load = AssemblerARM64.write_far_gc_table_load.__func__


class FakeAssembler(object):
    def __init__(self):
        self.mc = InstrBuilder()
        self.gc_table_far_patches = []


def decode(mc):
    n = mc.get_relative_pos(break_basic_block=False)
    buf = lltype.malloc(rffi.CCHARP.TO, n, flavor='raw')
    try:
        mc._copy_to_raw_memory(rffi.cast(lltype.Signed, buf))
        b = [ord(buf[i]) for i in range(n)]
    finally:
        lltype.free(buf, flavor='raw')
    return [b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24)
            for i in range(0, n, 4)]


def address_of(adrp, ldr, p_location):
    pages = (((adrp >> 5) & 0x7ffff) << 2) | ((adrp >> 29) & 0x3)
    if pages >= (1 << 20):
        pages -= 1 << 21
    return ((p_location & ~0xFFF) + (pages << 12) +
            (((ldr >> 10) & 0xFFF) << 3))


def test_near_offset_stays_one_instruction():
    asm = FakeAssembler()
    asm.mc.NOP()
    before = asm.mc.get_relative_pos(break_basic_block=False)
    load_from_gc_table(asm, r.ip0.value, 3)
    assert asm.mc.get_relative_pos(break_basic_block=False) - before == 4
    assert asm.gc_table_far_patches == []


def test_far_offset_reserves_two_instructions():
    index = 3
    asm = FakeAssembler()
    while index * WORD - asm.mc.get_relative_pos(break_basic_block=False) >= -(1 << 20):
        asm.mc.NOP()
    patch_pos = asm.mc.get_relative_pos(break_basic_block=False)
    load_from_gc_table(asm, r.ip0.value, index)
    assert asm.gc_table_far_patches == [(patch_pos, r.ip0.value, index)]
    assert asm.mc.get_relative_pos(break_basic_block=False) - patch_pos == 8


def test_patched_pair_addresses_the_entry():
    asm = FakeAssembler()
    for rawstart in [0x400000000, 0x402a4c000, 0x42f000ff0]:
        for pos in [0x100000, 0x100ff8, 0x2abcd0]:
            for index in [0, 3, 511]:
                entry = rawstart + index * WORD
                p_location = rawstart + pos
                mc = InstrBuilder()
                write_far_gc_table_load(asm, mc, r.ip0.value, entry,
                                        p_location)
                adrp, ldr = decode(mc)
                assert address_of(adrp, ldr, p_location) == entry
