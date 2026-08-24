"""The PE decoder must tile a code object exactly as dispatch_bytecode does."""

import dis
import py

from pypy.interpreter import pe_offline
from pypy.interpreter.pycode import BytecodeCorruption
from pypy.tool.stdlib_opcode import bytecode_spec


class FakeCode(object):
    def __init__(self, co_code):
        self.co_code = co_code


def decode_all(co_code):
    """Every (pc, opcode, oparg) the decoder yields, walking the whole code."""
    code = FakeCode(co_code)
    result = []
    pc = 0
    while pc < len(co_code):
        opcode, bindings = pe_offline.decode_instruction(code, pc)
        result.append((pc, opcode, bindings["oparg"]))
        assert bindings["pc"] > pc
        pc = bindings["pc"]
    return result


def reference(co_code):
    """The same walk, done by the standard library's disassembler."""
    result = []
    pc = 0
    extended = 0
    start = 0
    while pc < len(co_code):
        opcode = ord(co_code[pc])
        if extended == 0:
            start = pc
        pc += 1
        oparg = 0
        if opcode >= dis.HAVE_ARGUMENT:
            oparg = ord(co_code[pc]) | (ord(co_code[pc + 1]) << 8)
            pc += 2
        if opcode == dis.EXTENDED_ARG:
            extended = (extended | oparg) << 16
            continue
        result.append((start, opcode, extended | oparg))
        extended = 0
    return result


def test_decode_matches_dis_on_real_functions():
    def loop(n):
        total = 0
        for i in range(n):
            if i % 3:
                total += i * 2
            else:
                total -= i
        return total

    def exceptions(x):
        try:
            return 1 / x
        except ZeroDivisionError:
            return 0
        finally:
            x = None

    for func in (loop, exceptions, reference, decode_all):
        co_code = func.func_code.co_code
        assert decode_all(co_code) == reference(co_code), func.__name__


def test_extended_arg_folds_into_the_following_instruction():
    ext = bytecode_spec.opcodedesc.EXTENDED_ARG.index
    jump = bytecode_spec.opcodedesc.JUMP_ABSOLUTE.index
    # EXTENDED_ARG 0x0001 ; JUMP_ABSOLUTE 0x0002  ->  one instruction, arg
    # 0x00010002, six bytes wide.
    co_code = (chr(ext) + chr(1) + chr(0) + chr(jump) + chr(2) + chr(0))
    assert decode_all(co_code) == [(0, jump, 0x00010002)]
    assert decode_all(co_code) == reference(co_code)


def test_opcode_keys_are_unique_and_in_range():
    keys = pe_offline.opcode_keys()
    assert keys == sorted(set(keys))
    assert keys and 0 <= keys[0] and keys[-1] < 256


def test_invalid_entry_pc_declines_instead_of_asserting():
    code = FakeCode(chr(bytecode_spec.opcodedesc.RETURN_VALUE.index))
    py.test.raises(BytecodeCorruption, pe_offline.decode_instruction,
                   code, -1)
    py.test.raises(BytecodeCorruption, pe_offline.decode_instruction,
                   code, len(code.co_code))


def test_break_loop_keeps_enclosing_loop_across_handler_pop_block():
    def loop_with_handler(items):
        while items:
            try:
                items.pop()
            except IndexError:
                break
            if items:
                break

    co_code = loop_with_handler.func_code.co_code
    break_loop = bytecode_spec.opcodedesc.BREAK_LOOP.index
    targets = []
    pc = 0
    while pc < len(co_code):
        opcode, bindings = pe_offline.decode_instruction(
            loop_with_handler.func_code, pc)
        if opcode == break_loop:
            targets.append(bindings["break_target"])
        pc = bindings["pc"]
    assert targets
    assert min(targets) >= 0


if __name__ == "__main__":
    test_decode_matches_dis_on_real_functions()
    test_extended_arg_folds_into_the_following_instruction()
    test_opcode_keys_are_unique_and_in_range()
    print "ok"
