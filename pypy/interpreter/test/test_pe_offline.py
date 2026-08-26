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


def _setup_loop_target(co_code, setup_pc):
    """The break target CPython encodes at a SETUP_LOOP instruction."""
    setup_loop = bytecode_spec.opcodedesc.SETUP_LOOP.index
    assert ord(co_code[setup_pc]) == setup_loop
    oparg = ord(co_code[setup_pc + 1]) | (ord(co_code[setup_pc + 2]) << 8)
    return setup_pc + 3 + oparg


def _break_loop_targets(func_code):
    break_loop = bytecode_spec.opcodedesc.BREAK_LOOP.index
    co_code = func_code.co_code
    targets = []
    pc = 0
    while pc < len(co_code):
        opcode, bindings = pe_offline.decode_instruction(func_code, pc)
        if opcode == break_loop:
            targets.append(bindings["break_target"])
        pc = bindings["pc"]
    return targets


def test_break_inside_try_body_keeps_enclosing_loop():
    # Shaped like eatWhitespace: the `break` inside the try body's normal
    # exit is POP_BLOCK (closes SETUP_EXCEPT) + BREAK_LOOP -- a block-stack
    # scan mistakes that POP_BLOCK for the try's own and loses the loop.
    def eat_whitespace(f):
        while True:
            try:
                x = f()
                if x == 1:
                    break
                elif x == 2:
                    y = x
                else:
                    y = 0
            except EOFError:
                break
        return y

    co_code = eat_whitespace.func_code.co_code
    setup_loop = bytecode_spec.opcodedesc.SETUP_LOOP.index
    setup_pc = co_code.index(chr(setup_loop))
    expected = _setup_loop_target(co_code, setup_pc)

    targets = _break_loop_targets(eat_whitespace.func_code)
    assert targets
    assert targets == [expected] * len(targets)


def test_nested_loops_resolve_to_their_own_loop():
    def nested(a, b):
        while a:
            while b:
                if b:
                    break
                b = None
            if a:
                break
        return a

    func_code = nested.func_code
    co_code = func_code.co_code
    setup_loop = bytecode_spec.opcodedesc.SETUP_LOOP.index
    setup_positions = [i for i in range(len(co_code))
                        if ord(co_code[i]) == setup_loop]
    assert len(setup_positions) == 2
    outer_target = _setup_loop_target(co_code, setup_positions[0])
    inner_target = _setup_loop_target(co_code, setup_positions[1])
    assert outer_target != inner_target

    ends = pe_offline._loop_ends(co_code)
    break_loop = bytecode_spec.opcodedesc.BREAK_LOOP.index
    seen = set()
    pc = 0
    while pc < len(co_code):
        opcode, bindings = pe_offline.decode_instruction(func_code, pc)
        if opcode == break_loop:
            seen.add(bindings["break_target"])
            inner_body_start = setup_positions[1] + 3
            if inner_body_start <= pc < inner_target:
                assert bindings["break_target"] == inner_target
            else:
                assert bindings["break_target"] == outer_target
        pc = bindings["pc"]
    assert seen == set([outer_target, inner_target])


def test_break_loop_without_enclosing_loop_declines():
    break_loop = bytecode_spec.opcodedesc.BREAK_LOOP.index
    code = FakeCode(chr(break_loop))
    py.test.raises(BytecodeCorruption, pe_offline.decode_instruction,
                   code, 0)


if __name__ == "__main__":
    test_decode_matches_dis_on_real_functions()
    test_extended_arg_folds_into_the_following_instruction()
    test_opcode_keys_are_unique_and_in_range()
    test_break_inside_try_body_keeps_enclosing_loop()
    test_nested_loops_resolve_to_their_own_loop()
    test_break_loop_without_enclosing_loop_declines()
    print "ok"


class FakeProfiler(object):
    def __init__(self, tracing_s):
        self.tracing_s = tracing_s

    def get_times(self, num):
        from rpython.rlib.jit import Counters
        if num == Counters.TRACING:
            return self.tracing_s
        return 0.0


def test_gate_does_not_charge_the_estimate(monkeypatch):
    state = pe_offline._gate_state
    monkeypatch.setattr(state, "k", 1.0)
    monkeypatch.setattr(state, "env_read", True)
    monkeypatch.setattr(state, "spent_ns", 0.0)
    profiler = FakeProfiler(tracing_s=1.0)
    code_size = 1000   # 20ms estimated: within k and within the budget
    assert pe_offline._gate_allows(profiler, code_size)
    assert pe_offline._gate_allows(profiler, code_size)
    assert state.spent_ns == 0.0
    state.spent_ns = pe_offline.GATE_BUDGET_FRACTION * 1e9
    assert not pe_offline._gate_allows(profiler, code_size)
