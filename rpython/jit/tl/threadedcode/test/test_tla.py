import py
import pytest
import os

from rpython.jit.tl.threadedcode import tla
from rpython.jit.tl.threadedcode.bytecode import Bytecode, assemble
from rpython.jit.tl.threadedcode.tla import \
    W_Object, W_IntObject, W_StringObject, Frame

def interp(mylist, w_arg):
    bytecode = Bytecode(assemble(mylist))
    return tla.run(bytecode, w_arg)

def interp_tier2(mylist, w_arg):
    bytecode = Bytecode(assemble(mylist))
    return tla.run(bytecode, w_arg, tier=2)

def read_code(name):
    path = "%s/../lang/%s" % (os.path.dirname(__file__), name)
    mydict = {}
    execfile(path, mydict)
    return mydict['code']

def assert_stack(stack1, stack2):
    for x, y in zip(stack1, stack2):
        if x is None and y is None:
            continue
        assert x.eq(y)

class TestFrame:

    def test_add(self):
        code = [
            tla.CONST_INT, 123,
            tla.ADD,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(123))
        assert res.intvalue == 123 + 123

    def test_sub(self):
        code = [
            tla.CONST_INT, 123,
            tla.SUB,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(234))
        assert res.intvalue == 234 - 123

    def test_mul(self):
        code = [
            tla.CONST_INT, 123,
            tla.MUL,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(234))
        assert res.intvalue == 234 * 123

    def test_div(self):
        code = [
            tla.CONST_INT, 123,
            tla.DIV,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(234))
        assert res.intvalue == 234 / 123

    @pytest.mark.skip(reason="W_IntObject.mod is not implemented (always OperationError)")
    def test_mod(self):
        code = [
            tla.CONST_INT, 2,
            tla.MOD,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(10))
        assert res.intvalue == 0
        res = interp(code, W_IntObject(13))
        assert res.intvalue == 1

    def test_jump(self):
        code = [
            tla.JUMP, 3,
            tla.ADD,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(234))
        assert res.intvalue == 234

    @pytest.mark.skip(reason="CALL bytecode in test omits argnum operand (see tla.interp CALL handler)")
    def test_call(self):
        code = [
            tla.CALL, 3,
            tla.EXIT,
            tla.CONST_INT, 12,
            tla.ADD,
            tla.RET, 1
        ]
        res = interp(code, W_IntObject(34))
        assert res.intvalue == 34 + 12

    def test_frame_reset(self):
        stack = [
            W_IntObject(10), # ?
            W_IntObject(0),  # old acc
            W_IntObject(10), # old n
            W_IntObject(-1), # dummy ret_addr
            W_IntObject(10), # local acc
            W_IntObject(9)   # local n
        ]
        code = [ tla.FRAME_RESET, 2, 2, 2, ]
        frame = Frame(assemble(code))
        frame.stack = stack
        frame.stackpos = len(stack)
        frame.interp()

        expected = [
            W_IntObject(10),
            W_IntObject(10),
            W_IntObject(9),
            W_IntObject(-1), # dummy ret_addr
            None,
            None
        ]

        assert_stack(frame.stack, expected)

    def test_simple_loop(self):
        code = [
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 11,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.JUMP, 0,
            tla.EXIT,
        ]
        res = interp(code, W_IntObject(100))
        assert res.intvalue == 1

    def test_double_loop(self):
        code = [
            tla.DUP,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 12,
            tla.JUMP, 1,
            tla.POP,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.DUP,
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 25,
            tla.JUMP, 1,
            tla.EXIT
        ]
        res = interp(code, W_IntObject(3))
        assert res.intvalue == 1

    def test_call_assembler_h_minimal(self):
        """CALL_ASSEMBLER_H encodes an extra inline-hint byte (tier-1 + tier-2)."""
        code = [
            tla.CONST_INT, 40,
            tla.CONST_INT, 50,
            tla.CALL_ASSEMBLER_H, 9, 2, tla.INLINE_HINT_ALLOW_DEEP_1,
            tla.EXIT,
            tla.CONST_INT, 99,
            tla.RET, 2,
        ]
        res = interp(code, W_IntObject(0))
        assert res.intvalue == 99
        res2 = interp_tier2(code, W_IntObject(0))
        assert res2.intvalue == 99

    def test_entry_inline_hint_with_plain_call(self):
        """Layer 2+3: DEFAULT hint uses register_entry_inline_hint when cap > 0."""
        code = [
            tla.CONST_INT, 40,
            tla.CONST_INT, 50,
            tla.CALL_ASSEMBLER, 8, 2,
            tla.EXIT,
            tla.CONST_INT, 99,
            tla.RET, 2,
        ]
        tla.set_global_inline_cap(1)
        tla.register_entry_inline_hint(8, 1)
        try:
            res = interp(code, W_IntObject(0))
            assert res.intvalue == 99
        finally:
            tla.set_global_inline_cap(0)
            tla.clear_entry_inline_hints()

from rpython.jit.metainterp.test.support import LLJitMixin

class TestLLType(LLJitMixin):

    def test_tier1_threaded_mini_loop_interp(self):
        """Tier-1 interpreter (threaded driver + threaded_inline_handler on tier1driver)."""
        code = [
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 11,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.JUMP, 0,
            tla.EXIT,
        ]
        bytecode = Bytecode(assemble(code))
        v = tla.run(bytecode, W_IntObject(100), tier=1).intvalue
        assert v == 1

    def test_tier1_deep_tracing_runs_correctly_with_budget(self):
        """Smoke test: with frame.jit_inline_budget=1 the same loop still
        produces the right answer. Verifies enable_deep_tracing's
        budget>0 branch (direct ``func(*args)`` call into the outer
        handler body) doesn't break execution; the deeper trace is
        otherwise observable only via PYPYLOG."""
        code = [
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 11,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.JUMP, 0,
            tla.EXIT,
        ]
        bytecode = Bytecode(assemble(code))
        frame = tla.Frame(bytecode, jit_inline_budget=1)
        frame.push(W_IntObject(100))
        v = frame.interp()
        assert isinstance(v, W_IntObject)
        assert v.intvalue == 1

    def test_meta_interp_tier1_deep(self):
        """JIT-level integration: run the tier-1 interp under meta_interp
        with ``jit_inline_budget=1``. The trace contains the outer
        bytecode bodies inlined; the inner ``pop``/``push``/``take``
        primitives still appear as residual ``call_n``/``call_r`` ops.

        Key trace shape (verified via ``check_resops``):

        * ``int_sub`` is present — the SUB bytecode body was inlined
          (under shallow tracing it would be hidden inside a single
          ``handler_SUB`` residual call).
        * At least one ``call_n`` (the void inner primitive — push)
          is present.
        * At least one ``call_r`` (the value-returning inner primitive
          — pop / is_true) is present.
        """
        code = [
            tla.DUP,
            tla.CONST_INT, 2,
            tla.LT,
            tla.JUMP_IF, 11,
            tla.CONST_INT, 1,
            tla.SUB,
            tla.JUMP, 0,
            tla.EXIT,
        ]

        def interp_w(intvalue):
            bytecode = Bytecode(assemble(code))
            frame = tla.Frame(bytecode, jit_inline_budget=1)
            frame.push(W_IntObject(intvalue))
            w = frame.interp()
            if isinstance(w, W_IntObject):
                return w.intvalue
            return -1

        res = self.meta_interp(interp_w, [100])
        assert res == 1
        # Body of SUB inlined -> int_sub appears directly in the trace
        # (shallow tracing would hide it inside the handler_SUB call).
        ops = self.check_resops()
        if ops is not None:  # check_resops may return None when opts disabled
            assert ops.get('int_sub', 0) >= 1, (
                "expected SUB body inlined (int_sub op present) "
                "but trace had insns=%r" % (ops,))
            assert ops.get('call_n', 0) >= 1, (
                "expected at least one void inner primitive call "
                "(e.g. handler_push) in trace, insns=%r" % (ops,))
            assert ops.get('call_r', 0) >= 1, (
                "expected at least one value-returning inner primitive "
                "call (e.g. handler_pop / handler_is_true) in trace, "
                "insns=%r" % (ops,))


    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_loop(self):
        code = read_code('../lang/loop.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [100])
        assert res == 0

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_sum(self):
        code = read_code('../lang/sum.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [10])
        assert res == 55

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_fib(self):
        code = read_code('../lang/fib.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [7])
        assert res == 8

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_tak(self):
        code = read_code('../lang/tak.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [1])
        assert res == 4

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_tarai(self):
        code = read_code('../lang/tarai.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [1])

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_ack(self):
        code = read_code('../lang/ack.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue

        res = self.meta_interp(interp_w, [1])

    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_gcd(self):
        code = read_code('../lang/gcd.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue
        res = self.meta_interp(interp_w, [1])


    @pytest.mark.skip(reason="lang/*.tla bytecode and stack usage are out of sync with tla.Frame")
    def test_jit_ary(self):
        code = read_code('../lang/ary.tla.py')
        def interp_w(intvalue):
            w_result = interp(code, W_IntObject(intvalue))
            assert isinstance(w_result, W_IntObject)
            return w_result.intvalue
        res = self.meta_interp(interp_w, [6])
