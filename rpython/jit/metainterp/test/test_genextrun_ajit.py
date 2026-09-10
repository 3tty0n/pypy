import py

from rpython.rlib.jit import JitDriver
from rpython.jit.codewriter.genextrun import RunModeGenerator
from rpython.jit.metainterp import blackhole
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats


loopdriver = JitDriver(greens=[], reds=['n', 'res'])


def simple_loop(n):
    res = 0
    while n > 0:
        loopdriver.can_enter_jit(n=n, res=res)
        loopdriver.jit_merge_point(n=n, res=res)
        res += n
        n -= 1
    return res


guarddriver = JitDriver(greens=[], reds=['n', 'res'])


def guard_loop(n):
    res = 0
    while n > 0:
        guarddriver.can_enter_jit(n=n, res=res)
        guarddriver.jit_merge_point(n=n, res=res)
        if n % 7 == 3:
            res += 10
        else:
            res += 1
        n -= 1
    return res


class _Assembler(object):
    def __init__(self, insn_names):
        self.insns = dict((name, index)
                          for index, name in enumerate(insn_names))


def _mainjitcode():
    metainterp_sd = get_stats().metainterp_sd
    jitcode = metainterp_sd.jitdrivers_sd[0].mainjitcode
    assembler = _Assembler(metainterp_sd.blackholeinterpbuilder._insns)
    return metainterp_sd, assembler, jitcode


class TestRunModeAjit(LLJitMixin):

    @py.test.mark.xfail(strict=True, raises=NotImplementedError,
                        reason="run mode lacks -live-, jit_merge_point "
                               "and residual_call_ir_i")
    def test_loop_jitcode_runs(self):
        assert self.meta_interp(simple_loop, [10]) == simple_loop(10)
        metainterp_sd, assembler, jitcode = _mainjitcode()
        gen = RunModeGenerator(assembler, jitcode._ssarepr, jitcode)
        jit_run = gen.generate_run()
        bh = metainterp_sd.blackholeinterpbuilder.acquire_interp()
        bh.setposition(jitcode, 0)
        bh.setarg_i(0, 10)
        pc = jit_run(bh)
        if pc == -1:
            assert bh._final_result_anytype() == simple_loop(10)
        else:
            name = metainterp_sd.opcode_names[ord(jitcode.code[pc])]
            assert getattr(RunModeGenerator, "emit_run_" + name.split("/")[0],
                           None) is not None

    def _run_guard_loop(self):
        counter = []
        orig = blackhole.resume_in_blackhole

        def counting(*args, **kwds):
            counter.append(1)
            return orig(*args, **kwds)

        blackhole.resume_in_blackhole = counting
        try:
            res = self.meta_interp(guard_loop, [30])
        finally:
            blackhole.resume_in_blackhole = orig
        return res, len(counter)

    def test_guard_failure_lands_in_run_mode(self):
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count > 0

    @py.test.mark.xfail(strict=True,
                        reason="guard failures do not land in jit_run yet")
    def test_guard_failure_routed_through_run_mode(self):
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count == 0
