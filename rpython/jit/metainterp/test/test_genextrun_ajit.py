import py

from rpython.rlib.jit import JitDriver
from rpython.jit.codewriter.genextrun import RunModeGenerator
from rpython.jit.metainterp import blackhole, jitexc
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

    def _generate_loop_jit_run(self):
        assert self.meta_interp(simple_loop, [10]) == simple_loop(10)
        metainterp_sd, assembler, jitcode = _mainjitcode()
        gen = RunModeGenerator(assembler, jitcode._ssarepr, jitcode)
        return metainterp_sd, jitcode, gen.generate_run()

    def _new_bh(self, metainterp_sd, jitcode, position, n, res):
        bh = metainterp_sd.blackholeinterpbuilder.acquire_interp()
        bh.nextblackholeinterp = None
        bh.setposition(jitcode, position)
        bh.setarg_i(0, n)
        bh.setarg_i(1, res)
        return bh

    def _pc_of(self, jitcode, opname):
        ssarepr = jitcode._ssarepr
        index = [i for i, insn in enumerate(ssarepr.insns)
                 if insn[0] == opname][0]
        return ssarepr._insns_pos[index]

    def test_loop_jitcode_runs(self):
        metainterp_sd, jitcode, jit_run = self._generate_loop_jit_run()
        bh = self._new_bh(metainterp_sd, jitcode, 0, 10, 0)
        with py.test.raises(jitexc.ContinueRunningNormally) as excinfo:
            jit_run(bh)
        assert excinfo.value.red_int == [10, 0]

    def test_resume_mid_iteration_stops_at_the_merge_point(self):
        metainterp_sd, jitcode, jit_run = self._generate_loop_jit_run()
        start = self._pc_of(jitcode, 'int_add')
        bh = self._new_bh(metainterp_sd, jitcode, start, 10, 0)
        with py.test.raises(jitexc.ContinueRunningNormally) as excinfo:
            jit_run(bh)
        assert excinfo.value.red_int == [9, 10]

    def test_resume_in_the_last_iteration_returns(self):
        metainterp_sd, jitcode, jit_run = self._generate_loop_jit_run()
        start = self._pc_of(jitcode, 'int_add')
        bh = self._new_bh(metainterp_sd, jitcode, start, 1, 41)
        assert jit_run(bh) == -1
        assert bh._final_result_anytype() == 42

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
