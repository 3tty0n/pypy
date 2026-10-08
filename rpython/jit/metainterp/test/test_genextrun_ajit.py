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


class Box(object):
    def __init__(self, x):
        self.x = x


fielddriver = JitDriver(greens=[], reds=['n', 'res', 'box'])


def field_loop(n):
    box = Box(n)
    res = 0
    while n > 0:
        fielddriver.can_enter_jit(n=n, res=res, box=box)
        fielddriver.jit_merge_point(n=n, res=res, box=box)
        if n % 7 == 3:
            res += box.x
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

    def _run_guard_loop(self, func=guard_loop):
        counter = []
        orig = blackhole.resume_in_blackhole

        def counting(*args, **kwds):
            counter.append(1)
            return orig(*args, **kwds)

        blackhole.resume_in_blackhole = counting
        try:
            res = self.meta_interp(func, [30])
        finally:
            blackhole.resume_in_blackhole = orig
        return res, len(counter)

    def test_guard_failure_lands_in_run_mode(self):
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count > 0

    def _install_run_mode(self, monkeypatch):
        from rpython.jit.codewriter.codewriter import CodeWriter
        entries = []
        generated = {}
        make_jitcodes = CodeWriter.make_jitcodes

        def install(cw, verbose=False):
            jitcodes = make_jitcodes(cw, verbose)
            for jitcode in jitcodes:
                gen = RunModeGenerator(cw.assembler, jitcode._ssarepr, jitcode)
                jit_run = gen.generate_run()

                def counting(bh, jit_run=jit_run):
                    entries.append((bh.jitcode.name, bh.position))
                    return jit_run(bh)
                jitcode.genext_run_function = counting
                generated[jitcode.name] = jitcode
            return jitcodes

        monkeypatch.setattr(CodeWriter, 'make_jitcodes', install)
        return entries, generated

    def test_e2e_loop_exit_runs_in_run_mode(self, monkeypatch):
        entries, _ = self._install_run_mode(monkeypatch)
        assert self.meta_interp(simple_loop, [100]) == simple_loop(100)
        assert entries == [('simple_loop', 27)]

    def test_e2e_guard_failures_run_in_run_mode(self, monkeypatch):
        entries, _ = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count == 2
        assert entries == [('guard_loop', 28), ('guard_loop', 28)]

    def test_e2e_fallback_returns_to_run_mode(self, monkeypatch):
        entries, generated = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(field_loop)
        assert res == field_loop(30)
        assert count == 2
        assert entries == [('field_loop', 29), ('field_loop', 39),
                           ('field_loop', 29)]
        assert generated['field_loop']._genext_run_source == """def jit_run(bh): # field_loop
    pc = bh.position
    i0 = bh.registers_i[0]
    i1 = bh.registers_i[1]
    i2 = bh.registers_i[2]
    r0 = bh.registers_r[0]
    try:
        while 1:
            if pc == 0: # live
                pc = 3
            elif pc == 3: # jit_merge_point
                bh.position = 14
                try:
                    bh.bhimpl_jit_merge_point(0, [], [], [], [i0, i1], [r0], [])
                except LeaveFrame:
                    return -1
            elif pc == 14: # live
                pc = 17
            elif pc == 17: # residual_call_ir_i
                bh.position = 26
                i2 = plain_int(bh.cpu.bh_call_i(glob0, [i0, 7], [], None, glob1))
                pc = 26
                continue
            elif pc == 26: # live
                pc = 29
            elif pc == 29: # goto_if_not_int_eq
                if i2 == 3:
                    pc = 34
                else:
                    pc = 62
                continue
            elif pc == 34: # getfield_gc_i
                return 34
            elif pc == 39: # int_add
                i1 = plain_int(glob2(i1, i2))
                pc = 43
                continue
            elif pc == 43: # int_sub
                i0 = plain_int(glob3(i0, 1))
                pc = 47
                continue
            elif pc == 47: # live
                pc = 50
            elif pc == 50: # goto_if_not_int_gt
                if i0 > 0:
                    pc = 55
                else:
                    pc = 60
                continue
            elif pc == 55: # loop_header
                glob4(0)
                pc = 57
                continue
            elif pc == 57: # goto
                pc = 0
                continue
            elif pc == 60: # int_return
                bh.tmpreg_i = i1
                bh._return_type = 'i'
                return -1
            elif pc == 62: # int_add
                i1 = plain_int(glob5(i1, 1))
                pc = 66
                continue
            elif pc == 66: # goto
                pc = 43
                continue
            else:
                return pc
    finally:
        bh.registers_i[0] = i0
        bh.registers_i[1] = i1
        bh.registers_i[2] = i2"""
