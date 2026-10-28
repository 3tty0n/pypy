import py

from rpython.rlib.jit import JitDriver
from rpython.jit.codewriter.genextrun import RunModeGenerator
from rpython.jit.metainterp import blackhole, jitexc
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp.warmspot import get_stats
from rpython.rtyper.lltypesystem import lltype


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


ARRAY = lltype.GcArray(lltype.Signed)
HOLDER = lltype.GcStruct('RunModeHolder', ('x', lltype.Signed))
arraydriver = JitDriver(greens=[], reds=['n', 'res', 'box', 'items'])


def object_array_loop(n):
    box = lltype.malloc(HOLDER)
    box.x = 0
    items = lltype.malloc(ARRAY, 1)
    items[0] = 0
    res = 0
    while n > 0:
        arraydriver.can_enter_jit(n=n, res=res, box=box, items=items)
        arraydriver.jit_merge_point(n=n, res=res, box=box, items=items)
        if n % 7 == 3:
            box.x = box.x + n
            items[0] = items[0] + box.x
            res += box.x + items[0]
        n -= 1
    return res + box.x + items[0]


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
        fallbacks = []
        generated = {}
        make_jitcodes = CodeWriter.make_jitcodes

        def install(cw, verbose=False):
            jitcodes = make_jitcodes(cw, verbose)
            for jitcode in jitcodes:
                gen = RunModeGenerator(cw.assembler, jitcode._ssarepr, jitcode)
                jit_run = gen.generate_run()

                def counting(bh, jit_run=jit_run):
                    entries.append((bh.jitcode.name, bh.position))
                    pc = jit_run(bh)
                    if pc >= 0:
                        fallbacks.append((bh.jitcode.name, pc))
                    return pc
                jitcode.genext_run_function = counting
                generated[jitcode.name] = jitcode
            return jitcodes

        monkeypatch.setattr(CodeWriter, 'make_jitcodes', install)
        return entries, generated, fallbacks

    def test_e2e_loop_exit_runs_in_run_mode(self, monkeypatch):
        entries, _, _ = self._install_run_mode(monkeypatch)
        assert self.meta_interp(simple_loop, [100]) == simple_loop(100)
        assert entries == [('simple_loop', 27)]

    def test_e2e_guard_failures_run_in_run_mode(self, monkeypatch):
        entries, _, _ = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count == 2
        assert entries == [('guard_loop', 28), ('guard_loop', 28)]

    def test_e2e_getfield_runs_in_run_mode(self, monkeypatch):
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(field_loop)
        assert res == field_loop(30)
        assert count == 2
        assert entries == [('field_loop', 29), ('field_loop', 29)]
        assert not [pc for name, pc in fallbacks if name == 'field_loop']
        source = generated['field_loop']._genext_run_source
        assert 'elif pc == 34: # getfield_gc_i' in source
        assert 'bh.cpu.bh_getfield_gc_i' in source

    def test_e2e_object_and_array_access(self, monkeypatch):
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(object_array_loop)
        assert res == object_array_loop(30)
        assert count == 2

        jitcode = generated['object_array_loop']
        fallback_names = [
            jitcode._ssarepr.insns[
                jitcode._ssarepr._insns_pos.index(pc)][0]
            for name, pc in fallbacks if name == 'object_array_loop'
        ]
        instruction_names = [insn[0] for insn in jitcode._ssarepr.insns]
        assert 'getfield_gc_i' in instruction_names
        assert 'getarrayitem_gc_i' in instruction_names
        assert 'setfield_gc_i' in instruction_names
        assert 'setarrayitem_gc_i' in instruction_names
        assert 'setfield_gc_i' not in fallback_names
        assert 'setarrayitem_gc_i' not in fallback_names
        assert 'getfield_gc_i' not in fallback_names
        assert 'getarrayitem_gc_i' not in fallback_names
        assert any(name == 'object_array_loop' for name, _ in entries)
        source = jitcode._genext_run_source
        assert 'bh.cpu.bh_setfield_gc_i' in source
        assert 'bh.cpu.bh_setarrayitem_gc_i' in source
