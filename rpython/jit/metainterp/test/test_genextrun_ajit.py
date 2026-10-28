import re

import py

from rpython.rlib import jit
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


class RunModeTestError(Exception):
    def __init__(self, value):
        self.value = value


class RunModeOtherTestError(Exception):
    pass


@jit.dont_look_inside
def maybe_raise(n):
    if n % 5 == 0:
        raise RunModeTestError(n)
    if n % 7 == 0:
        raise RunModeOtherTestError()
    return n


exceptiondriver = JitDriver(greens=[], reds=['n', 'res'])


def exception_loop(n):
    # no bridges: the llgraph backend cannot compile one that starts at a
    # guard_exception, so failing guards resume in blackhole instead
    jit.set_param(exceptiondriver, 'trace_eagerness', 1000)
    res = 0
    while n > 0:
        exceptiondriver.can_enter_jit(n=n, res=res)
        exceptiondriver.jit_merge_point(n=n, res=res)
        try:
            res += maybe_raise(n)
        except RunModeTestError as e:
            res += 100 + e.value
        except RunModeOtherTestError:
            res += 1000
        n -= 1
    return res


switchdriver = JitDriver(greens=[], reds=['n', 'res'])


def switch_loop(n):
    res = 0
    while n > 0:
        switchdriver.can_enter_jit(n=n, res=res)
        switchdriver.jit_merge_point(n=n, res=res)
        k = n % 6
        if k == 0:
            res += 1
        elif k == 1:
            res += 20
        elif k == 2:
            res += 300
        elif k == 3:
            res += 4000
        elif k == 4:
            res += 50000
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

    def _run_guard_loop(self, func=guard_loop, **kwds):
        counter = []
        orig = blackhole.resume_in_blackhole

        def counting(*args, **kwds):
            counter.append(1)
            return orig(*args, **kwds)

        blackhole.resume_in_blackhole = counting
        try:
            res = self.meta_interp(func, [30], **kwds)
        finally:
            blackhole.resume_in_blackhole = orig
        return res, len(counter)

    def test_guard_failure_lands_in_run_mode(self):
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count == 2

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

    def _baseline_trace(self, func, args, **kwds):
        res = self.meta_interp(func, args, **kwds)
        assert res == func(*args)
        baseline = self.get_trace_signature()
        assert baseline, 'the baseline run did not produce a trace'
        return baseline

    def _check_trace_matches(self, baseline):
        actual = self.get_trace_signature()
        assert actual, 'the run-mode run did not produce a trace'
        assert actual == baseline, (
            'the run-mode execution changed the traced operations')

    def _generated_methods(self, jitcode):
        source = jitcode._genext_run_source
        return (re.findall(r'bh\.cpu\.(\w+)', source),
                re.findall(r'bh\.(bhimpl_\w+)', source))

    def _generated_lines(self, jitcode):
        return [line.strip()
                for line in jitcode._genext_run_source.splitlines()]

    def test_e2e_loop_exit_runs_in_run_mode(self, monkeypatch):
        baseline = self._baseline_trace(simple_loop, [100])
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        assert self.meta_interp(simple_loop, [100]) == simple_loop(100)
        self._check_trace_matches(baseline)
        assert entries == [('simple_loop', 27)]
        assert fallbacks == []
        assert self._generated_methods(generated['simple_loop']) == (
            [], ['bhimpl_jit_merge_point'])

    def test_e2e_guard_failures_run_in_run_mode(self, monkeypatch):
        baseline = self._baseline_trace(guard_loop, [30])
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop()
        assert res == guard_loop(30)
        assert count == 2
        self._check_trace_matches(baseline)
        assert entries == [('guard_loop', 28), ('guard_loop', 28)]
        assert fallbacks == []
        assert self._generated_methods(generated['guard_loop']) == (
            ['bh_call_i'], ['bhimpl_jit_merge_point'])

    def test_e2e_getfield_runs_in_run_mode(self, monkeypatch):
        baseline = self._baseline_trace(field_loop, [30])
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(field_loop)
        assert res == field_loop(30)
        assert count == 2
        self._check_trace_matches(baseline)
        assert entries == [('field_loop', 29), ('field_loop', 29)]
        assert fallbacks == []
        jitcode = generated['field_loop']
        assert self._generated_methods(jitcode) == (
            ['bh_call_i', 'bh_getfield_gc_i'], ['bhimpl_jit_merge_point'])
        lines = self._generated_lines(jitcode)
        assert 'elif pc == 34: # getfield_gc_i' in lines
        assert 'i2 = plain_int(bh.cpu.bh_getfield_gc_i(r0, glob2))' in lines

    def test_e2e_object_and_array_access(self, monkeypatch):
        baseline = self._baseline_trace(object_array_loop, [30])
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(object_array_loop)
        assert res == object_array_loop(30)
        assert count == 2
        self._check_trace_matches(baseline)
        assert entries == [('object_array_loop', 30),
                           ('object_array_loop', 30)]
        assert fallbacks == []

        jitcode = generated['object_array_loop']
        assert self._generated_methods(jitcode) == (
            ['bh_call_i', 'bh_getfield_gc_i', 'bh_setfield_gc_i',
             'bh_getarrayitem_gc_i', 'bh_getfield_gc_i',
             'bh_setarrayitem_gc_i', 'bh_getfield_gc_i',
             'bh_getarrayitem_gc_i', 'bh_getfield_gc_i',
             'bh_getarrayitem_gc_i'],
            ['bhimpl_jit_merge_point'])

    def test_e2e_exceptions(self, monkeypatch):
        baseline = self._baseline_trace(exception_loop, [30])
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        res, count = self._run_guard_loop(exception_loop)
        assert res == exception_loop(30)
        assert count == 7
        self._check_trace_matches(baseline)
        assert entries == [('exception_loop', 53)] * 6 + [
            ('exception_loop', 24)]
        assert fallbacks == []

        jitcode = generated['exception_loop']
        assert self._generated_methods(jitcode) == (
            ['bh_call_i', 'bh_getfield_gc_i'],
            ['bhimpl_jit_merge_point', 'bhimpl_goto_if_exception_mismatch',
             'bhimpl_last_exc_value',
             'bhimpl_goto_if_exception_mismatch', 'bhimpl_reraise'])
        lines = self._generated_lines(jitcode)
        assert [line for line in lines
                if 'bhimpl_goto_if_exception_mismatch' in line] == [
            'pc = bh.bhimpl_goto_if_exception_mismatch(glob5, 75, 57)',
            'pc = bh.bhimpl_goto_if_exception_mismatch(glob9, 86, 79)']

    def test_e2e_switch(self, monkeypatch):
        baseline = self._baseline_trace(switch_loop, [30], backendopt=True)
        entries, generated, fallbacks = self._install_run_mode(monkeypatch)
        # backendopt turns the if/elif chain into an int switch
        res, count = self._run_guard_loop(switch_loop, backendopt=True)
        assert res == switch_loop(30)
        assert count == 10
        self._check_trace_matches(baseline)
        assert entries == [('switch_loop', 28)] * 10
        assert fallbacks == []

        jitcode = generated['switch_loop']
        assert self._generated_methods(jitcode) == (
            ['bh_call_i'], ['bhimpl_jit_merge_point'])
        assert 'pc = glob2.get(i2, 32)' in self._generated_lines(jitcode)
