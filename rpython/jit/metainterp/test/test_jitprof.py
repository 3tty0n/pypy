
import py
from rpython.jit.metainterp.warmspot import ll_meta_interp
from rpython.rlib.jit import JitDriver, dont_look_inside, elidable, Counters
from rpython.jit.metainterp.test.support import LLJitMixin
from rpython.jit.metainterp import pyjitpl
from rpython.jit.metainterp.jitprof import Profiler

class FakeProfiler(Profiler):
    def start(self):
        self.counter = 123456
        Profiler.start(self)
        self.events = []
        self.times = [0] * (Counters.PE_COGEN_INSTALL + 1)
    
    def timer(self):
        self.counter += 1
        return self.counter - 1

    def _start(self, event):
        Profiler._start(self, event)
        self.events.append(event)

    def _end(self, event):
        Profiler._end(self, event)
        self.events.append(~event)

class ProfilerMixin(LLJitMixin):
    def meta_interp(self, *args, **kwds):
        kwds = kwds.copy()
        kwds['ProfilerClass'] = FakeProfiler
        return LLJitMixin.meta_interp(self, *args, **kwds)

class TestProfile(ProfilerMixin):

    def test_simple_loop(self):
        myjitdriver = JitDriver(greens = [], reds = ['x', 'y', 'res'])
        def f(x, y):
            res = 0
            while y > 0:
                myjitdriver.can_enter_jit(x=x, y=y, res=res)
                myjitdriver.jit_merge_point(x=x, y=y, res=res)
                res += x
                y -= 1
            return res * 2
        res = self.meta_interp(f, [6, 7])
        assert res == 84
        profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        expected = [
            Counters.TRACING,
            Counters.OPTIMIZING,
            ~ Counters.OPTIMIZING,
            Counters.OPTIMIZING,
            ~ Counters.OPTIMIZING,
            Counters.BACKEND,
            ~ Counters.BACKEND,
            Counters.BLACKHOLE,
            Counters.BLACKHOLE_DECODE,
            ~ Counters.BLACKHOLE_DECODE,
            ~ Counters.BLACKHOLE,
            ~ Counters.TRACING,
            ]
        assert profiler.events == expected
        assert profiler.times == [5, 2, 1, 2, 0, 1, 0, 0, 0]
        py.test.skip("disabled until unrolling")
        assert profiler.counters == [1, 1, 3, 3, 2, 15, 2, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0]

    def test_simple_loop_with_call(self):
        @dont_look_inside
        def g(n):
            pass
        
        myjitdriver = JitDriver(greens = [], reds = ['x', 'y', 'res'])
        def f(x, y):
            res = 0
            while y > 0:
                myjitdriver.can_enter_jit(x=x, y=y, res=res)
                myjitdriver.jit_merge_point(x=x, y=y, res=res)
                res += x
                g(x)
                y -= 1
            return res * 2
        res = self.meta_interp(f, [6, 7])
        assert res == 84
        profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        assert profiler.calls == 1

    def test_blackhole_pure(self):
        @elidable
        def g(n):
            return n+1
        
        myjitdriver = JitDriver(greens = ['z'], reds = ['y', 'x','res'])
        def f(x, y, z):
            res = 0
            while y > 0:
                myjitdriver.can_enter_jit(x=x, y=y, res=res, z=z)
                myjitdriver.jit_merge_point(x=x, y=y, res=res, z=z)
                res += x
                res += g(z)
                y -= 1
            return res * 2
        res = self.meta_interp(f, [6, 7, 2])
        assert res == f(6, 7, 2)
        profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        assert profiler.calls == 1

    def test_heapcache_stats(self):
        class A:
            pass
        class B(A):
            pass
        @dont_look_inside
        def extern(n):
            if n == -7:
                return None
            elif n:
                return A()
            else:
                return B()
        myjitdriver = JitDriver(greens = [], reds='auto')
        def f(x, y):
            res = 0
            while y > 0:
                myjitdriver.jit_merge_point()
                obj = extern(y)
                res += x + isinstance(obj, B) + isinstance(obj, B) + isinstance(obj, B) + isinstance(obj, B)
                res += x
                y -= 1
            return res * 2
        res = self.meta_interp(f, [6, 7])
        assert res == f(6, 7)
        profiler = pyjitpl._warmrunnerdesc.metainterp_sd.profiler
        assert profiler.counters[Counters.HEAPCACHED_OPS] == 3


def test_blackhole_cost_model_fit():
    from rpython.jit.metainterp.jitprof import Profiler
    from rpython.rlib.jit import Counters
    p = Profiler()
    clock = [0.0]
    p.timer = lambda: clock[0]
    p.start()
    # each failure: 1us blackhole + 1us stretch + 30ns per tail op
    for tail in range(100, 100 + 16 * 50, 50):
        p.start_blackhole()
        clock[0] += 1e-6
        p.end_blackhole(7, tail)
        clock[0] += 1e-6 + 30e-9 * tail   # interpreter stretch to compiled
        p.end_fail_stretch()
    c, b = p.blackhole_cost_model()
    assert abs(c - 2e-6) < 1e-9
    assert abs(b - 30e-9) < 1e-12
    # compile cost: 1us per recorded op, 2 recorded ops per opt op
    p.counters[Counters.RECORDED_OPS] = 2000
    p.counters[Counters.OPT_OPS] = 1000
    p.times[Counters.TRACING] = 2000 * 1e-6
    short = p.bridge_break_even(10)      # 20 rec ops: 20us / (0.3us + 2us)
    long = p.bridge_break_even(1000)     # 2000 rec ops: 2ms / (30us + 2us)
    assert abs(short - 20.0 / 2.3) < 1e-6
    assert abs(long - 2000.0 / 32.0) < 1e-6
    assert short < long


def test_blackhole_cost_model_needs_samples():
    from rpython.jit.metainterp.jitprof import Profiler
    p = Profiler()
    p.timer = lambda: 0.0
    p.start()
    assert p.blackhole_cost_model() == (-1.0, -1.0)
    assert p.bridge_break_even(10) == -1.0


def test_failure_histograms():
    from rpython.jit.metainterp.jitprof import Profiler
    prof = Profiler()
    prof.start()
    for count in range(1, 10):
        prof.note_guard_failure(count)
    prof.note_bridge_at(9)
    prof.note_bridge_at(1)
    assert prof.fail_hist[:5] == [1, 1, 1, 1, 0]
    assert prof.bridge_hist[:4] == [1, 0, 0, 1]
    assert prof._hist(prof.fail_hist) == "1 1 1 1"


def test_survivor_rule():
    from rpython.jit.metainterp.jitprof import Profiler
    from rpython.rlib.jit import Counters
    p = Profiler()
    clock = [0.0]
    p.timer = lambda: clock[0]
    p.start()
    p.counters[Counters.RECORDED_OPS] = 1000
    p.counters[Counters.OPT_OPS] = 1000
    # failure: 1us fixed + 10ns per tail op; bridge: 100us + 1us per op
    for insns in range(100, 100 + 16 * 50, 50):
        p.start_blackhole()
        clock[0] += 1e-6 + 10e-9 * insns
        p.end_blackhole(3, insns)
        p.end_fail_stretch()
        p.start_bridge_attempt()
        p.counters[Counters.RECORDED_OPS] += insns
        clock[0] += 100e-6 + 1e-6 * insns
        p.end_bridge_attempt()
    p.counters[Counters.RECORDED_OPS] = 1000
    a, b = p.bridge_fit.fit()
    assert abs(a - 100e-6) < 1e-9 and abs(b - 1e-6) < 1e-9
    assert abs(p.bridge_cost(10) - 110e-6) < 1e-9
    assert abs(p.failure_cost(10) - 1.1e-6) < 1e-12
    assert p.failures_until(1, 200) == (0.0, 0.0)
    # 8 guards reach 1 failure, 4 reach 2, 2 reach 4, 1 reaches 8
    p.fail_hist[0:4] = [8, 4, 2, 1]
    # from 1 failure up to a horizon of 8: 4*1 + 2*2 + 1*4 over 8 guards,
    # and 1 of the 8 reaches the horizon
    assert p.failures_until(1, 8) == (12.0 / 8, 1.0 / 8)
    # horizon 12 sits in bucket 3: its one guard fails 4 more times there
    assert p.failures_until(1, 12) == (16.0 / 8, 1.0 / 8)
    assert p.failures_until(8, 8) == (0.0, 0.0)
    assert not p.bridge_pays_off(1, 10, 8)      # 1 * 1.1us < 110us * 7/8
    p.fail_hist[0:11] = [8] * 11                 # every guard reaches 1024
    assert p.failures_until(1, 1024) == (1023.0, 1.0)
    assert p.bridge_pays_off(1, 10, 1024)       # nothing risked, much saved
    assert not p.bridge_pays_off(1, 10, 1)      # already past the horizon


def test_linear_fit_flat_falls_back_to_mean():
    from rpython.jit.metainterp.jitprof import LinearFit
    f = LinearFit()
    for x in range(16):
        f.add(float(x), 2e-6)
    c, b = f.fit()
    assert abs(c - 2e-6) < 1e-12 and b == 0.0
