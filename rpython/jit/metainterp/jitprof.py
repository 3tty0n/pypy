
""" A small helper module for profiling JIT
"""

import time
from rpython.rlib.debug import debug_print, debug_start, debug_stop
from rpython.rlib.debug import have_debug_prints
from rpython.jit.metainterp.jitexc import JitException
from rpython.rlib.jit import Counters


JITPROF_LINES = Counters.ncounters + 10
# TOTAL, calls, two PE instruction counts, three PE cogen outcomes
_CPU_LINES = 4       # the last 4 lines are stored on the cpu

class BaseProfiler(object):
    pass

class EmptyProfiler(BaseProfiler):
    initialized = True

    def start(self):
        pass

    def finish(self):
        pass

    def start_tracing(self):
        pass

    def end_tracing(self):
        pass

    def start_backend(self):
        pass

    def start_blackhole(self):
        pass

    def end_blackhole(self, insns):
        pass

    def start_portal_call(self, insns):
        pass

    def end_portal_call(self, insns):
        pass

    def start_bridge_attempt(self):
        pass

    def end_bridge_attempt(self):
        pass

    def bridge_break_even(self, tail_ops):
        return -1.0

    def end_backend(self):
        pass

    def start_optimizing(self):
        pass

    def end_optimizing(self):
        pass

    def start_pe_cogen(self):
        pass

    def end_pe_cogen(self):
        pass

    def start_pe_cogen_scan(self):
        pass

    def end_pe_cogen_scan(self):
        pass

    def start_pe_cogen_install(self):
        pass

    def end_pe_cogen_install(self):
        pass

    def count(self, kind, inc=1):
        pass

    def count_ops(self, opnum, kind=Counters.OPS):
        pass

    def get_counter(self, num):
        return 0

    def get_times(self, num):
        return 0.0

class Profiler(BaseProfiler):
    initialized = False
    timer = staticmethod(time.time)
    starttime = 0
    t1 = 0
    times = None
    counters = None
    calls = 0
    current = None
    cpu = None

    # Least-squares fit of blackhole time per resume = C + B * insns;
    # see bridge_break_even().
    bh_n = 0
    bh_sum_m = 0.0
    bh_sum_mm = 0.0
    bh_sum_t = 0.0
    bh_sum_mt = 0.0
    BH_MIN_SAMPLES = 16
    # Per active resume (innermost last): time and insns spent in portal
    # calls made from the blackhole, excluded from that resume's sample.
    bh_excl_time = None
    bh_excl_insns = None
    bh_call_t0 = 0.0
    bh_call_insns0 = 0
    # Wall time and recorded ops of tracing-from-a-guard (bridge attempts,
    # including their optimizing and backend work): the compile cost the
    # bridge model charges, kept apart from loop tracing.
    bridge_time = 0.0
    bridge_rec_ops = 0
    bridge_t0 = 0.0
    bridge_rec_ops0 = 0

    def start(self):
        self.starttime = self.timer()
        self.t1 = self.starttime
        self.times = [0] * (Counters.PE_COGEN_INSTALL + 1)
        self.counters = [0] * (Counters.ncounters - _CPU_LINES)
        self.calls = 0
        self.current = []

    def finish(self):
        self.tk = self.timer()
        self.print_stats()

    def _start(self, event):
        t0 = self.t1
        self.t1 = self.timer()
        if self.current:
            self.times[self.current[-1]] += self.t1 - t0
        self.counters[event] += 1
        self.current.append(event)

    def _end(self, event):
        t0 = self.t1
        self.t1 = self.timer()
        if not self.current:
            debug_print("BROKEN PROFILER DATA!")
            return
        ev1 = self.current.pop()
        if ev1 != event:
            debug_print("BROKEN PROFILER DATA!")
            return
        self.times[ev1] += self.t1 - t0

    def start_tracing(self):   self._start(Counters.TRACING)
    def end_tracing(self):     self._end  (Counters.TRACING)

    def start_optimizing(self): self._start(Counters.OPTIMIZING)
    def end_optimizing(self):   self._end  (Counters.OPTIMIZING)

    def start_backend(self):   self._start(Counters.BACKEND)
    def end_backend(self):     self._end  (Counters.BACKEND)

    def start_blackhole(self):
        self._start(Counters.BLACKHOLE)
        if self.bh_excl_time is None:
            self.bh_excl_time = []
            self.bh_excl_insns = []
        self.bh_excl_time.append(0.0)
        self.bh_excl_insns.append(0)

    def end_blackhole(self, insns):
        t0 = self.t1
        self._end(Counters.BLACKHOLE)
        elapsed = self.t1 - t0 - self.bh_excl_time.pop()
        insns -= self.bh_excl_insns.pop()
        if elapsed <= 0.0 or insns < 0:
            return
        m = float(insns)
        self.bh_n += 1
        self.bh_sum_m += m
        self.bh_sum_mm += m * m
        self.bh_sum_t += elapsed
        self.bh_sum_mt += m * elapsed

    def start_portal_call(self, insns):
        # A callee frame run from the blackhole: real execution, not
        # blackholing; keep it out of the enclosing resume's sample.
        self._start(Counters.BLACKHOLE_CALL)
        self.bh_call_t0 = self.t1
        self.bh_call_insns0 = insns

    def end_portal_call(self, insns):
        t0 = self.bh_call_t0
        insns0 = self.bh_call_insns0
        self._end(Counters.BLACKHOLE_CALL)
        if self.bh_excl_time:
            self.bh_excl_time[-1] += self.t1 - t0
            self.bh_excl_insns[-1] += insns - insns0

    def start_bridge_attempt(self):
        self.bridge_t0 = self.timer()
        self.bridge_rec_ops0 = self.counters[Counters.RECORDED_OPS]

    def end_bridge_attempt(self):
        self.bridge_time += self.timer() - self.bridge_t0
        self.bridge_rec_ops += (self.counters[Counters.RECORDED_OPS] -
                                self.bridge_rec_ops0)

    def blackhole_cost_model(self):
        """(C, B): seconds per resume and per blackholed insn; (-1, -1)
        until enough resumes were seen to fit them."""
        n = self.bh_n
        if n < self.BH_MIN_SAMPLES:
            return -1.0, -1.0
        denom = n * self.bh_sum_mm - self.bh_sum_m * self.bh_sum_m
        if denom <= 0.0:
            return -1.0, -1.0
        b = (n * self.bh_sum_mt - self.bh_sum_m * self.bh_sum_t) / denom
        c = (self.bh_sum_t - b * self.bh_sum_m) / n
        if b <= 0.0 or c <= 0.0:
            return -1.0, -1.0
        return c, b

    def bridge_break_even(self, tail_ops):
        """Guard failures after which compiling the bridge from a guard
        with 'tail_ops' optimized ops left to the end of its trace has
        cost as much as blackholing that tail on every failure.

        A failure without a bridge costs C + B * tail (resume/re-entry
        plus blackholing to the trace end); the bridge costs T * tail to
        trace and compile.  Short tails are dominated by C, so their
        bridges pay back after a few failures; long tails approach T/B.
        Returns -1.0 until the blackhole model is fitted.
        """
        c, b = self.blackhole_cost_model()
        if b < 0.0:
            return -1.0
        opt_ops = self.counters[Counters.OPT_OPS]
        rec_ops = self.counters[Counters.RECORDED_OPS]
        if opt_ops <= 0 or rec_ops <= 0:
            return -1.0
        # Per recorded op, from bridge attempts only once a few exist;
        # loop tracing (unrolling, two optimizer passes) costs more per op.
        if self.bridge_rec_ops >= 1000:
            t = self.bridge_time / self.bridge_rec_ops
        else:
            compile_time = (self.times[Counters.TRACING] +
                            self.times[Counters.OPTIMIZING] +
                            self.times[Counters.BACKEND])
            t = compile_time / rec_ops
        # Blackhole insns and recorded ops share a unit (jitcode insns);
        # optimized ops are fewer by the measured ratio.
        tail_rec = tail_ops * (float(rec_ops) / opt_ops)
        return (t * tail_rec) / (b * tail_rec + c)

    def start_pe_cogen(self):  self._start(Counters.PE_COGEN)
    def end_pe_cogen(self):    self._end  (Counters.PE_COGEN)

    def start_pe_cogen_scan(self): self._start(Counters.PE_COGEN_SCAN)
    def end_pe_cogen_scan(self):   self._end  (Counters.PE_COGEN_SCAN)

    def start_pe_cogen_install(self): self._start(Counters.PE_COGEN_INSTALL)
    def end_pe_cogen_install(self):   self._end  (Counters.PE_COGEN_INSTALL)

    def count(self, kind, inc=1):
        self.counters[kind] += inc

    def get_counter(self, num):
        if num == Counters.TOTAL_COMPILED_LOOPS:
            return self.cpu.tracker.total_compiled_loops
        elif num == Counters.TOTAL_COMPILED_BRIDGES:
            return self.cpu.tracker.total_compiled_bridges
        elif num == Counters.TOTAL_FREED_LOOPS:
            return self.cpu.tracker.total_freed_loops
        elif num == Counters.TOTAL_FREED_BRIDGES:
            return self.cpu.tracker.total_freed_bridges
        return self.counters[num]

    def get_times(self, num):
        return self.times[num]

    def count_ops(self, opnum, kind=Counters.OPS):
        from rpython.jit.metainterp.resoperation import OpHelpers
        self.counters[kind] += 1
        if OpHelpers.is_call(opnum) and kind == Counters.RECORDED_OPS:
            self.calls += 1

    def print_stats(self):
        debug_start("jit-summary")
        if have_debug_prints():
            self._print_stats()
        debug_stop("jit-summary")

    def _print_stats(self):
        cnt = self.counters
        tim = self.times
        calls = self.calls
        self._print_line_time("Tracing", cnt[Counters.TRACING],
                              tim[Counters.TRACING])
        self._print_line_time("Optimizing", cnt[Counters.OPTIMIZING],
                              tim[Counters.OPTIMIZING])
        self._print_line_time("Backend", cnt[Counters.BACKEND],
                              tim[Counters.BACKEND])
        self._print_line_time("Blackhole", cnt[Counters.BLACKHOLE],
                              tim[Counters.BLACKHOLE])
        self._print_line_time("Blackhole callee", cnt[Counters.BLACKHOLE_CALL],
                              tim[Counters.BLACKHOLE_CALL])
        c, b = self.blackhole_cost_model()
        debug_print("bridge model:\tC=%f us\tB=%f ns\tbreak-even(100)=%f" % (
            c * 1e6, b * 1e9, self.bridge_break_even(100)))
        t_bridge = 0.0
        if self.bridge_rec_ops:
            t_bridge = self.bridge_time / self.bridge_rec_ops * 1e6
        debug_print("bridge attempts:\t%f s\t%d rec ops\t%f us/op" % (
            self.bridge_time, self.bridge_rec_ops, t_bridge))
        self._print_line_time("PE cogen overhead", cnt[Counters.PE_COGEN],
                              tim[Counters.PE_COGEN])
        self._print_line_time("PE cogen scan", cnt[Counters.PE_COGEN_SCAN],
                              tim[Counters.PE_COGEN_SCAN])
        self._print_line_time("PE cogen install",
                              cnt[Counters.PE_COGEN_INSTALL],
                              tim[Counters.PE_COGEN_INSTALL])
        line = "TOTAL:      \t\t%f" % (self.tk - self.starttime, )
        debug_print(line)
        from rpython.jit.metainterp.pyjitpl import _pe_insn_counts
        from rpython.jit.codewriter.jitcode import _cogen_counters
        self._print_intline("pe cogen generated", _cogen_counters.generated)
        self._print_intline("pe cogen declined", _cogen_counters.declined)
        self._print_intline("pe cogen deferred", _cogen_counters.deferred)
        self._print_intline("pe insns generic", _pe_insn_counts.generic)
        self._print_intline("pe insns portal", _pe_insn_counts.portal)
        self._print_intline("pe insns residual", _pe_insn_counts.residual)
        self._print_intline("ops", cnt[Counters.OPS])
        self._print_intline("heapcached ops", cnt[Counters.HEAPCACHED_OPS])
        self._print_intline("recorded ops", cnt[Counters.RECORDED_OPS])
        self._print_intline("  calls", calls)
        self._print_intline("guards", cnt[Counters.GUARDS])
        self._print_intline("opt ops", cnt[Counters.OPT_OPS])
        self._print_intline("opt guards", cnt[Counters.OPT_GUARDS])
        self._print_intline("opt guards shared", cnt[Counters.OPT_GUARDS_SHARED])
        self._print_intline("forcings", cnt[Counters.OPT_FORCINGS])
        self._print_intline("abort: trace too long",
                            cnt[Counters.ABORT_TOO_LONG])
        self._print_intline("abort: compiling", cnt[Counters.ABORT_BRIDGE])
        self._print_intline("abort: vable escape", cnt[Counters.ABORT_ESCAPE])
        self._print_intline("abort: bad loop", cnt[Counters.ABORT_BAD_LOOP])
        self._print_intline("abort: force quasi-immut",
                            cnt[Counters.ABORT_FORCE_QUASIIMMUT])
        self._print_intline("abort: segmenting trace",
                            cnt[Counters.ABORT_SEGMENTED_TRACE])
        self._print_intline("virtualizables forced",
                            cnt[Counters.FORCE_VIRTUALIZABLES])
        self._print_intline("nvirtuals", cnt[Counters.NVIRTUALS])
        self._print_intline("nvholes", cnt[Counters.NVHOLES])
        self._print_intline("nvreused", cnt[Counters.NVREUSED])
        self._print_intline("vecopt tried", cnt[Counters.OPT_VECTORIZE_TRY])
        self._print_intline("vecopt success", cnt[Counters.OPT_VECTORIZED])
        cpu = self.cpu
        if cpu is not None:   # for some tests
            self._print_intline("Total # of loops",
                                cpu.tracker.total_compiled_loops)
            self._print_intline("Total # of bridges",
                                cpu.tracker.total_compiled_bridges)
            self._print_intline("Freed # of loops",
                                cpu.tracker.total_freed_loops)
            self._print_intline("Freed # of bridges",
                                cpu.tracker.total_freed_bridges)

    def _print_line_time(self, string, i, tim):
        final = "%s:%s\t%d\t%f" % (string, " " * max(0, 13-len(string)), i, tim)
        debug_print(final)

    def _print_intline(self, string, i):
        final = string + ':' + " " * max(0, 16-len(string))
        final += '\t' + str(i)
        debug_print(final)


class BrokenProfilerData(JitException):
    pass
