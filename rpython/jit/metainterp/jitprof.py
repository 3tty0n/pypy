
""" A small helper module for profiling JIT
"""

import time
from rpython.rlib.debug import debug_print, debug_start, debug_stop
from rpython.rlib.debug import have_debug_prints
from rpython.jit.metainterp.jitexc import JitException
from rpython.rlib.jit import Counters


JITPROF_LINES = Counters.ncounters + 13  # TOTAL, calls, PE/bridge stats
_CPU_LINES = 4       # the last 4 lines are stored on the cpu

class BaseProfiler(object):
    pass

class EmptyProfiler(BaseProfiler):
    initialized = True
    last_bridge_time = 0.0
    last_bridge_rec_ops = 0

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

    def end_blackhole(self, insns, tail_ops):
        pass

    def start_decode(self):
        pass

    def end_decode(self):
        pass

    def note_guard_failure(self, count):
        pass

    def note_bridge_at(self, count):
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

    def bridge_pays_off(self, count, tail_ops, horizon):
        return False

    def end_fail_stretch(self):
        pass

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

class LinearFit(object):
    MIN_SAMPLES = 16

    def __init__(self):
        self.n = 0
        self.sx = self.sxx = self.st = self.sxt = 0.0

    def add(self, x, t):
        self.n += 1
        self.sx += x
        self.sxx += x * x
        self.st += t
        self.sxt += x * t

    def fit(self):
        """(c, b) of t = c + b * x, or (-1, -1) until fitted."""
        n = self.n
        if n < self.MIN_SAMPLES:
            return -1.0, -1.0
        denom = n * self.sxx - self.sx * self.sx
        if denom <= 0.0:
            return -1.0, -1.0
        b = (n * self.sxt - self.sx * self.st) / denom
        c = (self.st - b * self.sx) / n
        if b <= 0.0 or c <= 0.0:
            # No usable slope: cost is flat, use the mean.
            return self.st / n, 0.0
        return c, b


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

    # Least-squares fits: guard failure cost = C + B * tail ops and bridge
    # time per attempt = a + b * rec ops; see bridge_break_even().
    bh_fit = None
    bridge_fit = None
    # A guard failure costs the blackhole run plus the interpreter stretch
    # until compiled code is entered again; the sample closes there.
    fail_pending = False
    fail_elapsed = 0.0
    fail_tail_ops = 0.0
    fail_t_end = 0.0
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
    last_bridge_time = 0.0
    last_bridge_rec_ops = 0
    # Survivor histogram: fail_hist[k] = guards that failed >= 2**k times;
    # bridge_hist[k] = bridges compiled from a guard at 2**k <= n < 2**k+1.
    HIST_BUCKETS = 32

    def start(self):
        self.starttime = self.timer()
        self.t1 = self.starttime
        self.times = [0] * (Counters.PE_COGEN_INSTALL + 1)
        self.counters = [0] * (Counters.ncounters - _CPU_LINES)
        self.calls = 0
        self.fail_hist = [0] * self.HIST_BUCKETS
        self.bridge_hist = [0] * self.HIST_BUCKETS
        self.bh_fit = LinearFit()
        self.bridge_fit = LinearFit()
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

    def end_blackhole(self, insns, tail_ops):
        # tail_ops: optimized ops after the failed guard (-1: unknown);
        # the interpreter stretch that follows scales with it.
        t0 = self.t1
        self._end(Counters.BLACKHOLE)
        elapsed = self.t1 - t0 - self.bh_excl_time.pop()
        self.bh_excl_insns.pop()
        if elapsed <= 0.0 or tail_ops < 0:
            return
        self.end_fail_stretch()
        self.fail_pending = True
        self.fail_elapsed = elapsed
        self.fail_tail_ops = float(tail_ops)
        self.fail_t_end = self.t1

    def end_fail_stretch(self):
        if not self.fail_pending:
            return
        self.fail_pending = False
        stretch = self.timer() - self.fail_t_end
        self.bh_fit.add(self.fail_tail_ops, self.fail_elapsed + stretch)

    def start_decode(self):
        self._start(Counters.BLACKHOLE_DECODE)

    def end_decode(self):
        self._end(Counters.BLACKHOLE_DECODE)

    @staticmethod
    def _bucket(count):
        k = 0
        while count > 1 and k < Profiler.HIST_BUCKETS - 1:
            count >>= 1
            k += 1
        return k

    def note_guard_failure(self, count):
        if count & (count - 1) == 0:
            self.fail_hist[self._bucket(count)] += 1

    def note_bridge_at(self, count):
        self.bridge_hist[self._bucket(count)] += 1

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
        self.last_bridge_time = self.timer() - self.bridge_t0
        self.last_bridge_rec_ops = (self.counters[Counters.RECORDED_OPS] -
                                    self.bridge_rec_ops0)
        self.bridge_time += self.last_bridge_time
        self.bridge_rec_ops += self.last_bridge_rec_ops
        self.bridge_fit.add(float(self.last_bridge_rec_ops),
                            self.last_bridge_time)

    def blackhole_cost_model(self):
        """(C, B): seconds per guard failure and per optimized op of its
        tail, measured until compiled code is entered again; (-1, -1)
        until enough failures were seen to fit them."""
        return self.bh_fit.fit()

    def _tail_rec_ops(self, tail_ops):
        # Optimized ops are fewer than recorded ones by the measured ratio.
        opt_ops = self.counters[Counters.OPT_OPS]
        rec_ops = self.counters[Counters.RECORDED_OPS]
        if opt_ops <= 0 or rec_ops <= 0:
            return -1.0
        return tail_ops * (float(rec_ops) / opt_ops)

    def failure_cost(self, tail_ops):
        """Seconds one guard failure costs without a bridge, or -1."""
        c, b = self.blackhole_cost_model()
        if b < 0.0:
            return -1.0
        return c + b * tail_ops

    def bridge_cost(self, tail_ops):
        """Seconds to trace and compile a bridge over tail_ops, or -1."""
        tail_rec = self._tail_rec_ops(tail_ops)
        if tail_rec < 0.0:
            return -1.0
        a, t = self.bridge_fit.fit()
        if a > 0.0:
            return a + t * tail_rec
        # Before enough bridges exist: per op from loop compiles, no fixed
        # part (unrolling and two optimizer passes cost more per op).
        rec_ops = self.counters[Counters.RECORDED_OPS]
        compile_time = (self.times[Counters.TRACING] +
                        self.times[Counters.OPTIMIZING] +
                        self.times[Counters.BACKEND])
        return compile_time / rec_ops * tail_rec

    def bridge_break_even(self, tail_ops):
        """Guard failures at which a bridge over 'tail_ops' optimized ops
        pays back its trace+compile cost vs. blackholing on each failure.
        Returns -1.0 until the blackhole model is fitted."""
        fail = self.failure_cost(tail_ops)
        if fail < 0.0:
            return -1.0
        return self.bridge_cost(tail_ops) / fail

    def failures_until(self, count, horizon):
        """(saved, reach): failures a guard at 'count' is expected to
        make before 'horizon', and its probability of getting there.
        Only the observed part of the histogram is used: no extrapolation
        past the base eagerness, so over-bridging feeds back negatively."""
        k = self._bucket(count)
        kh = self._bucket(horizon)
        at_k = self.fail_hist[k]
        if kh <= k or at_k == 0:
            return 0.0, 0.0
        saved = 0.0
        for j in range(k, kh):
            saved += self.fail_hist[j + 1] * float(1 << j)
        # Guards reaching the horizon's bucket fail on until the horizon.
        saved += self.fail_hist[kh] * float(horizon - (1 << kh))
        return saved / at_k, self.fail_hist[kh] / float(at_k)

    def bridge_pays_off(self, count, tail_ops, horizon):
        """Survivor rule: bridge now rather than at 'horizon' failures when
        the failures saved outweigh the bridge cost risked on a guard that
        would have died before the horizon."""
        saved, reach = self.failures_until(count, horizon)
        fail = self.failure_cost(tail_ops)
        if saved <= 0.0 or fail <= 0.0:
            return False
        return saved * fail > self.bridge_cost(tail_ops) * (1.0 - reach)

    # --- PE runtime-cogen timers (the rest of this file is generic) ---
    def start_pe_cogen(self):  self._start(Counters.PE_COGEN)
    def end_pe_cogen(self):    self._end  (Counters.PE_COGEN)

    def start_pe_cogen_scan(self): self._start(Counters.PE_COGEN_SCAN)
    def end_pe_cogen_scan(self):   self._end  (Counters.PE_COGEN_SCAN)

    def start_pe_cogen_install(self): self._start(Counters.PE_COGEN_INSTALL)
    def end_pe_cogen_install(self):   self._end  (Counters.PE_COGEN_INSTALL)
    # --- end PE runtime-cogen timers ---

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

    # --- PE runtime-cogen stats (the rest of this file is generic) ---
    def _print_pe_stats(self, cnt, tim):
        from rpython.jit.metainterp.pyjitpl import _pe_insn_counts
        from rpython.jit.codewriter.jitcode import _cogen_counters
        self._print_line_time("PE cogen overhead", cnt[Counters.PE_COGEN],
                              tim[Counters.PE_COGEN])
        self._print_line_time("PE cogen scan", cnt[Counters.PE_COGEN_SCAN],
                              tim[Counters.PE_COGEN_SCAN])
        self._print_line_time("PE cogen install",
                              cnt[Counters.PE_COGEN_INSTALL],
                              tim[Counters.PE_COGEN_INSTALL])
        self._print_intline("pe cogen generated", _cogen_counters.generated)
        self._print_intline("pe cogen declined", _cogen_counters.declined)
        self._print_intline("pe cogen deferred", _cogen_counters.deferred)
        self._print_intline("pe insns generic", _pe_insn_counts.generic)
        self._print_intline("pe insns portal", _pe_insn_counts.portal)
        self._print_intline("pe insns residual", _pe_insn_counts.residual)
    # --- end PE runtime-cogen stats ---

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
        self._print_line_time("Blackhole decode",
                              cnt[Counters.BLACKHOLE_DECODE],
                              tim[Counters.BLACKHOLE_DECODE])
        debug_print("guard failures >=2^k:\t" + self._hist(self.fail_hist))
        debug_print("bridges at 2^k:\t" + self._hist(self.bridge_hist))
        c, b = self.blackhole_cost_model()
        debug_print("bridge model:\tC=%f us\tB=%f ns\tbreak-even(100)=%f" % (
            c * 1e6, b * 1e9, self.bridge_break_even(100)))
        t_bridge = 0.0
        if self.bridge_rec_ops:
            t_bridge = self.bridge_time / self.bridge_rec_ops * 1e6
        debug_print("bridge attempts:\t%f s\t%d rec ops\t%f us/op" % (
            self.bridge_time, self.bridge_rec_ops, t_bridge))
        a, t = self.bridge_fit.fit()
        saved, reach = self.failures_until(32, 200)
        debug_print("survivor:\ta=%f us\tb=%f us/op\tsaved(32..200)=%f"
                    "\treach=%f" % (a * 1e6, t * 1e6, saved, reach))
        self._print_pe_stats(cnt, tim)
        line = "TOTAL:      \t\t%f" % (self.tk - self.starttime, )
        debug_print(line)
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

    def _hist(self, hist):
        last = len(hist)
        while last > 0 and hist[last - 1] == 0:
            last -= 1
        return " ".join([str(hist[i]) for i in range(last)])

    def _print_line_time(self, string, i, tim):
        final = "%s:%s\t%d\t%f" % (string, " " * max(0, 13-len(string)), i, tim)
        debug_print(final)

    def _print_intline(self, string, i):
        final = string + ':' + " " * max(0, 16-len(string))
        final += '\t' + str(i)
        debug_print(final)


class BrokenProfilerData(JitException):
    pass
