
import py
from rpython.jit.metainterp.warmspot import ll_meta_interp
from rpython.rlib.jit import JitDriver
from rpython.jit.backend.llgraph import runner
from rpython.jit.metainterp.jitprof import Profiler, JITPROF_LINES
from rpython.jit.tool.jitoutput import parse_prof
from rpython.tool.logparser import parse_log, extract_category

def test_really_run():
    """ This test checks whether output of jitprof did not change.
    It'll explode when someone touches jitprof.py
    """
    mydriver = JitDriver(reds = ['i', 'n'], greens = [])
    def f(n):
        i = 0
        while i < n:
            mydriver.can_enter_jit(i=i, n=n)
            mydriver.jit_merge_point(i=i, n=n)
            i += 1

    cap = py.io.StdCaptureFD()
    try:
        ll_meta_interp(f, [10], CPUClass=runner.LLGraphCPU,
                       ProfilerClass=Profiler)
    finally:
        out, err = cap.reset()

    log = parse_log(err.splitlines(True))
    err_sections = list(extract_category(log, 'jit-summary'))
    [err1] = err_sections    # there should be exactly one jit-summary
    assert err1.count("\n") == JITPROF_LINES
    info = parse_prof(err1)
    # assert did not crash
    # asserts below are a bit delicate, possibly they might be deleted
    assert info.tracing_no == 1
    assert info.optimizing_no == 2
    assert info.backend_no == 1
    assert info.ops.total == 2
    assert info.recorded_ops.total == 2
    assert info.recorded_ops.calls == 0
    assert info.guards == 2
    assert info.opt_ops == 11
    assert info.opt_guards == 2
    assert info.forcings == 0
    assert info.guard_fail_hist == [1]

DATA = '''Tracing:         1       0.006992
Optimizing:      1       0.001250
Backend:        1       0.000525
Blackhole:      2       0.000300
Blackhole callee: 1     0.000100
Blackhole decode: 2     0.000050
guard failures >=2^k:	5 3 1
bridges at 2^k:	0 0 1
bridge model:	C=2.500000 us	B=30.000000 ns	break-even(100)=6.000000
bridge attempts:	0.100000 s	2000 rec ops	50.000000 us/op
PE cogen overhead: 4       0.000400
PE cogen scan:  2       0.000100
PE cogen install: 1       0.000250
TOTAL:                  0.025532
pe cogen generated:     1
pe cogen declined:      1
pe cogen deferred:      2
pe insns generic:       10
pe insns portal:        5
pe insns residual:      20
ops:                    2
heapcached ops:         111
recorded ops:           6
  calls:                3
guards:                 1
opt ops:                6
opt guards:             1
opt guards shared:      1
forcings:               1
abort: trace too long:  10
abort: compiling:       11
abort: vable escape:    12
abort: bad loop:        135
abort: force quasi-immut: 3
abort: segmenting trace: 0
virtualizables forced:  1123
nvirtuals:              13
nvholes:                14
nvreused:               15
vecopt tried:           12
vecopt success:         4
Total # of loops:       100
Total # of bridges:     300
Freed # of loops:       99
Freed # of bridges:     299
'''

def test_parse():
    info = parse_prof(DATA)
    assert info.tracing_no == 1
    assert info.tracing_time == 0.006992
    assert info.optimizing_no == 1
    assert info.optimizing_time == 0.001250
    assert info.backend_no == 1
    assert info.backend_time == 0.000525
    assert abs(info.compilation_time - 0.008767) < 1e-12
    assert info.pe_cogen_no == 4
    assert info.pe_cogen_overhead_time == 0.000400
    assert info.pe_cogen_scan_no == 2
    assert info.pe_cogen_scan_time == 0.000100
    assert info.pe_cogen_install_no == 1
    assert info.pe_cogen_install_time == 0.000250
    assert abs(info.pe_cogen_time - 0.000750) < 1e-12
    assert info.pe_cogen_generated == 1
    assert info.pe_cogen_declined == 1
    assert info.pe_cogen_deferred == 2
    assert info.pe_insns_generic == 10
    assert info.pe_insns_residual == 20
    assert info.ops.total == 2
    assert info.heapcached_ops == 111
    assert info.recorded_ops.total == 6
    assert info.recorded_ops.calls == 3
    assert info.guards == 1
    assert info.opt_ops == 6
    assert info.opt_guards == 1
    assert info.forcings == 1
    assert info.abort.trace_too_long == 10
    assert info.abort.compiling == 11
    assert info.abort.vable_escape == 12
    assert info.abort.bad_loop == 135
    assert info.abort.force_quasiimmut == 3
    assert info.virtualizables_forced == 1123
    assert info.nvirtuals == 13
    assert info.nvholes == 14
    assert info.nvreused == 15
    assert info.vecopt_tried == 12
    assert info.vecopt_success == 4
    assert info.guard_fail_hist == [5, 3, 1]
    assert info.bridge_at_hist == [0, 0, 1]
