
""" Helpers for parsing various outputs jit produces.
Notably:
1. Statistics of log.ops
2. Parsing what jitprof produces
"""

import re

REGEXES = [
    (('tracing_no', 'tracing_time'), '^Tracing:\s+([\d.]+)\s+([\d.]+)$'),
    (('optimizing_no', 'optimizing_time'),
     '^Optimizing:\s+([\d.]+)\s+([\d.]+)$'),
    (('backend_no', 'backend_time'), '^Backend:\s+([\d.]+)\s+([\d.]+)$'),
    (('blackhole_no', 'blackhole_time'),
     '^Blackhole:\s+([\d.]+)\s+([\d.]+)$'),
    (('blackhole_call_no', 'blackhole_call_time'),
     '^Blackhole callee:\s+([\d.]+)\s+([\d.]+)$'),
    (('blackhole_decode_no', 'blackhole_decode_time'),
     '^Blackhole decode:\s+([\d.]+)\s+([\d.]+)$'),
    (('guard_fail_hist',), '^guard failures >=2\^k:\s*(.*)$'),
    (('bridge_at_hist',), '^bridges at 2\^k:\s*(.*)$'),
    (('bridge_model_c', 'bridge_model_b', 'bridge_model_be'),
     '^bridge model:\s+C=([-\d.]+) us\s+B=([-\d.]+) ns\s+'
     'break-even\(100\)=([-\d.]+)$'),
    (('bridge_attempt_time', 'bridge_attempt_ops', 'bridge_attempt_us'),
     '^bridge attempts:\s+([\d.]+) s\s+(\d+) rec ops\s+([\d.]+) us/op$'),
    (('survivor_a', 'survivor_b', 'survivor_saved', 'survivor_reach'),
     '^survivor:\s+a=([-\d.]+) us\s+b=([-\d.]+) exp\s+'
     'saved\(32\.\.200\)=([-\d.]+)\s+reach=([-\d.]+)$'),
    (('pe_cogen_no', 'pe_cogen_overhead_time'),
     '^PE cogen overhead:\s+([\d.]+)\s+([\d.]+)$'),
    (('pe_cogen_scan_no', 'pe_cogen_scan_time'),
     '^PE cogen scan:\s+([\d.]+)\s+([\d.]+)$'),
    (('pe_cogen_install_no', 'pe_cogen_install_time'),
     '^PE cogen install:\s+([\d.]+)\s+([\d.]+)$'),
    (('pe_cogen_generated',), '^pe cogen generated:\s+(\d+)$'),
    (('pe_cogen_declined',), '^pe cogen declined:\s+(\d+)$'),
    (('pe_cogen_deferred',), '^pe cogen deferred:\s+(\d+)$'),
    (('pe_insns_generic',), '^pe insns generic:\s+(\d+)$'),
    (('pe_insns_portal',), '^pe insns portal:\s+(\d+)$'),
    (('pe_insns_residual',), '^pe insns residual:\s+(\d+)$'),
    (None, '^TOTAL.*$'),
    (('ops.total',), '^ops:\s+(\d+)$'),
    (('heapcached_ops', ), '^heapcached ops:\s+(\d+)$'),
    (('recorded_ops.total',), '^recorded ops:\s+(\d+)$'),
    (('recorded_ops.calls',), '^\s+calls:\s+(\d+)$'),
    (('guards',), '^guards:\s+(\d+)$'),
    (('opt_ops',), '^opt ops:\s+(\d+)$'),
    (('opt_guards',), '^opt guards:\s+(\d+)$'),
    (('opt_guards_shared',), '^opt guards shared:\s+(\d+)$'),
    (('forcings',), '^forcings:\s+(\d+)$'),
    (('abort.trace_too_long',), '^abort: trace too long:\s+(\d+)$'),
    (('abort.compiling',), '^abort: compiling:\s+(\d+)$'),
    (('abort.vable_escape',), '^abort: vable escape:\s+(\d+)$'),
    (('abort.bad_loop',), '^abort: bad loop:\s+(\d+)$'),
    (('abort.force_quasiimmut',), '^abort: force quasi-immut:\s+(\d+)$'),
    (('abort.segmenting_trace',), '^abort: segmenting trace:\s+(\d+)$'),
    (('virtualizables_forced',), '^virtualizables forced:\s+(\d+)$'),
    (('nvirtuals',), '^nvirtuals:\s+(\d+)$'),
    (('nvholes',), '^nvholes:\s+(\d+)$'),
    (('nvreused',), '^nvreused:\s+(\d+)$'),
    (('vecopt_tried',), '^vecopt tried:\s+(\d+)$'),
    (('vecopt_success',), '^vecopt success:\s+(\d+)$'),
    (('total_compiled_loops',),   '^Total # of loops:\s+(\d+)$'),
    (('total_compiled_bridges',), '^Total # of bridges:\s+(\d+)$'),
    (('total_freed_loops',),      '^Freed # of loops:\s+(\d+)$'),
    (('total_freed_bridges',),    '^Freed # of bridges:\s+(\d+)$'),
    ]

class Ops(object):
    total = 0

class RecordedOps(Ops):
    calls = 0

class Aborts(object):
    trace_too_long = 0
    compiling = 0
    vable_escape = 0

class OutputInfo(object):
    compilation_time = 0.0
    tracing_no = 0
    tracing_time = 0.0
    optimizing_no = 0
    optimizing_time = 0.0
    backend_no = 0
    backend_time = 0.0
    blackhole_no = 0
    blackhole_time = 0.0
    blackhole_call_no = 0
    blackhole_call_time = 0.0
    bridge_attempt_time = 0.0
    bridge_attempt_ops = 0
    bridge_attempt_us = 0.0
    bridge_model_c = 0.0
    bridge_model_b = 0.0
    bridge_model_be = 0.0
    pe_cogen_no = 0
    pe_cogen_time = 0.0
    pe_cogen_overhead_time = 0.0
    pe_cogen_scan_no = 0
    pe_cogen_scan_time = 0.0
    pe_cogen_install_no = 0
    pe_cogen_install_time = 0.0
    pe_cogen_generated = 0
    pe_cogen_declined = 0
    pe_cogen_deferred = 0
    pe_insns_generic = 0
    pe_insns_portal = 0
    pe_insns_residual = 0
    asm_no = 0
    asm_time = 0.0
    guards = 0
    opt_ops = 0
    opt_guards = 0
    forcings = 0
    nvirtuals = 0
    nvholes = 0
    nvreused = 0
    vecopt_tried = 0
    vecopt_success = 0

    def __init__(self):
        self.ops = Ops()
        self.recorded_ops = RecordedOps()
        self.abort = Aborts()

def parse_prof(output):
    lines = output.splitlines()
    # assert len(lines) == len(REGEXES)
    info = OutputInfo()
    for (attrs, regexp), line in zip(REGEXES, lines):
        m = re.match(regexp, line)
        assert m is not None, "Error parsing line: %s" % line
        if attrs:
            for i, a in enumerate(attrs):
                v = m.group(i + 1)
                if a.endswith('_hist'):
                    v = [int(x) for x in v.split()]
                elif '.' in v:
                    v = float(v)
                else:
                    v = int(v)
                if '.' in a:
                    before, after = a.split('.')
                    setattr(getattr(info, before), after, v)
                else:
                    setattr(info, a, v)
    info.compilation_time = (info.tracing_time + info.optimizing_time +
                             info.backend_time)
    info.pe_cogen_time = (info.pe_cogen_overhead_time +
                          info.pe_cogen_scan_time +
                          info.pe_cogen_install_time)
    return info
