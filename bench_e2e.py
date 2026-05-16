#!/usr/bin/env python
"""bench_e2e.py - Measure tracing time and running time per benchmark.

Three metrics are captured per (binary, benchmark) invocation:

  - **Tracing time**: from PYPYLOG `jit-summary`, the
    `Tracing (total)` bucket (or older-format `Tracing` bucket).
    This is the time the JIT spent recording bytecode into the
    trace buffer + the optimizer + the resume-data encoder. It's
    what the T-catalogue optimizations are designed to reduce.

  - **Running time**: median of per-iteration times printed by the
    benchmark script's `util.run_benchmark` loop, dropping the
    first N iterations as warmup (default N=10). This is the
    steady-state user-code execution time *after* the JIT has
    compiled the hot loops — i.e. the time the assembly emitted
    by the JIT actually takes to do the workload.

  - **JIT total time**: from PYPYLOG `jit-summary`, the `TOTAL`
    bucket. This is the grand total of all JIT compilation work
    (tracing + optimizer + code generation + resume data + more).

Why these metrics:
  - The T-* optimizations live entirely in the slow trace recorder
    and are designed to make tracing cheaper without changing trace
    shape (loop count, bridge count, ops emitted are unchanged).
    So `tracing` should fall and `running` should be unchanged
    (or fall slightly via knock-on effects on bridge formation).

These map directly onto genextension's premise: genext skips the
recorder + optimizer + backend entirely for pure-arithmetic loops
("compile shortcut"). The T-* optimizations apply the same
"specialize the hot path; bypass general machinery" idea to the
slow path that genext can't take. Reducing tracing time is the
primary success metric; reducing running time is a downstream
effect (better bridges -> tighter compiled code).

Schema (per binary):
  {"pypy": <abs path>, "reps": int, "n": int,
   "raw":  { name: [{"wall": s, "tracing": s, "jit_total": s,
                     "running": s, "iter_times": [...]}] },
   "summary": { name: {
       "tracing_med": s, "tracing_min": s,
       "running_med": s, "running_min": s,
       "jit_total_med": s,
       "wall_med": s,
       "n": int }}}

Usage:

  # HBP-vs-baseline: one binary, two --jit policies, interleaved A/B
  # (drift-symmetric), writes e2e_baseline.json + e2e_hbp.json:
  bench_e2e.py --hbp --pypy ./pypy/goal/pypy-c --reps 10 -n 50

  # Same, but tune the two JIT arms explicitly:
  bench_e2e.py --hbp --pypy ./pypy/goal/pypy-c \\
      --jit-hbp enable_hot_bridge_promotion=1,enable_tracetree=1 \\
      --reps 10 -n 50

  # Bench both standard build targets (baseline `pypy-c` then proposed
  # `pypy-jit-ext-c`) and print the head-to-head comparison:
  bench_e2e.py --all --reps 10 -n 50

  # Bench a single binary, write a JSON:
  bench_e2e.py --pypy ./pypy/goal/pypy-c --reps 10 -n 50 \\
               --out e2e_baseline.json

  # Bench an arbitrary set of binaries sequentially.  With exactly
  # two binaries the comparison is printed automatically:
  bench_e2e.py --pypy ./pypy/goal/pypy-c \\
               --pypy ./pypy/goal/pypy-jit-ext-c \\
               --reps 10 -n 50 --out e2e_run1

  # Compare two JSONs that already exist on disk:
  bench_e2e.py --compare e2e_baseline.json e2e_proposed.json
"""
from __future__ import print_function
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(REPO, "benchmarks")

# (benchmark name, script path relative to benchmarks/, env, extra args)
BENCHMARKS = [
    ("ai",            "unladen_swallow/performance/bm_ai.py", {}, []),
    ("bm_chameleon",  "own/bm_chameleon.py",                  {"PYTHONPATH": "lib/chameleon/src"}, []),
    ("bm_dulwich_log","own/bm_dulwich_log.py",                {"PYTHONPATH": "lib/dulwich-0.19.13"}, []),
    ("bm_krakatau",   "own/bm_krakatau.py",                   {"PYTHONPATH": "lib/krakatau"}, []),
    ("bm_mako",       "own/bm_mako.py",                       {"PYTHONPATH": "lib/mako"}, []),
    ("bm_mdp",        "own/bm_mdp.py",                        {}, []),
    ("chaos",         "own/chaos.py",                         {}, []),
    ("crypto_pyaes",  "own/crypto_pyaes.py",                  {}, []),
    ("deltablue",     "own/deltablue.py",                     {}, []),
    ("django",        "unladen_swallow/performance/bm_django.py", {"PYTHONPATH": "unladen_swallow/lib/django:lib"}, []),
    ("eparse",        "own/eparse.py",                        {"PYTHONPATH": "lib/monte"}, []),
    ("fannkuch",      "own/fannkuch.py",                      {}, []),
    ("float",         "own/float.py",                         {}, []),
    ("genshi_text",   "own/bm_genshi.py",                     {"PYTHONPATH": "lib/genshi"}, ["--benchmark=text"]),
    ("genshi_xml",    "own/bm_genshi.py",                     {"PYTHONPATH": "lib/genshi"}, ["--benchmark=xml"]),
    ("go",            "own/go.py",                            {}, []),
    ("html5lib",      "unladen_swallow/performance/bm_html5lib.py", {"PYTHONPATH": "unladen_swallow/lib/html5lib:lib"}, []),
    ("json_bench",    "own/json_bench.py",                    {}, []),
    ("meteor-contest","own/meteor-contest.py",                {}, []),
    ("nbody_modified","own/nbody_modified.py",                {}, []),
    ("nqueens",       "own/nqueens.py",                       {}, []),
    ("pickle",        "unladen_swallow/performance/bm_pickle.py", {}, ["pickle"]),
    ("pickle_dict",   "unladen_swallow/performance/bm_pickle.py", {}, ["pickle_dict"]),
    ("pickle_list",   "unladen_swallow/performance/bm_pickle.py", {}, ["pickle_list"]),
    ("pidigits",      "own/pidigits.py",                      {}, []),
    ("pyflate-fast",  "own/pyflate-fast.py",                  {}, []),
    ("pyxl_bench",    "own/pyxl_bench.py",                    {"PYTHONPATH": "lib/pyxl"}, []),
    ("raytrace-simple","own/raytrace-simple.py",              {}, []),
    ("richards",      "unladen_swallow/performance/bm_richards.py", {}, []),
    ("scimark_fft",   "own/scimark.py",                       {}, ["--benchmark=FFT", "1024", "1000"]),
    ("scimark_lu",    "own/scimark.py",                       {}, ["--benchmark=LU", "100", "200"]),
    ("scimark_montecarlo", "own/scimark.py",                  {}, ["--benchmark=MonteCarlo", "5000000"]),
    ("scimark_sor",   "own/scimark.py",                       {}, ["--benchmark=SOR", "100", "5000", "Array2D"]),
    ("scimark_sparsematmult", "own/scimark.py",               {}, ["--benchmark=SparseMatMult", "1000", "50000", "2000"]),
    ("spambayes",     "unladen_swallow/performance/bm_spambayes.py", {"PYTHONPATH": "unladen_swallow/lib/spambayes:unladen_swallow/lib/lockfile"}, []),
    ("spectral-norm", "own/spectral-norm.py",                 {}, []),
    ("spitfire2",     "own/spitfire.py",                      {}, ["--benchmark=spitfire_o3"]),
    ("spitfire_cstringio2","own/spitfire.py",                 {}, ["--benchmark=python_cstringio"]),
    ("sqlalchemy_declarative","own/sqlalchemy_declarative.py",{"PYTHONPATH": "lib/sqlalchemy/lib"}, []),
    ("sqlalchemy_imperative", "own/sqlalchemy_imperative.py", {"PYTHONPATH": "lib/sqlalchemy/lib"}, []),
    ("sqlitesynth",   "own/sqlitesynth.py",                   {}, []),
    ("sympy_expand",  "own/bm_sympy.py",                      {"PYTHONPATH": "lib/sympy"}, ["--benchmark=expand"]),
    ("sympy_integrate","own/bm_sympy.py",                     {"PYTHONPATH": "lib/sympy"}, ["--benchmark=integrate"]),
    ("sympy_str",     "own/bm_sympy.py",                      {"PYTHONPATH": "lib/sympy"}, ["--benchmark=str"]),
    ("sympy_sum",     "own/bm_sympy.py",                      {"PYTHONPATH": "lib/sympy"}, ["--benchmark=sum"]),
    ("telco",         "own/telco.py",                         {}, []),
    ("unpickle",      "unladen_swallow/performance/bm_pickle.py", {}, ["unpickle"]),
    ("unpickle_list", "unladen_swallow/performance/bm_pickle.py", {}, ["unpickle_list"]),
]


_FLOAT = r"[0-9]+(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?)?"
_LINE_FLOAT_ONLY = re.compile(r"^\s*(" + _FLOAT + r")\s*$")


def parse_jit_summary(text):
    """Extract the JIT-self-measured warmup metrics from PYPYLOG jit-summary.

    Per design-doc S20 these are the *primary* warmup metrics (the VM
    measures them itself, so they are far less sensitive to OS/thermal/
    P-E-scheduler noise than wall-clock): the per-phase compilation time
    decomposition + the structural counts that drive it. genext targets
    exactly these.
    """
    res = {"tracing": float("nan"), "optimization": float("nan"),
           "backend": float("nan"), "jit_total": float("nan")}
    int_fields = [
        ("ops:", "ops"), ("recorded ops:", "recorded_ops"),
        ("guards:", "guards"), ("opt ops:", "opt_ops"),
        ("opt guards:", "opt_guards"), ("forcings:", "forcings"),
        ("abort: trace too long:", "abort_trace_too_long"),
        ("abort: compiling:", "abort_compiling"),
        ("abort: vable escape:", "abort_vable_escape"),
        ("abort: bad loop:", "abort_bad_loop"),
        ("abort: force quasi-immut:", "abort_force_quasi_immut"),
        ("abort: segmenting trace:", "abort_segmenting"),
        ("Total # of loops:", "loops"),
        ("Total # of bridges:", "bridges"),
        ("slow tracing function executions:", "slow_tracing_exec"),
        ("fast tracing function executions:", "fast_tracing_exec"),
    ]
    for _p, k in int_fields:
        res[k] = -1
    in_section = False
    for line in text.splitlines():
        if "{jit-summary" in line:
            in_section = True
            continue
        if in_section and "jit-summary}" in line:
            break
        if not in_section:
            continue
        s = line.strip()
        # time buckets: "Name (total):   <count>   <secs>"
        if s.startswith("Tracing (total):") or s.startswith("Tracing:"):
            nums = re.findall(_FLOAT, s.split(":", 1)[1])
            if len(nums) >= 2:
                res["tracing"] = float(nums[1])
        elif s.startswith("Optimization:"):
            nums = re.findall(_FLOAT, s.split(":", 1)[1])
            if len(nums) >= 2:
                res["optimization"] = float(nums[1])
            elif nums:
                res["optimization"] = float(nums[0])
        elif s.startswith("Backend:"):
            nums = re.findall(_FLOAT, s.split(":", 1)[1])
            if len(nums) >= 2:
                res["backend"] = float(nums[1])
            elif nums:
                res["backend"] = float(nums[0])
        elif s.startswith("TOTAL:"):
            nums = re.findall(_FLOAT, s.split(":", 1)[1])
            if nums:
                res["jit_total"] = float(nums[-1])
        else:
            for prefix, key in int_fields:
                if s.startswith(prefix):
                    nums = re.findall(r"-?\d+", s.split(":", 1)[1])
                    if nums:
                        res[key] = int(nums[-1])
                    break
    ab = 0
    for k in ("abort_trace_too_long", "abort_compiling", "abort_vable_escape",
              "abort_bad_loop", "abort_force_quasi_immut", "abort_segmenting"):
        if res.get(k, -1) > 0:
            ab += res[k]
    res["abort_total"] = ab
    return res


def parse_iter_times(text):
    """Pull per-iteration times printed by the benchmark's run_benchmark."""
    times = []
    for line in text.splitlines():
        m = _LINE_FLOAT_ONLY.match(line)
        if m:
            try:
                times.append(float(m.group(1)))
            except ValueError:
                pass
    return times


def _mad(xs):
    """Median absolute deviation (robust dispersion)."""
    if not xs:
        return float("nan")
    m = median(xs)
    return median([abs(x - m) for x in xs])


def analyze_warmup(iter_times):
    """Barrett-et-al.-style (OOPSLA'17) warmup classification of one
    per-iteration series. Pragmatic suffix-stability detector (not their
    exact PELT/krun), grounded in the same principles:

      - the steady state is the maximal *stable suffix* of the series
        (per-iter time stays inside a robust band derived from the tail's
        own MAD), NOT a fixed warmup-drop and NOT a noisy asymptote+
        threshold (the warmup_tax estimator that produced the retracted
        S12/S14 over-claims);
      - we do not assume a steady state exists -> we *classify*:
        flat / warmup / slowdown / no_steady / too_short;
      - the reported warmup scalars come from the detected change point.

    Returns: classification, steady_start_iter (-1 if none), steady_perf
    (median of the steady suffix), wall_to_steady (cum time before it),
    speedup_vs_first (first-iter / steady_perf).
    """
    t = list(iter_times)
    n = len(t)
    out = {"classification": "too_short", "steady_start_iter": -1,
           "steady_perf": float("nan"), "wall_to_steady": float("nan"),
           "speedup_vs_first": float("nan"), "n_iters": n}
    if n < 8:
        return out
    tail = t[-max(4, n // 3):]
    smed = median(tail)
    smad = _mad(tail)
    # robust half-band: 3 robust-sigma, floored at 5% relative
    band = max(3.0 * 1.4826 * smad, 0.05 * smed)
    lo, hi = smed - band, smed + band

    def stable_from(i):
        for v in t[i:]:
            if v < lo or v > hi:
                return False
        return True

    start = -1
    for i in range(n):
        if stable_from(i):
            start = i
            break
    out["steady_perf"] = smed
    if start < 0:
        out["classification"] = "no_steady"
        return out
    out["steady_start_iter"] = start
    out["wall_to_steady"] = sum(t[:start])
    if t[0] > 0:
        out["speedup_vs_first"] = t[0] / smed if smed > 0 else float("nan")
    if start == 0:
        out["classification"] = "flat"
        return out
    pre = t[:start]
    # slowdown: some pre-steady iteration was *faster* than steady (the
    # program ran fast early then degraded into a slower steady state)
    if min(pre) < lo:
        out["classification"] = "slowdown"
    else:
        out["classification"] = "warmup"
    return out


def cumulative_break_even(jit_series, nojit_series):
    """Iteration index where cumulative JIT-on time first drops to <=
    cumulative interpreter-only (--jit off) time, i.e. the warmup
    break-even point (S20 robust scalar). -1 if not reached within the
    measured window."""
    cj = 0.0
    cn = 0.0
    m = min(len(jit_series), len(nojit_series))
    for i in range(m):
        cj += jit_series[i]
        cn += nojit_series[i]
        if cj <= cn:
            return i
    return -1


def cpu_speed_limit_pct():
    """macOS thermal throttle indicator: CPU_Speed_Limit % from
    `pmset -g therm` (100 = not throttled). None if unavailable."""
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"],
                                      stderr=subprocess.STDOUT)
        for line in out.decode("utf-8", "replace").splitlines():
            if "CPU_Speed_Limit" in line:
                m = re.findall(r"(\d+)", line.split("=")[-1])
                if m:
                    return int(m[0])
    except Exception:
        pass
    return None


def _stable_child_env(env):
    """Deterministic, low-noise child env (S20): fixed hash seed for
    dict-order determinism across process executions; single-thread."""
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


_HAVE_CAFFEINATE = None


def _caffeinated(cmd):
    """Wrap cmd in `caffeinate -dimsu` so no idle/display/disk sleep or
    App Nap perturbs the timed run. No-op if caffeinate is absent."""
    global _HAVE_CAFFEINATE
    if _HAVE_CAFFEINATE is None:
        path = None
        if os.path.exists("/usr/bin/caffeinate"):
            path = "/usr/bin/caffeinate"
        else:
            for d in os.environ.get("PATH", "").split(os.pathsep):
                cand = os.path.join(d, "caffeinate")
                if d and os.path.exists(cand):
                    path = cand
                    break
        _HAVE_CAFFEINATE = path
    if _HAVE_CAFFEINATE:
        return [_HAVE_CAFFEINATE, "-dimsu"] + cmd
    return cmd


def run_one(pypy, script, env_extra, extra_args, n, warmup_iters=10,
            jit_off=False, prewarm=True, jit_params=None):
    """Run a single (binary, benchmark) once. Returns dict of metrics.

    `jit_params`: optional comma-separated `--jit` parameter string
    (e.g. "enable_hot_bridge_promotion=1,enable_tracetree=1"). When set
    (and jit_off is False) it is passed verbatim as `--jit <params>`,
    so the SAME binary can be A/B'd under different JIT policies (the
    HBP-vs-baseline mode). Ignored when jit_off=True (`--jit off` wins).

    Stable-setting hygiene (S20): deterministic child env, caffeinate
    wrap, and one untimed pre-warm run (fs/code-cache) discarded before
    the measured run. The full per-iteration series is kept (warmup is
    NOT dropped); analyze_warmup() derives the warmup scalars and
    `running` is the *detected* steady suffix (regression guard only)."""
    env = os.environ.copy()
    for k, v in env_extra.items():
        parts = []
        for piece in v.split(os.pathsep):
            if not piece:
                continue
            parts.append(piece if os.path.isabs(piece)
                         else os.path.join(BENCH_DIR, piece))
        env[k] = os.pathsep.join(parts)
    _stable_child_env(env)
    if jit_off:
        jit_args = ["--jit", "off"]
    elif jit_params:
        jit_args = ["--jit", jit_params]
    else:
        jit_args = []
    base_cmd = ([pypy] + jit_args
                + [os.path.join(BENCH_DIR, script), "-n", str(n)]
                + list(extra_args))

    def _exec(capture_log):
        tmp = tempfile.NamedTemporaryFile(prefix="pypylog_e2e_",
                                          suffix=".txt", delete=False)
        tmp.close()
        e = dict(env)
        e["PYPYLOG"] = ("jit-summary:%s" % tmp.name) if capture_log else "-"
        t0 = time.time()
        try:
            p = subprocess.Popen(_caffeinated(base_cmd),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=e)
            stdout, _ = p.communicate()
            rc = p.returncode
        except OSError as ex:
            try: os.unlink(tmp.name)
            except OSError: pass
            return None, None, None, ("OSError: %s" % ex)
        wall = time.time() - t0
        try:
            with open(tmp.name) as f:
                logtxt = f.read()
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass
        if rc != 0:
            return None, None, wall, ("rc=%d" % rc)
        return stdout, logtxt, wall, None

    # One untimed pre-warm run (fs/code cache) discarded -> the measured
    # run starts from a warm OS state (S20 stable-setting hygiene).
    if prewarm:
        _exec(False)

    stdout, log, wall, err = _exec(True)
    if err is not None:
        return {"error": err, "wall": wall}
    summary = parse_jit_summary(log)
    iter_times = parse_iter_times(stdout.decode())
    # Warmup is NOT dropped: keep the full series and let the changepoint
    # detector locate the steady suffix (S20).
    wm = analyze_warmup(iter_times)
    steady_i = wm["steady_start_iter"]
    if steady_i >= 0:
        steady = iter_times[steady_i:]
    else:
        # no detected steady state -> fall back to the legacy fixed drop
        # purely for the regression-guard number (flagged via classif.)
        steady = (iter_times[warmup_iters:]
                  if len(iter_times) > warmup_iters else iter_times)
    running_med = median(steady) if steady else float("nan")
    running_sum = sum(steady) if steady else float("nan")
    out = {
        "wall": wall,
        # S20 primary (JIT-self-measured) warmup metrics:
        "tracing": summary["tracing"],
        "optimization": summary["optimization"],
        "backend": summary["backend"],
        "jit_total": summary["jit_total"],
        # warmup-curve scalars (changepoint-derived, not warmup_tax):
        "warmup_class": wm["classification"],
        "steady_start_iter": wm["steady_start_iter"],
        "wall_to_steady": wm["wall_to_steady"],
        "speedup_vs_first": wm["speedup_vs_first"],
        # steady-state regression guard (secondary, NOT the headline):
        "running": running_med,
        "running_sum": running_sum,
        "iter_times": iter_times,
    }
    for k in ("ops", "recorded_ops", "guards", "opt_guards", "loops",
              "bridges", "abort_total", "fast_tracing_exec",
              "slow_tracing_exec"):
        out[k] = summary.get(k, -1)
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def aggregate(samples, key):
    vs = [s[key] for s in samples
          if isinstance(s.get(key), (int, float)) and s[key] == s[key]]
    if not vs:
        return float("nan"), float("nan"), float("nan"), 0
    return median(vs), min(vs), max(vs), len(vs)


def _variance(xs):
    xs = [x for x in xs if x == x]
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / float(n)
    return sum((x - m) ** 2 for x in xs) / float(n - 1)


def _build_summary(samples):
    """Aggregate one benchmark's reps into the summary dict.

    bench_e2e supports TWO co-equal, distinct measurements (S20):
      * WARMUP performance  -- the transient cost (interp + JIT
        compile + early machine-code exec). Measured by the
        JIT-self-reported phase decomposition + changepoint scalars.
      * STEADY-STATE (STABLE) performance -- the asymptotic per-
        iteration time once the JIT has finished compiling the hot
        code (how fast the *emitted machine code* actually is).
        Measured as the median of the *detected* steady segment,
        with variance + a reliability flag (only trustworthy if a
        genuine steady state was reached -- Barrett et al.: do NOT
        report steady perf for no_steady series).
    Neither is "secondary"; they answer different questions.
    """
    classes = [s.get("warmup_class") for s in samples if s.get("warmup_class")]
    cls = max(set(classes), key=classes.count) if classes else "n/a"
    be = [s["break_even"] for s in samples
          if isinstance(s.get("break_even"), int) and s["break_even"] >= 0]
    run_vals = [s["running"] for s in samples
                if isinstance(s.get("running"), (int, float))
                and s["running"] == s["running"]]
    run_med = median(run_vals) if run_vals else float("nan")
    run_var = _variance(run_vals)
    run_cv = ((run_var ** 0.5) / run_med) if (run_med and run_med == run_med
                                              and run_med != 0) else float("nan")
    # steady number is trustworthy only if the curve actually reaches a
    # steady state for the (majority) of reps.
    n_reached = sum(1 for s in samples
                    if s.get("warmup_class") in ("flat", "warmup", "slowdown")
                    and s.get("steady_start_iter", -1) >= 0)
    steady_reliable = bool(samples) and n_reached >= (len(samples) + 1) // 2
    d = {
        # ---- WARMUP performance (transient cost) ----
        "tracing_med": aggregate(samples, "tracing")[0],
        "tracing_min": aggregate(samples, "tracing")[1],
        "optimization_med": aggregate(samples, "optimization")[0],
        "backend_med": aggregate(samples, "backend")[0],
        "jit_total_med": aggregate(samples, "jit_total")[0],
        "jit_total_min": aggregate(samples, "jit_total")[1],
        "warmup_class": cls,
        "wall_to_steady_med": aggregate(samples, "wall_to_steady")[0],
        "steady_start_iter_med": aggregate(samples, "steady_start_iter")[0],
        "break_even_med": (median(be) if be else -1),
        # ---- STEADY-STATE (STABLE) performance (co-equal) ----
        "steady_perf_med": run_med,
        "steady_perf_min": (min(run_vals) if run_vals else float("nan")),
        "steady_perf_var": run_var,
        "steady_perf_cv": run_cv,           # robustness of the steady est.
        "steady_reliable": steady_reliable, # False -> steady_perf is n/a
        # back-compat alias (older report code / external readers):
        "running_med": run_med,
        "running_min": (min(run_vals) if run_vals else float("nan")),
        "wall_med": aggregate(samples, "wall")[0],
        "n": aggregate(samples, "tracing")[3],
    }
    for k in ("loops", "bridges", "guards", "abort_total",
              "fast_tracing_exec", "slow_tracing_exec"):
        d[k + "_med"] = aggregate(samples, k)[0]
    return d


def _thermal_guard(label, cooldown=8, max_wait=120):
    """S20 stable-setting: if the CPU is thermally throttled, idle until
    it recovers (bounded). Returns the speed-limit %% recorded for the
    upcoming rep (100 / None = not throttled / unknown)."""
    waited = 0
    while True:
        pct = cpu_speed_limit_pct()
        if pct is None or pct >= 100 or waited >= max_wait:
            return pct
        print("  [thermal] %s: CPU limited to %d%%, cooldown %ds" %
              (label, pct, cooldown))
        sys.stdout.flush()
        time.sleep(cooldown)
        waited += cooldown


def run_all_interleaved(base, cand, names, reps, n, warmup_iters=10,
                        break_even=False, prewarm=True):
    """S20 stable-setting comparison: for each (benchmark, rep) run the
    BASELINE then the CANDIDATE back to back (A,B,A,B...), never
    all-A-then-all-B -- so slow thermal/DVFS drift hits both arms
    symmetrically instead of confounding the A/B delta. Serial only
    (jobs=1) for decision-grade signal. Returns {pypy: (raw, summary)}."""
    by_name = dict((b[0], b) for b in BENCHMARKS)
    valid = [nm for nm in names if nm in by_name]
    sp = {base: dict((nm, []) for nm in valid),
          cand: dict((nm, []) for nm in valid)}
    for nm in valid:
        _, script, env, extra = by_name[nm]
        print("[%s] %d reps (interleaved baseline/candidate)" % (nm, reps))
        sys.stdout.flush()
        be_cache = {}
        if break_even:
            for who in (base, cand):
                ro = run_one(who, script, env, extra, n, warmup_iters,
                             jit_off=True, prewarm=prewarm)
                be_cache[who] = (ro.get("iter_times")
                                 if "error" not in ro else None)
        for i in range(reps):
            for who in (base, cand):
                thr = _thermal_guard("%s/%s" % (nm, os.path.basename(who)))
                r = run_one(who, script, env, extra, n, warmup_iters,
                            prewarm=prewarm)
                if "error" in r:
                    print("  [%s] rep %d %s ERROR %s" %
                          (nm, i, os.path.basename(who), r["error"]))
                    sys.stdout.flush()
                    continue
                r["cpu_speed_limit"] = thr
                if break_even and be_cache.get(who):
                    r["break_even"] = cumulative_break_even(
                        r.get("iter_times", []), be_cache[who])
                sp[who][nm].append(r)
                print("  [%s] rep %d %-7s trace=%.4f opt=%.4f back=%.4f "
                      "jit=%.4f %s run=%.5f" %
                      (nm, i, os.path.basename(who), r["tracing"],
                       r.get("optimization", float("nan")),
                       r.get("backend", float("nan")), r["jit_total"],
                       r["warmup_class"], r["running"]))
                sys.stdout.flush()
    res = {}
    for who in (base, cand):
        raw = {}
        summ = {}
        for nm in valid:
            raw[nm] = sp[who][nm]
            if sp[who][nm]:
                summ[nm] = _build_summary(sp[who][nm])
        res[who] = (raw, summ)
    return res


def run_hbp_interleaved(pypy, jit_base, jit_hbp, names, reps, n,
                        warmup_iters=10, break_even=False, prewarm=True):
    """S20 stable interleaved A/B of the SAME binary under two JIT
    policies: BASELINE (`jit_base`) vs HBP (`jit_hbp`).

    Identical to run_all_interleaved's drift-symmetric A,B,A,B schedule
    (so thermal/DVFS noise hits both policies equally), except the two
    arms differ only by their `--jit` parameter string, not by binary.
    `jit_base` "" means stock JIT defaults (HBP off). Returns
    {"baseline": (raw, summary), "hbp": (raw, summary)}."""
    by_name = dict((b[0], b) for b in BENCHMARKS)
    valid = [nm for nm in names if nm in by_name]
    arms = (("baseline", jit_base), ("hbp", jit_hbp))
    sp = dict((lbl, dict((nm, []) for nm in valid)) for lbl, _ in arms)
    for nm in valid:
        _, script, env, extra = by_name[nm]
        print("[%s] %d reps (interleaved baseline/hbp, same binary)" %
              (nm, reps))
        sys.stdout.flush()
        be_cache = {}
        if break_even:
            # JIT-off curve is policy-independent -> measure once, reuse.
            ro = run_one(pypy, script, env, extra, n, warmup_iters,
                         jit_off=True, prewarm=prewarm)
            be = ro.get("iter_times") if "error" not in ro else None
            for lbl, _ in arms:
                be_cache[lbl] = be
        for i in range(reps):
            for lbl, jp in arms:
                thr = _thermal_guard("%s/%s" % (nm, lbl))
                r = run_one(pypy, script, env, extra, n, warmup_iters,
                            prewarm=prewarm, jit_params=(jp or None))
                if "error" in r:
                    print("  [%s] rep %d %-8s ERROR %s" %
                          (nm, i, lbl, r["error"]))
                    sys.stdout.flush()
                    continue
                r["cpu_speed_limit"] = thr
                if break_even and be_cache.get(lbl):
                    r["break_even"] = cumulative_break_even(
                        r.get("iter_times", []), be_cache[lbl])
                sp[lbl][nm].append(r)
                print("  [%s] rep %d %-8s trace=%.4f opt=%.4f back=%.4f "
                      "jit=%.4f %s run=%.5f" %
                      (nm, i, lbl, r["tracing"],
                       r.get("optimization", float("nan")),
                       r.get("backend", float("nan")), r["jit_total"],
                       r["warmup_class"], r["running"]))
                sys.stdout.flush()
    res = {}
    for lbl, _ in arms:
        raw = {}
        summ = {}
        for nm in valid:
            raw[nm] = sp[lbl][nm]
            if sp[lbl][nm]:
                summ[nm] = _build_summary(sp[lbl][nm])
        res[lbl] = (raw, summ)
    return res


def _run_one_task(task):
    """Worker for parallel execution; returns (name, rep_idx, result)."""
    name, rep_idx, pypy, script, env, extra, n, warmup_iters = task
    return (name, rep_idx, run_one(pypy, script, env, extra, n, warmup_iters))


def _run_one_task_multi(task):
    """Worker for cross-binary parallel execution; returns (pypy, name, rep_idx, result)."""
    pypy, name, rep_idx, script, env, extra, n, warmup_iters = task
    return (pypy, name, rep_idx,
            run_one(pypy, script, env, extra, n, warmup_iters))


def run_all(pypy, names, reps, n, warmup_iters=10, jobs=1):
    by_name = dict((b[0], b) for b in BENCHMARKS)
    valid_names = []
    for name in names:
        if name not in by_name:
            print("[skip] %s (not in BENCHMARKS list)" % name, file=sys.stderr)
            continue
        valid_names.append(name)
    samples_per = dict((name, []) for name in valid_names)

    if jobs > 1:
        tasks = []
        for name in valid_names:
            _, script, env, extra = by_name[name]
            for i in range(reps):
                tasks.append((name, i, pypy, script, env, extra, n,
                              warmup_iters))
        print("[parallel] %d tasks across %d workers" % (len(tasks), jobs))
        sys.stdout.flush()
        from multiprocessing import Pool
        pool = Pool(processes=jobs)
        try:
            for (name, rep_idx, r) in pool.imap_unordered(_run_one_task,
                                                          tasks):
                if "error" in r:
                    print("[%s] rep %d ERROR: %s" %
                          (name, rep_idx, r["error"]))
                else:
                    samples_per[name].append(r)
                    print("[%s] rep %d  trace=%.4fs  run=%.4fs  "
                          "jit_total=%.4fs  wall=%.3fs" %
                          (name, rep_idx, r["tracing"], r["running"],
                           r["jit_total"], r["wall"]))
                sys.stdout.flush()
        finally:
            pool.close()
            pool.join()
    else:
        for name in valid_names:
            _, script, env, extra = by_name[name]
            print("[%s] %d reps..." % (name, reps))
            sys.stdout.flush()
            for i in range(reps):
                r = run_one(pypy, script, env, extra, n, warmup_iters)
                if "error" in r:
                    print("  rep %d ERROR: %s" % (i, r["error"]))
                    sys.stdout.flush()
                    continue
                samples_per[name].append(r)
                print("  rep %d  trace=%.4fs  run=%.4fs  jit_total=%.4fs  "
                      "wall=%.3fs" %
                      (i, r["tracing"], r["running"], r["jit_total"],
                       r["wall"]))
                sys.stdout.flush()

    raw = {}
    summary = {}
    for name in valid_names:
        samples = samples_per[name]
        raw[name] = samples
        if samples:
            summary[name] = _build_summary(samples)
    return raw, summary


def run_all_binaries(pypys, names, reps, n, warmup_iters=10, jobs=1):
    """Run all (binary, benchmark, rep) tasks in a single shared pool; returns {pypy: (raw, summary)}."""
    by_name = dict((b[0], b) for b in BENCHMARKS)
    valid_names = []
    for name in names:
        if name not in by_name:
            print("[skip] %s (not in BENCHMARKS list)" % name, file=sys.stderr)
            continue
        valid_names.append(name)

    samples_per = {}
    for pypy in pypys:
        samples_per[pypy] = dict((name, []) for name in valid_names)

    tasks = []
    for pypy in pypys:
        for name in valid_names:
            _, script, env, extra = by_name[name]
            for i in range(reps):
                tasks.append((pypy, name, i, script, env, extra, n,
                              warmup_iters))

    print("[parallel-binaries] %d tasks across %d binaries, %d workers" %
          (len(tasks), len(pypys), jobs))
    sys.stdout.flush()
    from multiprocessing import Pool
    pool = Pool(processes=jobs)
    try:
        for (pypy, name, rep_idx, r) in pool.imap_unordered(
                _run_one_task_multi, tasks):
            tag = _binary_short_name(pypy)
            if "error" in r:
                print("[%s|%s] rep %d ERROR: %s" %
                      (tag, name, rep_idx, r["error"]))
            else:
                samples_per[pypy][name].append(r)
                print("[%s|%s] rep %d  trace=%.4fs  run=%.4fs  "
                      "jit_total=%.4fs  wall=%.3fs" %
                      (tag, name, rep_idx, r["tracing"], r["running"],
                       r["jit_total"], r["wall"]))
            sys.stdout.flush()
    finally:
        pool.close()
        pool.join()

    results = {}
    for pypy in pypys:
        raw = {}
        summary = {}
        for name in valid_names:
            samples = samples_per[pypy][name]
            raw[name] = samples
            if samples:
                t_med, t_min, t_max, n_t = aggregate(samples, "tracing")
                r_med, r_min, r_max, n_r = aggregate(samples, "running")
                j_med, j_min, j_max, n_j = aggregate(samples, "jit_total")
                w_med, w_min, w_max, n_w = aggregate(samples, "wall")
                summary[name] = {
                    "tracing_med": t_med, "tracing_min": t_min,
                    "tracing_max": t_max,
                    "running_med": r_med, "running_min": r_min,
                    "running_max": r_max,
                    "jit_total_med": j_med, "jit_total_min": j_min,
                    "wall_med": w_med, "wall_min": w_min,
                    "n": n_t,
                }
        results[pypy] = (raw, summary)
    return results


def _binary_short_name(pypy_path):
    """Strip directory + trailing `-c` to get a short tag for output naming."""
    name = os.path.basename(pypy_path)
    if name.endswith("-c"):
        name = name[:-2]
    return name


def cmd_run_one(pypy, out_path, args):
    """Run all benchmarks against a single binary; write JSON to out_path."""
    pypy = os.path.abspath(pypy)
    if not os.access(pypy, os.X_OK):
        sys.stderr.write("not executable: %s\n" % pypy)
        sys.exit(2)
    if args.benchmarks:
        names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    else:
        names = [b[0] for b in BENCHMARKS]
    raw, summary = run_all(pypy, names, args.reps, args.n,
                           args.warmup_iters, jobs=args.jobs)
    out = {
        "pypy": pypy, "reps": args.reps, "n": args.n,
        "warmup_iters": args.warmup_iters,
        "jobs": args.jobs,
        "benchmarks": sorted(summary.keys()),
        "raw": raw, "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote %s" % out_path)
    print_summary_table(out)
    return out_path


def cmd_run(args):
    """Dispatch single- or multi-binary run mode."""
    if args.hbp:
        pypy = os.path.abspath(args.pypy[0] if args.pypy
                               else "./pypy/goal/pypy-c")
        if not os.access(pypy, os.X_OK):
            sys.stderr.write("not executable: %s\n" % pypy); sys.exit(2)
        if args.benchmarks:
            names = [b.strip() for b in args.benchmarks.split(",")
                     if b.strip()]
        else:
            names = [b[0] for b in BENCHMARKS]
        print("\n%s\nS20 stable interleaved A/B (same binary, JIT policy):\n"
              "  binary    = %s\n  baseline  = --jit %s\n  hbp       = "
              "--jit %s\nreps=%d n=%d break_even=%s prewarm=%s\n%s" %
              ("=" * 70, pypy, args.jit_baseline or "(stock defaults)",
               args.jit_hbp, args.reps, args.n, args.break_even,
               not args.no_prewarm, "=" * 70))
        res = run_hbp_interleaved(pypy, args.jit_baseline, args.jit_hbp,
                                  names, args.reps, args.n,
                                  args.warmup_iters,
                                  break_even=args.break_even,
                                  prewarm=not args.no_prewarm)
        out_paths = ["e2e_baseline.json", "e2e_hbp.json"]
        for lbl, op in (("baseline", out_paths[0]), ("hbp", out_paths[1])):
            raw, summary = res[lbl]
            jp = args.jit_baseline if lbl == "baseline" else args.jit_hbp
            with open(op, "w") as f:
                json.dump({"pypy": pypy, "policy": lbl, "jit_params": jp,
                           "reps": args.reps, "n": args.n,
                           "warmup_iters": args.warmup_iters,
                           "interleaved": True,
                           "benchmarks": sorted(summary.keys()),
                           "raw": raw, "summary": summary}, f, indent=2)
            print("\nWrote %s" % op)
            print_summary_table(json.load(open(op)))
        print("\n%s\nComparison: baseline vs hbp\n%s\n" %
              ("=" * 70, "=" * 70))
        _compare_two_paths(out_paths[0], out_paths[1])
        return
    if args.all:
        base = os.path.abspath("./pypy/goal/pypy-c")
        cand = os.path.abspath("./pypy/goal/pypy-jit-ext-c")
        for p in (base, cand):
            if not os.access(p, os.X_OK):
                sys.stderr.write("not executable: %s\n" % p); sys.exit(2)
        if args.benchmarks:
            names = [b.strip() for b in args.benchmarks.split(",")
                     if b.strip()]
        else:
            names = [b[0] for b in BENCHMARKS]
        print("\n%s\nS20 stable interleaved A/B: baseline=%s candidate=%s\n"
              "reps=%d n=%d break_even=%s prewarm=%s\n%s" %
              ("=" * 70, base, cand, args.reps, args.n,
               args.break_even, not args.no_prewarm, "=" * 70))
        res = run_all_interleaved(base, cand, names, args.reps, args.n,
                                  args.warmup_iters,
                                  break_even=args.break_even,
                                  prewarm=not args.no_prewarm)
        out_paths = ["e2e_baseline.json", "e2e_proposed.json"]
        for who, op in ((base, out_paths[0]), (cand, out_paths[1])):
            raw, summary = res[who]
            with open(op, "w") as f:
                json.dump({"pypy": who, "reps": args.reps, "n": args.n,
                           "warmup_iters": args.warmup_iters,
                           "interleaved": True,
                           "benchmarks": sorted(summary.keys()),
                           "raw": raw, "summary": summary}, f, indent=2)
            print("\nWrote %s" % op)
            print_summary_table(json.load(open(op)))
        print("\n%s\nComparison: %s vs %s\n%s\n" %
              ("=" * 70, out_paths[0], out_paths[1], "=" * 70))
        _compare_two_paths(out_paths[0], out_paths[1])
        return
    if False:
        pypys = []
        out_paths = []
    else:
        pypys = list(args.pypy)
        if len(pypys) == 1 and args.out:
            out_paths = [args.out]
        else:
            prefix = args.out or "e2e"
            out_paths = ["%s_%s.json" % (prefix, _binary_short_name(p))
                         for p in pypys]

    if args.parallel_binaries and len(pypys) > 1:
        pypys_abs = []
        for pypy in pypys:
            p = os.path.abspath(pypy)
            if not os.access(p, os.X_OK):
                sys.stderr.write("not executable: %s\n" % p)
                sys.exit(2)
            pypys_abs.append(p)
        if args.benchmarks:
            names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        else:
            names = [b[0] for b in BENCHMARKS]
        print("\n%s" % ("=" * 70))
        print("Bench %d binaries in parallel (--parallel-binaries)" %
              len(pypys_abs))
        for p, op in zip(pypys_abs, out_paths):
            print("  %s -> %s" % (p, op))
        print("=" * 70)
        results = run_all_binaries(pypys_abs, names, args.reps, args.n,
                                   args.warmup_iters, jobs=args.jobs)
        for pypy, out_path in zip(pypys_abs, out_paths):
            raw, summary = results[pypy]
            out = {
                "pypy": pypy, "reps": args.reps, "n": args.n,
                "warmup_iters": args.warmup_iters,
                "jobs": args.jobs,
                "parallel_binaries": True,
                "benchmarks": sorted(summary.keys()),
                "raw": raw, "summary": summary,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print("\nWrote %s" % out_path)
            print_summary_table(out)
    else:
        for i, (pypy, out_path) in enumerate(zip(pypys, out_paths)):
            print("\n%s" % ("=" * 70))
            print("Bench %d/%d: %s -> %s" % (i + 1, len(pypys), pypy, out_path))
            print("=" * 70)
            cmd_run_one(pypy, out_path, args)

    # When exactly two binaries were benched, print a head-to-head
    # comparison automatically.  Treat the first as baseline.
    if len(out_paths) == 2:
        print("\n%s\nComparison: %s vs %s\n%s\n" % (
            "=" * 70, out_paths[0], out_paths[1], "=" * 70))
        _compare_two_paths(out_paths[0], out_paths[1])


def print_summary_table(out):
    # Two co-equal metric families (S20): WARMUP (JIT-self-measured
    # transient cost) | STEADY-STATE (stable per-iter perf, '*' = no
    # genuine steady state reached -> steady value not trustworthy).
    print("\n%-20s | WARMUP: %7s %7s %7s %8s %-9s %6s %6s | STEADY: %10s %6s"
          % ("benchmark", "trace", "opt", "backnd", "jit_tot",
             "class", "loops", "bridg", "perf", "cv%"))
    for name in sorted(out["summary"].keys()):
        s = out["summary"][name]
        rel = "" if s.get("steady_reliable", True) else "*"
        cv = s.get("steady_perf_cv", float("nan"))
        print("%-20s |         %7.4f %7.4f %7.4f %8.4f %-9s %6s %6s | "
              "%10.5f%1s %5.1f" %
              (name, s["tracing_med"], s.get("optimization_med", float("nan")),
               s.get("backend_med", float("nan")), s["jit_total_med"],
               s.get("warmup_class", "n/a"),
               s.get("loops_med", -1), s.get("bridges_med", -1),
               s.get("steady_perf_med", float("nan")), rel,
               (cv * 100.0) if cv == cv else float("nan")))
    print("  ('*' on STEADY = no genuine steady state reached for the "
          "majority of reps; that steady number is not trustworthy.)")


def cmd_compare(args):
    return _compare_two_paths(args.compare[0], args.compare[1])


def _compare_two_paths(baseline_path, optim_path):
    base = json.load(open(baseline_path))
    opt = json.load(open(optim_path))
    common = sorted(set(base["summary"]) & set(opt["summary"]))

    def pct(b, x):
        if b == 0 or x == 0 or b != b or x != x:
            return float("nan")
        return (b - x) / b * 100.0

    print("Baseline: %s (reps=%d, n=%d, warmup=%d)" %
          (base.get("pypy"), base.get("reps"), base.get("n"), base.get("warmup_iters", 1)))
    print("Optim:    %s (reps=%d, n=%d, warmup=%d)" %
          (opt.get("pypy"), opt.get("reps"), opt.get("n"), opt.get("warmup_iters", 1)))
    print("Common benchmarks: %d" % len(common))

    print("\n%-26s  %12s %12s %+8s   %12s %12s %+8s   %12s %12s %+8s" %
          ("benchmark",
           "trace_b", "trace_o", "Δtrace%",
           "run_b",   "run_o",   "Δrun%",
           "jit_tot_b","jit_tot_o","Δjit%"))
    deltas_trace = []
    deltas_running = []
    deltas_jit_total = []
    for b in common:
        sb = base["summary"][b]; so = opt["summary"][b]
        dtr = pct(sb["tracing_med"], so["tracing_med"])
        drun = pct(sb["running_med"], so["running_med"])
        djt = pct(sb["jit_total_med"], so["jit_total_med"])
        if dtr == dtr: deltas_trace.append((b, dtr))
        if drun == drun: deltas_running.append((b, drun))
        if djt == djt: deltas_jit_total.append((b, djt))
        print("%-26s  %12.4f %12.4f %+7.2f%%   %12.4f %12.4f %+7.2f%%   %12.4f %12.4f %+7.2f%%"
              % (b,
                 sb["tracing_med"], so["tracing_med"], dtr,
                 sb["running_med"], so["running_med"], drun,
                 sb["jit_total_med"], so["jit_total_med"], djt))

    import math

    def _summarize(label, deltas, key):
        if not deltas:
            return
        s = 0.0; c = 0
        for b, _ in deltas:
            sb = base["summary"][b][key]
            so = opt["summary"][b][key]
            if sb > 0 and so > 0:
                s += math.log(sb / so); c += 1
        gm = (math.exp(s / c) - 1.0) * 100.0 if c else 0.0
        med = median([d for _, d in deltas])
        n_pos = sum(1 for _, d in deltas if d > 0)
        n_neg = sum(1 for _, d in deltas if d < 0)
        print("\n=== %s ===" % label)
        print("  geomean speedup: %+.2f%%" % gm)
        print("  median delta:    %+.2f%%" % med)
        print("  wins: %d, regressions: %d, total: %d" %
              (n_pos, n_neg, len(deltas)))
        for b, d in sorted(deltas, key=lambda x: -x[1])[:5]:
            print("  +%-7.2f%% (win)        %s" % (d, b))
        for b, d in sorted(deltas, key=lambda x: x[1])[:5]:
            print("  %-7.2f%% (regression) %s" % (d, b))

    def _deltas_for(key):
        ds = []
        for b in common:
            sb = base["summary"][b].get(key)
            so = opt["summary"][b].get(key)
            if isinstance(sb, (int, float)) and isinstance(so, (int, float)):
                d = pct(sb, so)
                if d == d:
                    ds.append((b, d))
        return ds

    # S20 PRIMARY metrics first (JIT-self-measured, low-noise):
    _summarize("TRACING TIME (S20 primary)", deltas_trace, "tracing_med")
    _summarize("OPTIMIZATION TIME (S20 primary)",
               _deltas_for("optimization_med"), "optimization_med")
    _summarize("BACKEND CODEGEN TIME (S20 primary)",
               _deltas_for("backend_med"), "backend_med")
    _summarize("JIT TOTAL (S20 primary)", deltas_jit_total, "jit_total_med")
    # Structural counts (deterministic-ish drivers; +% = candidate has fewer):
    for ck in ("loops_med", "bridges_med", "guards_med", "abort_total_med",
               "slow_tracing_exec_med", "fast_tracing_exec_med"):
        _summarize("COUNT " + ck, _deltas_for(ck), ck)
    # Warmup-class shift (how many benchmarks each VM gets to steady):
    from collections import Counter
    cb = Counter(base["summary"][b].get("warmup_class", "n/a") for b in common)
    co = Counter(opt["summary"][b].get("warmup_class", "n/a") for b in common)
    print("\n=== WARMUP CLASS (baseline -> optim), benches=%d ===" % len(common))
    print("  baseline: %s" % dict(cb))
    print("  optim:    %s" % dict(co))
    # ---- STEADY-STATE (STABLE) PERFORMANCE: co-equal, distinct metric ----
    # Aggregated ONLY over benchmarks that reached a genuine steady state
    # for BOTH binaries (Barrett et al.: steady perf is undefined for
    # no_steady series). Excluded benches are reported, not silently
    # folded in (that conflation is part of what caused the S12/S14
    # over-claims).
    steady_common = [b for b in common
                     if base["summary"][b].get("steady_reliable", True)
                     and opt["summary"][b].get("steady_reliable", True)]
    excluded = [b for b in common if b not in steady_common]
    steady_deltas = []
    for b in steady_common:
        d = pct(base["summary"][b].get("steady_perf_med"),
                opt["summary"][b].get("steady_perf_med"))
        if d == d:
            steady_deltas.append((b, d))
    print("\n=== STEADY-STATE (STABLE) PERFORMANCE "
          "[reliable benches: %d/%d] ===" % (len(steady_common), len(common)))
    if excluded:
        print("  excluded (no genuine steady state for >=1 binary): %s"
              % ", ".join(excluded))
    _summarize("STEADY-STATE per-iter time", steady_deltas, "steady_perf_med")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pypy", action="append", default=[],
                    help=("path to a pypy binary; repeat to bench multiple "
                          "binaries sequentially.  When exactly two binaries "
                          "are benched, a head-to-head comparison is "
                          "printed automatically (first treated as baseline)."))
    ap.add_argument("--all", action="store_true",
                    help=("shortcut for benching `./pypy/goal/pypy-c` "
                          "(baseline) followed by "
                          "`./pypy/goal/pypy-jit-ext-c` (proposed); writes "
                          "e2e_baseline.json and e2e_proposed.json and "
                          "prints the comparison."))
    ap.add_argument("--hbp", action="store_true",
                    help=("HBP-vs-baseline mode: interleaved S20 A/B of a "
                          "SINGLE binary under two --jit policies. Uses "
                          "the first --pypy (or ./pypy/goal/pypy-c). "
                          "Writes e2e_baseline.json + e2e_hbp.json and "
                          "prints the comparison. Tune the two arms with "
                          "--jit-baseline / --jit-hbp."))
    ap.add_argument("--jit-baseline", default="",
                    help=("--jit param string for the baseline arm of "
                          "--hbp mode (default: empty = stock JIT "
                          "defaults, HBP off)."))
    ap.add_argument("--jit-hbp",
                    default="enable_hot_bridge_promotion=1,"
                            "enable_tracetree=1",
                    help=("--jit param string for the HBP arm of --hbp "
                          "mode (default: "
                          "enable_hot_bridge_promotion=1,"
                          "enable_tracetree=1)."))
    ap.add_argument("--break-even", action="store_true",
                    help=("also run each benchmark with `--jit off` and "
                          "report the cumulative JIT-on-vs-interpreter "
                          "break-even iteration (S20 robust warmup scalar; "
                          "doubles run cost)."))
    ap.add_argument("--no-prewarm", action="store_true",
                    help=("skip the untimed pre-warm run before each "
                          "measured run (S20 hygiene; default: prewarm on)."))
    ap.add_argument("--reps", type=int, default=10,
                    help="reps per (binary, benchmark) (default: 10)")
    ap.add_argument("-n", type=int, default=60,
                    help="-n value passed to each benchmark (default: 60)")
    ap.add_argument("--warmup-iters", type=int, default=30,
                    help="iterations to drop as warmup (default: 50)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help=("run benchmark processes in parallel with N "
                          "workers (default: 1 = sequential).  N>1 "
                          "trades measurement noise for wall-clock speed; "
                          "use 0 to default to all available CPUs."))
    ap.add_argument("--parallel-binaries", action="store_true",
                    help=("when multiple --pypy binaries are passed, run "
                          "their (binary, benchmark, rep) tasks in a single "
                          "shared pool so different binaries can execute "
                          "side by side.  Honors --jobs as the total worker "
                          "count.  Adds more measurement noise than serial "
                          "per-binary execution; use --jobs 1 with this off "
                          "for decision-grade signal."))
    ap.add_argument("--out", help=("output JSON path; with multiple --pypy "
                                   "binaries, treated as a prefix and each "
                                   "binary writes <prefix>_<name>.json"))
    ap.add_argument("--benchmarks", default="",
                    help="comma-separated subset (default: all in list)")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "OPT"),
                    help="compare two JSON outputs already on disk")
    args = ap.parse_args()
    if args.compare:
        return cmd_compare(args)
    if not (args.all or args.hbp or args.pypy):
        ap.error("one of --pypy / --all / --hbp / --compare required")
    if (not args.all and not args.hbp and len(args.pypy) == 1
            and not args.out):
        ap.error("--out required when --pypy is given exactly once")
    if args.jobs == 0:
        import multiprocessing
        args.jobs = multiprocessing.cpu_count()
    if args.jobs < 0:
        ap.error("--jobs must be >= 0")
    return cmd_run(args)


if __name__ == "__main__":
    main()
