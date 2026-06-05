#!/usr/bin/env python3
"""End-to-end warmup + steady-state A/B benchmark harness for pypy-jit-ext-c.

Each benchmark is run as matched baseline/candidate pairs, reporting the
JIT-self-measured warmup phase decomposition, changepoint-derived warmup
scalars, and a steady-state regression guard. Comparisons are paired
(per-rep log-ratios) because each rep includes both binaries. The default
pair order alternates A/B then B/A to avoid fixed-order bias.

Examples:
  bench_e2e.py --all --preset genext --reps 10 -n 50 --warmup-iters 0
  bench_e2e.py --pypy ./pypy/goal/pypy-c --out base.json --preset smoke
  bench_e2e.py --compare base.json proposed.json
"""
from __future__ import print_function
import argparse
import json
import math
import os
import random
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


PRESETS = {
    "genext": [
        # symbolic math: canonical heavy slow-path tracing
        "sympy_expand", "sympy_integrate", "sympy_str", "sympy_sum",
        # parsers / serialization (strong tracing + resume signal)
        "html5lib", "json_bench",
        # templating (very large resume-data signal)
        "genshi_text", "pyxl_bench",
        # web / ORM real workloads
        "django", "sqlalchemy_imperative",
        # misc real workloads with strong tracing wins
        "bm_dulwich_log", "spambayes", "chaos", "telco",
        # regression-watch: families that showed steady-state regressions
        # under GenExtension (ORM / serialization / templating).  Included so
        # A/B runs surface the whack-a-mole regressors a partial set hides:
        # tuning that recovers one regressor (e.g. sympy_integrate via a
        # higher function_threshold) tends to surface a fresh regression in
        # the ORM/serialization families, which a winner-heavy set never sees.
        "sqlalchemy_declarative", "unpickle", "genshi_xml",
    ],
    # Regression-focused working subset for cheap iteration: the confirmed
    # steady-state regressors + their collateral + representative winners and
    # high-variance sentinels, balanced so a quick `--benchmarks`/`--preset
    # genext_reg` sweep predicts the full-genext geomean.  Use this instead of
    # an ad-hoc winner-heavy subset, which over-predicted a candidate's gain
    # by omitting sqlalchemy_imperative (the collateral regressor).
    "genext_reg": [
        # original premature-compile regressors (recovered by decay=80)
        "sympy_sum", "sympy_integrate",
        # collateral regressors (surface when inlining/threshold is tuned)
        "sqlalchemy_imperative", "json_bench", "bm_dulwich_log",
        # sibling regression-watch
        "sqlalchemy_declarative", "unpickle",
        # reliable winners: geomean anchors
        "pyxl_bench", "spambayes",
        # mids / high-variance sentinels
        "telco", "chaos", "sympy_str",
    ],
    # Fast sanity check: 3 substantial-but-quicker benches, signal only.
    "smoke": ["sympy_expand", "chaos", "html5lib"],
}


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
    res = {"tracing": float("nan"), "resume_data": float("nan"),
           "optimization": float("nan"), "backend": float("nan"),
           "jit_total": float("nan"), "resume_data_count": -1}
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
        elif s.startswith("Resume data:"):
            # indented sub-line of Tracing: "<count>\t<secs>"
            nums = re.findall(_FLOAT, s.split(":", 1)[1])
            if len(nums) >= 2:
                res["resume_data_count"] = int(float(nums[0]))
                res["resume_data"] = float(nums[1])
            elif nums:
                res["resume_data"] = float(nums[-1])
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
    start_tail = n - len(tail)
    # Reject an early plateau whose suffix is still slower than the true
    # tail (common when warmup has a temporary stable band well above the
    # eventual asymptote, e.g. spambayes with genext's faster early tracing).
    if start >= 0 and start < start_tail:
        suffix_med = median(t[start:])
        if suffix_med > smed * 1.02:
            start = start_tail
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
    break-even point (robust scalar). -1 if not reached within the
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
    """Deterministic, low-noise child env : fixed hash seed for
    dict-order determinism across process executions; single-thread."""
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


_HAVE_CAFFEINATE = None


def _caffeinated(cmd):
    """Wrap cmd in macOS `caffeinate -dimsu` so no idle/display/disk sleep or
    App Nap perturbs the timed run. No-op on non-Darwin or if absent.

    Linux may ship an unrelated `caffeinate` (e.g. cups-of-caffeine); never
    wrap there."""
    global _HAVE_CAFFEINATE
    if sys.platform != "darwin":
        return cmd
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


def _wrapped_child_cmd(cmd, cpu=None):
    if cpu is not None:
        if sys.platform.startswith("linux"):
            cmd = ["taskset", "-c", str(cpu)] + cmd
        else:
            print("[bench_e2e] warning: --cpu is only implemented via "
                  "taskset on Linux; ignoring on %s" % sys.platform,
                  file=sys.stderr)
    return _caffeinated(cmd)


def run_one(pypy, script, env_extra, extra_args, n, warmup_iters=10,
            jit_off=False, jit_log=True, cpu=None):
    """Run a single (binary, benchmark) once. Returns dict of metrics.

    Stable-setting hygiene: deterministic child env and a caffeinate
    wrap. The full per-iteration series is kept (warmup is NOT dropped);
    analyze_warmup() derives the warmup scalars and `running` is the
    *detected* steady suffix (regression guard only)."""
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
    jit_args = ["--jit", "off"] if jit_off else []
    # Runtime JIT-param override sweep hook: BENCH_JIT_PARAMS="threshold=768,..."
    # injects `--jit <params>` so build-time PARAMETERS defaults can be probed
    # without a retranslation.  Ignored in jit-off (break-even) runs.
    _jp = os.environ.get("BENCH_JIT_PARAMS")
    # Apply only to the genext binary so a paired A/B keeps vanilla pristine.
    if _jp and not jit_off and "jit-ext" in pypy:
        jit_args = ["--jit", _jp]
    base_cmd = ([pypy] + jit_args
                + [os.path.join(BENCH_DIR, script), "-n", str(n)]
                + list(extra_args))

    def _exec():
        tmp = None
        if jit_log:
            tmp = tempfile.NamedTemporaryFile(prefix="pypylog_e2e_",
                                              suffix=".txt", delete=False)
            tmp.close()
        e = dict(env)
        if jit_log:
            e["PYPYLOG"] = "jit-summary:%s" % tmp.name
        else:
            e.pop("PYPYLOG", None)
        t0 = time.perf_counter()
        try:
            p = subprocess.Popen(_wrapped_child_cmd(base_cmd, cpu=cpu),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=e)
            stdout, _ = p.communicate()
            rc = p.returncode
        except OSError as ex:
            if tmp is not None:
                try: os.unlink(tmp.name)
                except OSError: pass
            return None, None, None, ("OSError: %s" % ex)
        wall = time.perf_counter() - t0
        logtxt = ""
        if tmp is not None:
            try:
                with open(tmp.name) as f:
                    logtxt = f.read()
            finally:
                try: os.unlink(tmp.name)
                except OSError: pass
        if rc != 0:
            return None, None, wall, ("rc=%d" % rc)
        return stdout, logtxt, wall, None

    stdout, log, wall, err = _exec()
    if err is not None:
        return {"error": err, "wall": wall}
    summary = parse_jit_summary(log)
    iter_times = parse_iter_times(stdout.decode())
    # Warmup is NOT dropped: keep the full series and let the changepoint
    # detector locate the steady suffix.
    wm = analyze_warmup(iter_times)
    steady_i = wm["steady_start_iter"]
    # Steady-state regression guard: use the asymptotic tail (same window
    # as analyze_warmup's band), not the full suffix from an early
    # changepoint.  Warmup scalars still come from steady_i above.
    n_iters = len(iter_times)
    tail_len = max(4, n_iters // 3) if n_iters else 0
    steady_tail = (iter_times[-tail_len:] if tail_len else [])
    if not steady_tail:
        if steady_i >= 0:
            steady_tail = iter_times[steady_i:]
        else:
            steady_tail = (iter_times[warmup_iters:]
                           if len(iter_times) > warmup_iters else iter_times)
    running_med = median(steady_tail) if steady_tail else float("nan")
    running_sum = sum(steady_tail) if steady_tail else float("nan")
    out = {
        "wall": wall,
        # primary (JIT-self-measured) warmup metrics:
        "tracing": summary["tracing"],
        "resume_data": summary["resume_data"],
        "optimization": summary["optimization"],
        "backend": summary["backend"],
        "jit_total": summary["jit_total"],
        # warmup-curve scalars (changepoint-derived, not warmup_tax):
        "warmup_class": wm["classification"],
        "steady_start_iter": wm["steady_start_iter"],
        "wall_to_steady": wm["wall_to_steady"],
        "speedup_vs_first": wm["speedup_vs_first"],
        "n_iters": wm["n_iters"],
        # steady-state regression guard (secondary, NOT the headline):
        "running": running_med,
        "running_sum": running_sum,
        "iter_times": iter_times,
    }
    for k in ("ops", "recorded_ops", "guards", "opt_guards", "loops",
              "bridges", "abort_total", "fast_tracing_exec",
              "slow_tracing_exec", "resume_data_count"):
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

    bench_e2e supports TWO co-equal, distinct measurements:
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
    iter_counts = [s.get("n_iters", len(s.get("iter_times", [])))
                   for s in samples]
    iter_med = median(iter_counts) if iter_counts else 0
    run_metric = "cold" if iter_med < 8 else "steady"
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
    agg_tracing = aggregate(samples, "tracing")
    agg_resume = aggregate(samples, "resume_data")
    agg_jit = aggregate(samples, "jit_total")
    d = {
        # ---- WARMUP performance (transient cost) ----
        "tracing_med": agg_tracing[0],
        "tracing_min": agg_tracing[1],
        "resume_data_med": agg_resume[0],
        "resume_data_min": agg_resume[1],
        "optimization_med": aggregate(samples, "optimization")[0],
        "backend_med": aggregate(samples, "backend")[0],
        "jit_total_med": agg_jit[0],
        "jit_total_min": agg_jit[1],
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
        "run_metric": run_metric,
        "n_iters_med": iter_med,
        # back-compat alias (older report code / external readers):
        "running_med": run_med,
        "running_min": (min(run_vals) if run_vals else float("nan")),
        "wall_med": aggregate(samples, "wall")[0],
        "n": agg_tracing[3],
    }
    for k in ("loops", "bridges", "guards", "abort_total",
              "fast_tracing_exec", "slow_tracing_exec",
              "resume_data_count"):
        d[k + "_med"] = aggregate(samples, k)[0]
    return d


def _thermal_guard(label, cooldown=8, max_wait=120):
    """stable-setting: if the CPU is thermally throttled, idle until
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
                        break_even=False, jit_log=True, cpu=None,
                        cooldown=0.0, order="alternating", seed=None):
    """Stable-setting comparison: for each (benchmark, rep) run one baseline
    and one candidate process as a matched pair.  The pair order is
    alternating by default (A,B then B,A) so fixed-order cache/thermal/DVFS
    effects do not systematically favor either binary.  Serial only (jobs=1)
    for decision-grade signal.  Returns {pypy: (raw, summary)}."""
    by_name = dict((b[0], b) for b in BENCHMARKS)
    valid = [nm for nm in names if nm in by_name]
    sp = {base: dict((nm, []) for nm in valid),
          cand: dict((nm, []) for nm in valid)}
    rng = random.Random(seed)
    first_child = True

    def _run_with_cooldown(who, script, env, extra, jit_off=False):
        nonlocal first_child
        if not first_child and cooldown > 0.0:
            time.sleep(cooldown)
        first_child = False
        return run_one(who, script, env, extra, n, warmup_iters,
                       jit_off=jit_off, jit_log=jit_log, cpu=cpu)

    for nm in valid:
        _, script, env, extra = by_name[nm]
        print("[%s] %d reps (interleaved baseline/candidate)" % (nm, reps))
        sys.stdout.flush()
        be_cache = {}
        if break_even:
            for who in (base, cand):
                ro = _run_with_cooldown(who, script, env, extra,
                                        jit_off=True)
                be_cache[who] = (ro.get("iter_times")
                                 if "error" not in ro else None)
        for i in range(reps):
            if order == "fixed":
                pair = (base, cand)
            elif order == "random":
                pair = [base, cand]
                rng.shuffle(pair)
            else:
                pair = (base, cand) if i % 2 == 0 else (cand, base)
            for who in pair:
                thr = _thermal_guard("%s/%s" % (nm, os.path.basename(who)))
                r = _run_with_cooldown(who, script, env, extra)
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
                print("  [%s] rep %d %-18s trace=%.4f resume=%.4f "
                      "opt=%.4f back=%.4f run=%.5f jit=%.4f %s" %
                      (nm, i, os.path.basename(who), r["tracing"],
                       r.get("resume_data", float("nan")),
                       r.get("optimization", float("nan")),
                       r.get("backend", float("nan")), r["running"],
                       r["jit_total"], r["warmup_class"]))
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


def _run_one_task(task):
    """Worker for parallel execution; returns (name, rep_idx, result)."""
    name, rep_idx, pypy, script, env, extra, n, warmup_iters, jit_log, cpu = task
    return (name, rep_idx, run_one(pypy, script, env, extra, n, warmup_iters,
                                   jit_log=jit_log, cpu=cpu))


def _run_one_task_multi(task):
    """Worker for cross-binary parallel execution; returns (pypy, name, rep_idx, result)."""
    pypy, name, rep_idx, script, env, extra, n, warmup_iters, jit_log, cpu = task
    return (pypy, name, rep_idx,
            run_one(pypy, script, env, extra, n, warmup_iters,
                    jit_log=jit_log, cpu=cpu))


def run_all(pypy, names, reps, n, warmup_iters=10, jobs=1,
            jit_log=True, cpu=None, cooldown=0.0):
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
                              warmup_iters, jit_log, cpu))
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
                    print("[%s] rep %d  trace=%.4f resume=%.4f opt=%.4f "
                          "back=%.4f run=%.5f jit=%.4f" %
                          (name, rep_idx, r["tracing"],
                           r.get("resume_data", float("nan")),
                           r.get("optimization", float("nan")),
                           r.get("backend", float("nan")),
                           r["running"], r["jit_total"]))
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
                if i > 0 and cooldown > 0.0:
                    time.sleep(cooldown)
                r = run_one(pypy, script, env, extra, n, warmup_iters,
                            jit_log=jit_log, cpu=cpu)
                if "error" in r:
                    print("  rep %d ERROR: %s" % (i, r["error"]))
                    sys.stdout.flush()
                    continue
                samples_per[name].append(r)
                print("  rep %d  trace=%.4f resume=%.4f opt=%.4f "
                      "back=%.4f run=%.5f jit=%.4f" %
                      (i, r["tracing"],
                       r.get("resume_data", float("nan")),
                       r.get("optimization", float("nan")),
                       r.get("backend", float("nan")),
                       r["running"], r["jit_total"]))
                sys.stdout.flush()

    raw = {}
    summary = {}
    for name in valid_names:
        samples = samples_per[name]
        raw[name] = samples
        if samples:
            summary[name] = _build_summary(samples)
    return raw, summary


def run_all_binaries(pypys, names, reps, n, warmup_iters=10, jobs=1,
                     jit_log=True, cpu=None):
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
                              warmup_iters, jit_log, cpu))

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
                print("[%s|%s] rep %d  trace=%.4f resume=%.4f opt=%.4f "
                      "back=%.4f run=%.5f jit=%.4f" %
                      (tag, name, rep_idx, r["tracing"],
                       r.get("resume_data", float("nan")),
                       r.get("optimization", float("nan")),
                       r.get("backend", float("nan")),
                       r["running"], r["jit_total"]))
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


def _pypy_version(pypy_path):
    try:
        out = subprocess.check_output([pypy_path, "--version"],
                                      stderr=subprocess.STDOUT)
        return out.decode("utf-8", "replace").strip()
    except Exception as ex:
        return "unknown: %s" % (ex,)


def _resolve_names(args):
    """The explicit --benchmarks subset, else every name in BENCHMARKS."""
    if args.benchmarks:
        return [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    return [b[0] for b in BENCHMARKS]


def _write_interleaved(res, base, cand, out_paths, args):
    """Write the two per-binary JSONs from an interleaved A/B run and
    print each summary table."""
    for who, op in ((base, out_paths[0]), (cand, out_paths[1])):
        raw, summary = res[who]
        with open(op, "w") as f:
            json.dump({"pypy": who, "reps": args.reps, "n": args.n,
                       "pypy_version": _pypy_version(who),
                       "warmup_iters": args.warmup_iters,
                       "jit_log": args.jit_log,
                       "cpu": args.cpu,
                       "cooldown": args.cooldown,
                       "order": args.order,
                       "interleaved": True,
                       "benchmarks": sorted(summary.keys()),
                       "raw": raw, "summary": summary}, f, indent=2)
        print("\nWrote %s" % op)
        print_summary_table(json.load(open(op)))


def cmd_run_one(pypy, out_path, args):
    """Run all benchmarks against a single binary; write JSON to out_path."""
    pypy = os.path.abspath(pypy)
    if not os.access(pypy, os.X_OK):
        sys.stderr.write("not executable: %s\n" % pypy)
        sys.exit(2)
    names = _resolve_names(args)
    raw, summary = run_all(pypy, names, args.reps, args.n,
                           args.warmup_iters, jobs=args.jobs,
                           jit_log=args.jit_log, cpu=args.cpu,
                           cooldown=args.cooldown)
    out = {
        "pypy": pypy, "reps": args.reps, "n": args.n,
        "pypy_version": _pypy_version(pypy),
        "warmup_iters": args.warmup_iters,
        "jit_log": args.jit_log,
        "cpu": args.cpu,
        "cooldown": args.cooldown,
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
    if args.all:
        base = os.path.abspath("./pypy/goal/pypy-c")
        cand = os.path.abspath("./pypy/goal/pypy-jit-ext-c")
        for p in (base, cand):
            if not os.access(p, os.X_OK):
                sys.stderr.write("not executable: %s\n" % p); sys.exit(2)
        names = _resolve_names(args)
        print("\n%s\nstable interleaved A/B: baseline=%s candidate=%s\n"
              "reps=%d n=%d break_even=%s jit_log=%s order=%s cpu=%s cooldown=%.2f\n%s" %
              ("=" * 70, base, cand, args.reps, args.n,
               args.break_even, args.jit_log, args.order, args.cpu,
               args.cooldown, "=" * 70))
        res = run_all_interleaved(base, cand, names, args.reps, args.n,
                                  args.warmup_iters,
                                  break_even=args.break_even,
                                  jit_log=args.jit_log, cpu=args.cpu,
                                  cooldown=args.cooldown,
                                  order=args.order, seed=args.seed)
        out_paths = ["e2e_baseline.json", "e2e_proposed.json"]
        _write_interleaved(res, base, cand, out_paths, args)
        print("\n%s\nComparison: %s vs %s\n%s\n" %
              ("=" * 70, out_paths[0], out_paths[1], "=" * 70))
        _compare_two_paths(out_paths[0], out_paths[1])
        return

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
        names = _resolve_names(args)
        print("\n%s" % ("=" * 70))
        print("Bench %d binaries in parallel (--parallel-binaries)" %
              len(pypys_abs))
        for p, op in zip(pypys_abs, out_paths):
            print("  %s -> %s" % (p, op))
        print("=" * 70)
        results = run_all_binaries(pypys_abs, names, args.reps, args.n,
                                   args.warmup_iters, jobs=args.jobs,
                                   jit_log=args.jit_log, cpu=args.cpu)
        for pypy, out_path in zip(pypys_abs, out_paths):
            raw, summary = results[pypy]
            out = {
                "pypy": pypy, "reps": args.reps, "n": args.n,
                "pypy_version": _pypy_version(pypy),
                "warmup_iters": args.warmup_iters,
                "jit_log": args.jit_log,
                "cpu": args.cpu,
                "jobs": args.jobs,
                "parallel_binaries": True,
                "benchmarks": sorted(summary.keys()),
                "raw": raw, "summary": summary,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print("\nWrote %s" % out_path)
            print_summary_table(out)
    elif len(pypys) == 2:
        # Exactly two binaries: run matched A/B pairs.  The default pair
        # order alternates A,B then B,A so fixed-order thermal/cache effects
        # do not systematically favor one binary.
        base = os.path.abspath(pypys[0])
        cand = os.path.abspath(pypys[1])
        for p in (base, cand):
            if not os.access(p, os.X_OK):
                sys.stderr.write("not executable: %s\n" % p); sys.exit(2)
        names = _resolve_names(args)
        print("\n%s\nstable interleaved A/B: baseline=%s candidate=%s\n"
              "reps=%d n=%d break_even=%s jit_log=%s order=%s cpu=%s cooldown=%.2f\n%s" %
              ("=" * 70, base, cand, args.reps, args.n,
               args.break_even, args.jit_log, args.order, args.cpu,
               args.cooldown, "=" * 70))
        res = run_all_interleaved(base, cand, names, args.reps, args.n,
                                  args.warmup_iters,
                                  break_even=args.break_even,
                                  jit_log=args.jit_log, cpu=args.cpu,
                                  cooldown=args.cooldown,
                                  order=args.order, seed=args.seed)
        _write_interleaved(res, base, cand, out_paths, args)
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
    # Two co-equal metric families: WARMUP (JIT-self-measured
    # transient cost) | STEADY-STATE (stable per-iter perf, '*' = no
    # genuine steady state reached -> steady value not trustworthy).
    run_label = "COLD" if out.get("n", 0) < 8 else "STEADY"
    print("\n%-20s | WARMUP: %7s %7s %7s %7s %8s %-9s %6s %6s | "
          "%-6s: %10s %6s"
          % ("benchmark", "trace", "resume", "opt", "backnd", "jit_tot",
             "class", "loops", "bridg", run_label, "perf", "cv%"))
    for name in sorted(out["summary"].keys()):
        s = out["summary"][name]
        rel = "" if s.get("steady_reliable", True) else "*"
        cv = s.get("steady_perf_cv", float("nan"))
        print("%-20s |         %7.4f %7.4f %7.4f %7.4f %8.4f %-9s %6s %6s | "
              "%10.5f%1s %5.1f" %
              (name, s["tracing_med"],
               s.get("resume_data_med", float("nan")),
               s.get("optimization_med", float("nan")),
               s.get("backend_med", float("nan")), s["jit_total_med"],
               s.get("warmup_class", "n/a"),
               s.get("loops_med", -1), s.get("bridges_med", -1),
               s.get("steady_perf_med", float("nan")), rel,
               (cv * 100.0) if cv == cv else float("nan")))
    if run_label == "COLD":
        print("  (COLD = one/few measured iterations; this is cold latency, "
              "not steady-state machine-code speed.)")
    else:
        print("  ('*' on STEADY = no genuine steady state reached for the "
              "majority of reps; that steady number is not trustworthy.)")


def cmd_compare(args):
    return _compare_two_paths(args.compare[0], args.compare[1])


def _compare_two_paths(baseline_path, optim_path):
    base = json.load(open(baseline_path))
    opt = json.load(open(optim_path))
    common = sorted(set(base["summary"]) & set(opt["summary"]))
    cold_run = min(base.get("n", 0), opt.get("n", 0)) < 8
    jit_log_available = bool(base.get("jit_log", True) and
                             opt.get("jit_log", True))

    def pct(b, x):
        if b == 0 or x == 0 or b != b or x != x:
            return float("nan")
        return (b - x) / b * 100.0

    print("Baseline: %s (reps=%d, n=%d, warmup=%d)" %
          (base.get("pypy"), base.get("reps"), base.get("n"), base.get("warmup_iters", 1)))
    print("Optim:    %s (reps=%d, n=%d, warmup=%d)" %
          (opt.get("pypy"), opt.get("reps"), opt.get("n"), opt.get("warmup_iters", 1)))
    bv = base.get("pypy_version")
    ov = opt.get("pypy_version")
    if bv and ov and bv != ov:
        print("[warn] binaries report different --version strings:")
        print("       baseline: %s" % bv.replace("\n", " | "))
        print("       optim:    %s" % ov.replace("\n", " | "))
    if cold_run:
        print("[note] n<8: RUNNING is cold iteration latency, not steady-state speed.")
    if not jit_log_available:
        print("[note] jit_log=false: JIT phase metrics are unavailable for this comparison.")
    print("Common benchmarks: %d" % len(common))

    # ---- Paired computation -------------------------------------------
    # The harness runs one baseline and one optim sample per rep
    # (run_all_interleaved), so raw[bench][i] is a MATCHED pair taken
    # close together under similar thermal/load state.  The correct effect is
    # therefore paired, not a comparison of two marginal medians:
    #   - point estimate = geomean of per-rep ratios base_i/opt_i
    #     (+ve % => optim faster), scale-robust;
    #   - significance    = paired t on the log-ratios + a sign-consistency
    #     gate (robust to a couple of heavy-tailed outlier reps);
    #   - a delta is a win/regression ONLY if it clears that paired test.
    # Marginal-median deltas are kept solely as a fallback when a JSON has
    # no per-rep raw[] (e.g. --parallel-binaries summary-only output).
    SIG_T = 2.0   # |t| gate; with the >=70% sign gate this is ~p<.05 at n~10
    base_raw = base.get("raw", {}) or {}
    opt_raw = opt.get("raw", {}) or {}

    def _steady_running(sample):
        """Asymptotic per-iter time from iter_times (tail window)."""
        t = sample.get("iter_times")
        if not t:
            v = sample.get("running")
            return v if isinstance(v, (int, float)) and v == v else None
        n = len(t)
        tail_len = max(4, n // 3)
        tail = t[-tail_len:]
        return median(tail) if tail else float("nan")

    def _series(raw, b, rk):
        out = []
        for s in (raw.get(b) or []):
            if rk == "running":
                v = _steady_running(s)
            else:
                v = s.get(rk)
            out.append(v if isinstance(v, (int, float)) and v == v else None)
        return out

    def _paired(b, rk):
        xb = _series(base_raw, b, rk)
        xo = _series(opt_raw, b, rk)
        n = min(len(xb), len(xo))
        L = []
        for i in range(n):
            a, c = xb[i], xo[i]
            if a is None or c is None or a <= 0 or c <= 0:
                continue
            L.append(math.log(a / c))          # >0 => optim faster
        m = len(L)
        if m == 0:
            return None
        mean = sum(L) / m
        if m >= 3:
            var = sum((x - mean) ** 2 for x in L) / (m - 1)
            se = (var ** 0.5) / math.sqrt(m)
            t = mean / se if se > 0 else float("nan")
        else:
            se = float("nan"); t = float("nan")
        npos = sum(1 for x in L if x > 0)
        nneg = sum(1 for x in L if x < 0)
        eff = (math.exp(mean) - 1.0) * 100.0   # geomean speedup %
        band = (math.exp(2.0 * se) - 1.0) * 100.0 if se == se else float("nan")
        return {"n": m, "eff": eff, "t": t, "npos": npos, "nneg": nneg,
                "band": band, "mean": mean}

    # raw per-rep key, table header, summary-median key (median fallback)
    METRICS = [
        ("tracing",      "Δtrace%",  "tracing_med",      "TRACING TIME"),
        ("resume_data",  "Δresume%", "resume_data_med",  "RESUME DATA TIME"),
        ("optimization", "Δopt%",    "optimization_med", "OPTIMIZATION TIME"),
        ("backend",      "Δback%",   "backend_med",      "BACKEND CODEGEN TIME"),
        ("running",      "Δrun%",    "running_med",
         "COLD ITERATION TIME" if cold_run else "RUNNING TIME"),
        ("jit_total",    "Δjit%",    "jit_total_med",    "JIT TOTAL"),
    ]
    if not jit_log_available:
        METRICS = [m for m in METRICS if m[0] == "running"]
    MED_KEY = dict((rk, mk) for rk, _h, mk, _l in METRICS)
    P = {}
    for rk, _h, _mk, _lbl in METRICS:
        P[rk] = dict((b, _paired(b, rk)) for b in common)

    have_raw = any(P[rk][b] for rk, _h, _mk, _l in METRICS for b in common)
    if not have_raw:
        print("  [warn] no per-rep raw[] in JSON -> falling back to "
              "marginal-median deltas (paired test unavailable)")

    def _classify(b, rk, reliability_guard):
        """-> ('win'|'reg'|'noise', eff%, t).  noise = effect does not
        clear the paired test, or (steady running) an unreliable steady arm."""
        ps = P[rk].get(b)
        if ps is None:                       # no raw -> marginal fallback
            mk = MED_KEY[rk]
            sb = base["summary"][b].get(mk)
            so = opt["summary"][b].get(mk)
            d = pct(sb, so) if isinstance(sb, (int, float)) and \
                isinstance(so, (int, float)) else float("nan")
            return ("noise", d, float("nan"))
        if reliability_guard and not cold_run:
            sb = base["summary"].get(b, {}); so = opt["summary"].get(b, {})
            if not (sb.get("steady_reliable", True) and
                    so.get("steady_reliable", True)):
                return ("noise", ps["eff"], ps["t"])
        tot = ps["npos"] + ps["nneg"]
        lopsided = tot > 0 and max(ps["npos"], ps["nneg"]) >= 0.7 * tot
        sig = (ps["t"] == ps["t"] and abs(ps["t"]) >= SIG_T and lopsided)
        if not sig:
            return ("noise", ps["eff"], ps["t"])
        return (("win" if ps["eff"] > 0 else "reg"), ps["eff"], ps["t"])

    # ---- Per-bench table (paired geomean %, '*' = paired-significant) --
    print("\n%-24s %8s %8s %8s %8s %8s %8s" %
          ("benchmark", "Δtrace%", "Δresume%", "Δopt%",
           "Δback%", "Δrun%", "Δjit%"))

    def _cell(b, rk):
        if rk not in P:
            return "   n/a "
        kind, eff, _t = _classify(b, rk, rk == "running")
        if eff != eff:
            return "   n/a "
        return "%+6.2f%s" % (eff, "*" if kind in ("win", "reg") else " ")
    for b in common:
        print("%-24s %8s %8s %8s %8s %8s %8s" %
              (b, _cell(b, "tracing"), _cell(b, "resume_data"),
               _cell(b, "optimization"), _cell(b, "backend"),
               _cell(b, "running"), _cell(b, "jit_total")))
    print("  ('*' = paired-significant: |t|>=%.1f on per-rep log-ratios "
          "AND >=70%% reps agree in sign; unmarked = within paired noise)"
          % SIG_T)

    def _summarize(rk, label):
        items = [(b, _classify(b, rk, rk == "running")) for b in common]
        # Aggregate over benches from the per-bench effect (paired geomean
        # %, or marginal % in the no-raw fallback) so both paths report a
        # meaningful headline rather than nan.
        effs = [e for _b, (_k, e, _t) in items if e == e]
        gm = ((math.exp(sum(math.log(max(1e-9, 1.0 + e / 100.0))
                            for e in effs) / len(effs)) - 1.0) * 100.0) \
            if effs else float("nan")
        med = median(effs) if effs else float("nan")
        wins = [(b, e, t) for b, (k, e, t) in items if k == "win"]
        regs = [(b, e, t) for b, (k, e, t) in items if k == "reg"]
        noise = [(b, e, t) for b, (k, e, t) in items if k == "noise"]
        print("\n=== %s ===" % label)
        print("  geomean speedup (paired): %+.2f%%" % gm)
        print("  median delta (paired):    %+.2f%%" % med)
        print("  wins: %d, regressions: %d, within-noise: %d, total: %d" %
              (len(wins), len(regs), len(noise), len(items)))
        print("  (win/regression = paired-significant only; "
              "within-noise = effect does not clear the paired test)")
        for b, e, t in sorted(wins, key=lambda x: -x[1])[:5]:
            print("  %+7.2f%%  t=%+5.2f  (win)        %s" % (e, t, b))
        for b, e, t in sorted(regs, key=lambda x: x[1])[:5]:
            print("  %+7.2f%%  t=%+5.2f  (regression) %s" % (e, t, b))
        for b, e, t in sorted(noise, key=lambda x: x[1])[:6]:
            ts = ("t=%+5.2f" % t) if t == t else "t= n/a"
            print("  %+7.2f%%  %s  (noise)      %s" % (e, ts, b))

    for rk, _h, _mk, lbl in METRICS:
        _summarize(rk, lbl)

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
    ap.add_argument("--break-even", action="store_true",
                    help=("also run each benchmark with `--jit off` and "
                          "report the cumulative JIT-on-vs-interpreter "
                          "break-even iteration (robust warmup scalar; "
                          "doubles run cost)."))
    ap.add_argument("--no-jit-log", dest="jit_log", action="store_false",
                    default=True,
                    help=("disable PYPYLOG=jit-summary. Use this for "
                          "steady-state/runtime measurements; JIT phase "
                          "columns will be NaN."))
    ap.add_argument("--order", choices=("alternating", "random", "fixed"),
                    default="alternating",
                    help=("A/B order for two-binary runs. alternating runs "
                          "AB,BA,AB...; random shuffles each pair; fixed "
                          "keeps the old baseline-then-candidate order."))
    ap.add_argument("--seed", type=int,
                    help="random seed used by --order=random")
    ap.add_argument("--cpu", type=int,
                    help="pin child benchmark processes to this Linux CPU via taskset")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep between child benchmark processes")
    ap.add_argument("--reps", type=int, default=10,
                    help="reps per (binary, benchmark) (default: 10)")
    ap.add_argument("-n", type=int, default=50,
                    help="-n value passed to each benchmark (default: 50)")
    ap.add_argument("--warmup-iters", type=int, default=0,
                    help="iterations to drop as warmup (default: 0)")
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
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help=("curated benchmark group (%s); ignored if "
                          "--benchmarks is given." %
                          ", ".join(sorted(PRESETS))))
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "OPT"),
                    help="compare two JSON outputs already on disk")
    args = ap.parse_args()
    if args.compare:
        return cmd_compare(args)
    # --benchmarks (explicit) wins over --preset; otherwise expand the
    # preset into the same comma-separated form every run path reads.
    if not args.benchmarks and args.preset:
        args.benchmarks = ",".join(PRESETS[args.preset])
    if args.preset == "genext" and args.n < 30:
        print("[bench_e2e] warning: -n %d is short for genext A/B; "
              "use -n 50+ so warmup/steady metrics stabilize." % args.n,
              file=sys.stderr)
    if not (args.all or args.pypy):
        ap.error("one of --pypy / --all / --compare required")
    if not args.all and len(args.pypy) == 1 and not args.out:
        ap.error("--out required when --pypy is given exactly once")
    if args.jobs == 0:
        import multiprocessing
        args.jobs = multiprocessing.cpu_count()
    if args.jobs < 0:
        ap.error("--jobs must be >= 0")
    return cmd_run(args)


if __name__ == "__main__":
    main()
