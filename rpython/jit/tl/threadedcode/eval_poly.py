#!/usr/bin/env python3
"""Measure the tier2 (residual, type-agnostic) vs tier3 (inlined, type-specialised)
difference on monomorphic vs type-polymorphic workloads.

For each (benchmark, tier) it runs ./targettla-c under PYPYLOG=jit-summary and
collects:
  loops, bridges            -- number of compiled traces (specialisation count)
  opt_ops                   -- total optimised resops  (code-size proxy)
  guards                    -- optimised guard count   (type/branch checks)
  backend_s                 -- backend/codegen time     (compile-cost proxy)
  nvirtuals                 -- objects kept virtual (unboxing)
  steady_us                 -- median of the last half of the per-iter times
  result                    -- last stdout line (correctness)

Usage: python3 eval_poly.py BENCH[:X[:N]] ...    (defaults X=40 N=400)
"""
import os, re, sys, subprocess, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
BIN  = os.path.join(HERE, "targettla-c")

def _grep(text, pat, group=1, cast=float, default=None):
    m = re.search(pat, text, re.M)
    return cast(m.group(group)) if m else default

def run(bench, tier, x, n):
    tlc = os.path.join(HERE, "lang", bench + ".tlc")
    summ = "/tmp/poly_%s_t%d.jitlog" % (bench, tier)
    env = dict(os.environ, PYPYLOG="jit-summary:" + summ)
    try:
        out = subprocess.run([BIN, "--tier", str(tier), tlc, str(x), str(n)],
                             env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "<timeout>"}
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if not lines or "Fatal RPython error" in out.stderr:
        err = out.stderr.strip().splitlines()
        return {"status": "<crash:%s>" % (err[-1] if err else "?")}
    result = lines[-1]
    times = []
    for l in lines[:-1]:
        try: times.append(float(l))
        except ValueError: pass
    steady = None
    if times:
        half = times[len(times)//2:] or times
        steady = statistics.median(half) * 1e6   # microseconds
    s = open(summ).read() if os.path.exists(summ) else ""
    return {
        "status": "ok",
        "result": result,
        "loops":   _grep(s, r"Total # of loops:\s+(\d+)", cast=int, default=0),
        "bridges": _grep(s, r"Total # of bridges:\s+(\d+)", cast=int, default=0),
        "opt_ops": _grep(s, r"^opt ops:\s+(\d+)", cast=int, default=0),
        "guards":  _grep(s, r"^opt guards:\s+(\d+)", cast=int, default=0),
        "backend_s": _grep(s, r"Backend:\s+\d+\s+([0-9.]+)", default=0.0),
        "trace_s":   _grep(s, r"Tracing:\s+\d+\s+([0-9.]+)", default=0.0),
        "nvirt":   _grep(s, r"^nvirtuals:\s+(\d+)", cast=int, default=0),
        "steady_us": steady,
    }

def main(argv):
    specs = argv or ["mono_rec", "poly_rec", "float_rec"]
    hdr = ("bench", "tier", "loops", "brdg", "optOps", "guards",
           "back_ms", "trace_ms", "nvirt", "steady_us", "result")
    print("%-12s %-4s %-5s %-5s %-7s %-6s %-8s %-8s %-6s %-10s %-8s" % hdr)
    rows = []
    for spec in specs:
        parts = spec.split(":")
        bench = parts[0]
        x = int(parts[1]) if len(parts) > 1 else 40
        n = int(parts[2]) if len(parts) > 2 else 400
        for tier in (2, 3, 4):
            r = run(bench, tier, x, n)
            if r["status"] != "ok":
                print("%-12s %-4d %s" % (bench, tier, r["status"]))
                rows.append((bench, tier, r["status"]))
                continue
            print("%-12s %-4d %-5d %-5d %-7d %-6d %-8.3f %-8.3f %-6d %-10s %-8s" % (
                bench, tier, r["loops"], r["bridges"], r["opt_ops"], r["guards"],
                r["backend_s"]*1e3, r["trace_s"]*1e3, r["nvirt"],
                ("%.2f" % r["steady_us"]) if r["steady_us"] is not None else "-",
                r["result"]))
            rows.append((bench, tier, r))
    return rows

if __name__ == "__main__":
    main(sys.argv[1:])
