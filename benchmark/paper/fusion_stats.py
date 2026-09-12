#!/usr/bin/env python3
"""Mechanism accounting: what the fusion pass actually built, per model.

One run per model, with three things turned on at once:

  PYPYLOG=jit-log-opt:<log>   the optimized traces, where every fusion region
                              shows up as one residual tensor_launch call
  RTENSOR_KERNEL_CACHE=0      every kernel that is compiled writes its Triton
  TMPDIR=<fresh dir>          IR next to its name, rtensor_k<n>.ttir
  RTENSOR_STATS=1             the model script prints the kernel-name counter
                              either side of the forward, which is the <n>
                              range belonging to this model rather than to the
                              start-up single-op kernels

The kernel struct is a ConstPtr in the trace, so the trace cannot say how big
a region is; the TTIR can (it is generated from the very same KERNEL object).
The trace in turn is the only place that says what cut a region short.

Writes $OUT/fusion.tsv.  See README, "Fusion statistics".
"""

import argparse
import collections
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "applevel")

# model -> (applevel script, weights dir, extra argv), as in run_models.sh
MODELS = collections.OrderedDict([
    ("tiny-gpt2",    ("gpt2.py", "tiny-gpt2", [])),
    ("bert-tiny",    ("bert.py", "bert-tiny", [])),
    ("bert-mini",    ("bert.py", "bert-mini", [])),
    ("distilgpt2",   ("gpt2.py", "distilgpt2", [])),
    ("vit-tiny",     ("vit.py", "vit-tiny", [])),
    ("mixer_b16",    ("mixer.py", "mixer_b16", [])),
    ("smollm2-135m", ("llama.py", "smollm2-135m", [])),
    ("resnet18-b1",  ("resnet.py", "resnet18", ["1"])),
])

COLUMNS = ["model", "kernels", "launches_per_iter", "nodes_min", "nodes_median",
           "nodes_max", "extra_outputs_total", "forced_library", "forced_item",
           "forced_loop", "forced_leafcap", "forced_assign", "forced_other"]

# --- the TTIR side: one file per compiled kernel ---------------------------

def parse_ttir(path):
    """(inputs, folded constants, DAG nodes, extra outputs, body signature).

    to_ttir names value %v<k>: the first ninputs are the loaded inputs, then
    one per folded scalar constant, then one per KERNEL node in order, so the
    highest %v index gives the node count.  The signature is every line that
    defines a value, which is what makes two compiles of the same region (the
    pass recompiles a kernel when a node escapes as an extra output) one
    kernel again.
    """
    src = open(path).read()
    sig = re.search(r"tt\.func public @\w+\(([^)]*)\)", src)
    if not sig:
        return None
    nin = len(re.findall(r"%in\d+:", sig.group(1)))
    nextra = len(re.findall(r"%out\d+:", sig.group(1)))
    nconst = len(re.findall(r"%v\d+ = arith\.constant dense<", src))
    vs = [int(v) for v in re.findall(r"%v(\d+)", src)]
    if not vs:
        return None
    body = "\n".join(l.strip() for l in src.splitlines()
                     if re.match(r"\s*%v\d+ = ", l))
    return nin, nconst, max(vs) + 1 - nin - nconst, nextra, body


def kernels_of(tmpdir, lo, hi):
    """The distinct fusion kernels compiled between the two counter reads."""
    by_body = collections.OrderedDict()
    for n in range(lo, hi):
        path = os.path.join(tmpdir, "rtensor_k%d.ttir" % n)
        if not os.path.exists(path):
            continue  # a gather kernel (rtensor_g<n>), not a fused region
        k = parse_ttir(path)
        if k is None:
            continue
        prev = by_body.get(k[4])
        if prev is None or k[3] > prev[3]:
            by_body[k[4]] = k
    return list(by_body.values())

# --- the trace side: why each region ended ---------------------------------

OP = re.compile(r"^\s*[+\-]?\d*:?\s*(?:(\w+)\s*=\s*)?(\w+)\((.*)\)\s*$")
FUSABLE = r"_ll_\d+_tensor_(add|mul|relu|sum|relugrad|sub|div|exp|sqrt|maxr|eqmask)__"
REASON = [
    # The library kernels are plain C names, with and without a tensor_
    # prefix (im2col_nhwc, maxpool2_nhwc, rowgather, ...).
    ("library", r"tensor_(matmul|bmm|conv|embed|transpose|pad|repeat|slice)"
                r"|im2col|col2chw|maxpool|head_split|head_merge|gather"),
    ("item",    r"tensor_(item|host|download|flat|to_list|getitem|value)"),
    ("assign",  r"tensor_assign"),
    ("leafcap", r"tensor_launch|" + FUSABLE),
]
# A forced chain whose value is stored into a heap object, handed to the next
# iteration or left live at the end of the trace has escaped the region the
# pass could see; that is the loop/guard boundary.
ESCAPE = ("jump", "finish", "label", "setfield_gc", "setarrayitem_gc",
          "setinteriorfield_gc", "call_may_force", "call_release_gil")


def trace_ops(path):
    """Yield one list of (result, opname, args) per optimized trace."""
    cur = None
    for line in open(path, errors="replace"):
        if line.startswith("[") and "jit-log-opt" in line:
            if "{jit-log-opt" in line:
                cur = []
            elif cur is not None:
                yield cur
                cur = None
            continue
        if cur is None:
            continue
        m = OP.match(line)
        if m:
            cur.append((m.group(1), m.group(2), m.group(3)))


def force_reasons(path, unknown):
    """Count, over every compiled trace, what first consumed each launch."""
    counts = collections.Counter()
    for ops in trace_ops(path):
        for i, (res, name, args) in enumerate(ops):
            if "tensor_launch" not in args or not res:
                continue
            use = re.compile(r"\b%s\b" % res)
            consumer = None
            for r2, n2, a2 in ops[i + 1:]:
                if not use.search(a2):
                    continue
                # Reading extra output k of this very kernel is not what cut
                # it; keep looking for the consumer that did.
                if "tensor_output" in a2:
                    continue
                consumer = (n2, a2)
                break
            if consumer is None:
                counts["loop"] += 1
                continue
            n2, a2 = consumer
            if n2 in ESCAPE:
                counts["loop"] += 1
                continue
            for reason, pat in REASON:
                if re.search(pat, a2):
                    counts[reason] += 1
                    break
            else:
                counts["other"] += 1
                fn = re.search(r"ConstClass\((\w+)\)", a2)
                unknown[fn.group(1) if fn else n2] += 1
    return counts

# --- running ---------------------------------------------------------------

def run_model(model, args, unknown):
    script, weights, extra = MODELS[model]
    tmpdir = os.path.join(args.tmp, model)
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir)
    log = os.path.join(args.tmp, "fusion-%s.log" % model)
    env = dict(os.environ,
               TMPDIR=tmpdir,
               RTENSOR_KERNEL_CACHE="0",
               RTENSOR_STATS="1",
               PYPYLOG="jit-log-opt:" + log)
    cmd = ([args.pypy] + args.jit_flags.split() +
           [os.path.join(APP, script), os.path.join(args.weights, weights),
            str(args.iters), str(args.warmup)] + extra)
    print("== %s" % model, flush=True)
    out = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                         universal_newlines=True).stdout
    window = re.search(r"kernel_count_begin=(\d+) kernel_count_end=(\d+)", out)
    if not window:
        print("%s: no kernel_count window in the output, skipped:\n%s"
              % (model, out[-400:]), file=sys.stderr)
        return None
    lo, hi = int(window.group(1)), int(window.group(2))
    launches = re.search(r"launches_per_iter=([\d.]+)", out)
    ks = kernels_of(tmpdir, lo, hi)
    if not ks:
        print("%s: no kernels compiled in [%d,%d)" % (model, lo, hi),
              file=sys.stderr)
        return None
    nodes = sorted(k[2] for k in ks)
    r = force_reasons(log, unknown)
    return [model, len(ks), launches.group(1) if launches else "",
            nodes[0], "%g" % statistics.median(nodes), nodes[-1],
            sum(k[3] for k in ks),
            r["library"], r["item"], r["loop"], r["leafcap"], r["assign"],
            r["other"]]


SELFTEST_TTIR = """module {
  tt.func public @rtensor_k7(%in0: !tt.ptr<f32>, %in1: !tt.ptr<f32>, \
%out: !tt.ptr<f32>, %out0: !tt.ptr<f32>, %n: i64, %c: i64) {
    %zero = arith.constant dense<0.0> : tensor<8xf32>
    %v0 = tt.load %q0, %mask, %zero : tensor<8xf32>
    %v1 = tt.load %q1, %mask, %zero : tensor<8xf32>
    %v2 = arith.constant dense<0.5> : tensor<8xf32>
    %v3 = arith.addf %v0, %v1 : tensor<8xf32>
    %v4 = arith.mulf %v3, %v2 : tensor<8xf32>
  }
}
"""

SELFTEST_LOG = """[abc] {jit-log-opt-loop
# Loop 0
+10: p1 = call_r(ConstClass(_ll_9_tensor_launch__X), ConstPtr(p0), p9, descr=d)
+20: p2 = call_r(ConstClass(_ll_2_tensor_output__X), p1, 0, descr=d)
+30: p3 = call_r(ConstClass(tensor_matmul), p1, p4, descr=d)
+40: p5 = call_r(ConstClass(_ll_9_tensor_launch__X), ConstPtr(p6), p3, descr=d)
+50: setfield_gc(p7, p5, descr=<FieldP Tensor.inst_t 32>)
[abc] jit-log-opt-loop}
"""


def selftest(tmp):
    path = os.path.join(tmp, "rtensor_k7.ttir")
    open(path, "w").write(SELFTEST_TTIR)
    assert parse_ttir(path)[:4] == (2, 1, 2, 1), parse_ttir(path)[:4]
    assert len(kernels_of(tmp, 0, 9)) == 1
    log = os.path.join(tmp, "t.log")
    open(log, "w").write(SELFTEST_LOG)
    unknown = collections.Counter()
    r = force_reasons(log, unknown)
    assert not unknown and r["library"] == 1 and r["loop"] == 1, r
    print("selftest ok")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default=os.environ.get("OUT"))
    p.add_argument("models", nargs="*", default=[])
    p.add_argument("--pypy", default=os.environ.get(
        "PYPY", os.path.join(HERE, "build", "pypy-c")))
    p.add_argument("--weights", default=os.environ.get(
        "WEIGHTS", os.path.join(HERE, "weights")))
    p.add_argument("--jit-flags", default=os.environ.get("JIT_FLAGS", ""))
    p.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", 200)))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", 30)))
    p.add_argument("--tmp", default=os.path.join(
        tempfile.gettempdir(), "fusion-stats"),
        help="where the traces and the per-model TTIR dumps go; they are "
             "megabytes per model, so by default they stay out of $OUT")
    p.add_argument("--selftest", action="store_true",
                   help="check the two parsers on a sample and exit")
    args = p.parse_args(argv)
    if args.selftest:
        selftest(tempfile.mkdtemp())
        return 0
    if not args.out:
        p.error("no result directory given and $OUT is unset")
    os.makedirs(args.tmp, exist_ok=True)
    for m in args.models:
        if m not in MODELS:
            p.error("unknown model %r, valid: %s" % (m, " ".join(MODELS)))

    unknown = collections.Counter()
    rows = [r for r in (run_model(m, args, unknown)
                        for m in (args.models or MODELS)) if r]
    path = os.path.join(args.out, "fusion.tsv")
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    if unknown:
        print("unattributed consumers (counted as forced_other): %s"
              % ", ".join("%s=%d" % kv for kv in unknown.most_common()),
              file=sys.stderr)
    print("wrote %s" % path)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
