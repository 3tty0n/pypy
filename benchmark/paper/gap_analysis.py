"""Where the JAX gap comes from: launches, fusion granularity, GPU utilisation.

    benchmark/paper/bench.sh gap            # writes $OUT/gap.tsv
    benchmark/paper/gap_analysis.py $OUT [MODEL...]

Three systems run the same five small transformers, and each is measured twice:
once plainly, for the steady-state wall time per forward, and twice under nsys
CUDA tracing, at two iteration counts.  The difference between the two traced
runs is the per-forward cost with setup, warm-up and compilation subtracted -
one measurement path for all three systems, so the numbers are comparable
without trusting any system's own accounting.

Nsys names every kernel, which is what makes the fusion axis readable: how many
launches a forward costs, how many distinct kernels those launches come from,
and how they split between cuBLAS GEMMs, generated (Triton/XLA) kernels and
copies.  gpu_util = gpu_busy_us / steady_us is what is left: the fraction of
the wall clock the GPU is actually running a kernel, the rest being dispatch.

This file is also its own worker.  The MetaTensor rows run under pypy-c, whose
interpreter is Python 2, so the whole file stays 2-and-3 compatible.
"""

from __future__ import print_function

import csv
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "applevel")

# model -> (pypy module, torch script, jax_models.py name, weights dir)
MODELS = [
    ("bert-tiny", "bert", "bert_torch.py", "bert", "bert-tiny"),
    ("bert-mini", "bert", "bert_torch.py", "bert", "bert-mini"),
    ("tiny-gpt2", "gpt2", "gpt2_torch.py", "gpt2", "tiny-gpt2"),
    ("distilgpt2", "gpt2", "gpt2_torch.py", "gpt2", "distilgpt2"),
    ("vit-tiny", "vit", "vit_torch.py", "vit", "vit-tiny"),
]

SYSTEMS = ["ours", "torch-compile", "jax"]

# The two traced iteration counts.  Their difference is the per-forward cost;
# the low point still has to be large enough that a slow first steady iteration
# does not dominate it.
TRACE_LO, TRACE_HI, TRACE_WARM = 20, 120, 10

NSYS = os.environ.get("NSYS", "/usr/local/cuda-13.1/bin/nsys")


# --- worker (runs under pypy-c) ---------------------------------------------

def worker_ours(module, weights, iters, warmup):
    """The applevel model scripts do not print a launch count, so the timed
    loop is rebuilt here around their own build()."""
    sys.path.insert(0, APP)
    import time
    import common
    import _metatensor
    mod = __import__(module)
    cfg, buf = common.load(weights)
    # build() returns the model followed by its inputs, and how many inputs
    # there are is the model's business: gpt2 also hands back the position
    # embedding.
    built = mod.build(cfg, buf, os.environ.get("RTENSOR_DTYPE", "float32"))
    model, inputs = built[0], built[1:]
    out = model(*inputs)
    out.sum().item()
    for _ in range(warmup):
        out = model(*inputs)
    out.sum().item()
    launches = _metatensor.launch_count()
    t0 = time.time()
    for _ in range(iters):
        out = model(*inputs)
    out.sum().item()
    steady = (time.time() - t0) / iters * 1e6
    print("gap steady_us=%.1f launches_per_iter=%.1f kernels=%d" % (
        steady, float(_metatensor.launch_count() - launches) / iters,
        _metatensor.kernel_count()))


# --- running the three systems ----------------------------------------------

def command(system, model, iters, warmup):
    _, pymod, torchscript, jaxmodel, wdir = model
    weights = os.path.join(os.environ.get("WEIGHTS", os.path.join(HERE, "weights")), wdir)
    if system == "ours":
        pypy = os.environ.get("PYPY", os.path.join(HERE, "build", "pypy-c"))
        flags = os.environ.get("JIT_FLAGS", "").split()
        return [pypy] + flags + [os.path.abspath(__file__), "--worker-ours",
                                 pymod, weights, str(iters), str(warmup)]
    if system == "torch-compile":
        py = os.environ.get("TORCH_PYTHON") or os.environ.get("RTENSOR_PYTHON")
        if not py:
            return None
        return [py, os.path.join(APP, torchscript), "compile", weights,
                str(iters), str(warmup)]
    if system == "jax":
        py = os.environ.get("JAX_PYTHON")
        if not py or not jaxmodel:
            return None
        return [py, os.path.join(APP, "jax_models.py"), jaxmodel, "jax",
                weights, str(iters), str(warmup)]
    return None


def run(cmd):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = p.communicate()
        return out.decode("utf-8", "replace") if p.returncode == 0 else None
    except OSError:
        return None


def steady_of(text):
    m = re.search(r"steady_us=([0-9.]+)", text or "")
    return float(m.group(1)) if m else None


# --- nsys -------------------------------------------------------------------

def trace(cmd, tmp, tag):
    """One nsys CUDA trace; returns {kernel name: (instances, total ns)}."""
    rep = os.path.join(tmp, tag)
    # Without --cuda-graph-trace=node the kernels XLA and Inductor launch from
    # inside a CUDA graph are invisible: the trace then shows only compilation
    # and autotuning, and every steady-state count comes out zero.
    if run([NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node",
            "-o", rep, "--force-overwrite", "true"] + cmd) is None:
        return None
    counts = {}
    for report in ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum"):
        text = run([NSYS, "stats", "--report", report, "--format", "csv",
                    "--force-export", "true", rep + ".nsys-rep"])
        if text is None:
            continue
        for row in csv.reader(text.splitlines()):
            # Time(%), Total Time(ns), Instances/Count, ..., Name
            if len(row) < 4 or not re.match(r"^[0-9.]+$", row[0].strip()):
                continue
            try:
                counts[row[-1]] = (int(row[2]), int(row[1]))
            except ValueError:
                continue
    return counts or None


def category(name):
    low = name.lower()
    if "memcpy" in low or "memset" in low:
        return "copy"
    # cuBLAS finishes a split-K GEMM, and a fused epilogue, with a second
    # kernel of its own.  Those are kernels, not calls: one cublasSgemm can
    # show up here as two launches, so they are kept apart from the GEMMs
    # themselves - otherwise a launch count read as "calls the model made"
    # overcounts by however many tiles cuBLAS chose to split.
    if "splitkreduce" in low or "epilogue::impl::globalkernel" in low:
        return "splitk"
    if ("cublas" in low or "cutlass" in low or "gemm" in low
            or "gemv" in low or "dot_kernel" in low):
        return "gemm"
    if (low.startswith("rtensor_") or low.startswith("triton_")
            or "fusion" in low or "loop_" in low or "wrapped_" in low):
        return "gen"
    return "other"


# Inside the gemm bucket: the batched products an attention layer issues.
# Inductor lowers scaled_dot_product_attention to batched matmuls, and the
# question "how many of torch's GEMM launches are the SDPA ones" is answered
# by the kernel name where cuBLAS/cutlass says so - the batched entry points
# and the fused-attention kernels carry it.  A name that does not say is
# counted as a plain GEMM rather than guessed at, so this column is a lower
# bound, never an inflated one.
BMM_NAME = re.compile(r"batch|bmm|fmha|flash|attention|sdpa|mha", re.I)


def per_forward(lo, hi):
    """Per-forward launches, GPU time and kernel mix, as the difference between
    the two traced runs: everything that does not scale with the iteration
    count - upload, compilation, warm-up - cancels."""
    n = float(TRACE_HI - TRACE_LO)
    launches, ns, mix, kernels, bmm = 0.0, 0.0, {}, 0, 0.0
    per_kernel = []
    for name, (count, total) in hi.items():
        d = count - lo.get(name, (0, 0))[0]
        if d <= 0:
            continue
        kernels += 1
        launches += d / n
        us = (total - lo.get(name, (0, 0))[1]) / n / 1e3
        ns += (total - lo.get(name, (0, 0))[1]) / n
        cat = category(name)
        mix[cat] = mix.get(cat, 0.0) + d / n
        if cat == "gemm" and BMM_NAME.search(name):
            bmm += d / n
        per_kernel.append((cat, d / n, us, name))
    per_kernel.sort(key=lambda r: -r[1])
    return launches, ns / 1e3, kernels, mix, bmm, per_kernel


# --- driver -----------------------------------------------------------------

# The five class columns are the free-text `notes` mix as numbers, one column
# each, so a consumer does not have to parse a string; they sum to
# launches_per_iter by construction and `measure` says so if they ever do not.
# `bmm` is a sub-count of `gemm`, not a sixth class.
CLASSES = ["gemm", "splitk", "gen", "copy", "other"]
# Per-kernel rows for gap_kernels.tsv, filled by measure().
KERNEL_ROWS = []
KERNEL_COLUMNS = ["model", "system", "class", "launches_per_iter", "us_per_iter",
                  "kernel"]
# Appended after `notes` on purpose: an older gap.tsv's header is then a
# prefix of this one and can be upgraded in place, the way tsv_init does it.
COLUMNS = (["model", "system", "steady_us", "launches_per_iter", "kernels",
            "gpu_busy_us", "gpu_util", "notes"] + CLASSES + ["bmm"])


def measure(system, model, rounds, tmp):
    name = model[0]
    cmd = command(system, model, int(os.environ.get("ITERS", 200)),
                  int(os.environ.get("WARMUP", 30)))
    row = dict(model=name, system=system, steady_us="", launches_per_iter="",
               kernels="", gpu_busy_us="", gpu_util="", notes="", bmm="")
    for c in CLASSES:
        row[c] = ""
    if cmd is None:
        row["notes"] = "not configured"
        return row
    # The GPU is shared, so the rounds are interleaved by the caller and the
    # median is what survives a neighbour's burst.
    steadies = [s for s in rounds if s is not None]
    if not steadies:
        row["notes"] = "run failed"
        return row
    steadies.sort()
    steady = steadies[len(steadies) // 2]
    row["steady_us"] = "%.1f" % steady

    lo = trace(command(system, model, TRACE_LO, TRACE_WARM), tmp,
               "%s-%s-lo" % (name, system))
    hi = trace(command(system, model, TRACE_HI, TRACE_WARM), tmp,
               "%s-%s-hi" % (name, system))
    if not lo or not hi:
        row["notes"] = "nsys unavailable"
        return row
    launches, busy_us, kernels, mix, bmm, per_kernel = per_forward(lo, hi)
    # The evidence behind the class columns, one line per distinct kernel.
    # cuBLAS reuses the same kernel name for a batched and an unbatched
    # product on this GPU (torch's SDPA attention products are
    # ampere_sgemm_128x128_nn, exactly what a plain GEMM can also pick), so
    # "how many of these GEMMs are the attention ones" is answered by the
    # per-kernel counts here and not by any name pattern.
    for cat, count, us, kname in per_kernel:
        KERNEL_ROWS.append([name, system, cat, "%.2f" % count, "%.1f" % us,
                            kname.replace("\t", " ")])
    row["launches_per_iter"] = "%.1f" % launches
    row["kernels"] = str(kernels)
    row["gpu_busy_us"] = "%.1f" % busy_us
    row["gpu_util"] = "%.3f" % (busy_us / steady) if steady else ""
    row["notes"] = " ".join("%s=%.0f" % (k, mix[k]) for k in sorted(mix))
    for c in CLASSES:
        row[c] = "%.1f" % mix.get(c, 0.0)
    row["bmm"] = "%.1f" % bmm
    # The five classes partition the launches; if they ever stop doing so a
    # category() change has dropped kernels on the floor, and the row says so
    # rather than quietly reporting a total nobody checked.
    if abs(sum(mix.values()) - launches) > 0.05:
        row["notes"] += " CLASS-SUM-MISMATCH"
    return row


def main(argv):
    if argv[1:2] == ["--worker-ours"]:
        worker_ours(argv[2], argv[3], int(argv[4]), int(argv[5]))
        return 0
    out = argv[1] if len(argv) > 1 else os.environ.get("OUT")
    if not out:
        print("gap_analysis.py: give a result directory (or set OUT)",
              file=sys.stderr)
        return 1
    wanted = argv[2:]
    models = [m for m in MODELS if not wanted or m[0] in wanted]

    nrounds = int(os.environ.get("ROUNDS", 3))
    tmp = tempfile.mkdtemp(prefix="gap-")
    rows = []
    for model in models:
        # Interleaved: every system sees the same slice of a shared GPU.
        timings = dict((s, []) for s in SYSTEMS)
        for _ in range(nrounds):
            for s in SYSTEMS:
                cmd = command(s, model, int(os.environ.get("ITERS", 200)),
                              int(os.environ.get("WARMUP", 30)))
                timings[s].append(steady_of(run(cmd)) if cmd else None)
        for s in SYSTEMS:
            row = measure(s, model, timings[s], tmp)
            print("\t".join(row[c] for c in COLUMNS))
            rows.append(row)

    # Appended, like every other tsv here, so one model can be re-measured
    # without losing the rest; the figure keys on (model, system) and keeps
    # the last row for a pair.
    path = os.path.join(out, "gap.tsv")
    header = "\t".join(COLUMNS)
    fresh = not os.path.exists(path)
    if not fresh:
        # A result set started before the per-class columns existed keeps its
        # rows readable by name: the old header is a prefix of this one, so
        # bringing the header line up to date is enough (the old rows simply
        # have the new fields empty).
        have = open(path).readline().rstrip("\n")
        if have != header and header.startswith(have):
            rest = open(path).readlines()[1:]
            with open(path, "w") as f:
                f.write(header + "\n")
                f.writelines(rest)
    with open(path, "a") as f:
        if fresh:
            f.write(header + "\n")
        for row in rows:
            f.write("\t".join(row[c] for c in COLUMNS) + "\n")
    print("wrote " + path)

    kpath = os.path.join(out, "gap_kernels.tsv")
    kfresh = not os.path.exists(kpath)
    with open(kpath, "a") as f:
        if kfresh:
            f.write("\t".join(KERNEL_COLUMNS) + "\n")
        for row in KERNEL_ROWS:
            f.write("\t".join(row) + "\n")
    print("wrote " + kpath)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
