#!/usr/bin/env python3
"""Which models of the PyTorch 2 benchmark suites fall inside the operator set.

The suites are the three the PyTorch 2 paper and the inductor dashboard run -
TorchBench, HuggingFace and TIMM - loaded through the dashboard's own runners
(pytorch/benchmarks/dynamo at the tag of the torch release the baselines use,
TorchBench at that tag's CI pin), with the dashboard's default inference batch
size.  Every model's eval forward runs once under a dispatch mode that records
each aten op it reaches; the op sets are then matched against SUPPORT.

    suite_census.py run   [--suite S] [--only NAME] [--out TSV]
    suite_census.py report TSV

`run` needs the TorchBench venv (setup: benchmark/paper/setup_suites.sh);
`report` needs nothing but the TSV.  Each model runs in its own process so a
crash, an OOM or a missing dependency is one row, not the end of the census.
"""
import collections
import json
import os
import subprocess
import sys

DYNAMO = os.environ.get("DYNAMO_BENCH",
                        os.path.expanduser("~/src/github.com/pytorch/"
                                           "dynamo-bench-v2.14.0"))
TORCHBENCH = os.environ.get("TORCHBENCH",
                            os.path.expanduser("~/src/github.com/pytorch/"
                                               "benchmark"))
SUITES = ("torchbench", "huggingface", "timm")
COLUMNS = ["suite", "model", "status", "batch", "dtype", "dashboard_skip",
           "ops", "detail"]

# aten op -> how _metatensor covers it, one of
#   native    a Tensor method does it (named after the colon)
#   composed  a short expression over Tensor methods, no new kernel
#   layout    a pure data-movement op; covered only in the patterns the
#             Tensor methods implement (reshape of a contiguous tensor,
#             attention's head split/merge, matmul's transpose flags), which
#             the census cannot see from the op name, so it is its own class
# Anything absent is missing.  Nothing here says more than the Tensor type
# in pypy/module/_metatensor/interp_tensor.py exposes.
SUPPORT = {
    "aten::mm": "native:matmul", "aten::addmm": "native:matmul",
    "aten::bmm": "native:bmm", "aten::baddbmm": "native:bmm",
    "aten::convolution": "native:conv2d",
    "aten::native_layer_norm": "native:layer_norm",
    "aten::_fused_rms_norm": "native:rms_norm",
    "aten::_softmax": "native:softmax",
    "aten::gelu": "native:gelu", "aten::silu": "native:silu",
    "aten::relu": "native:relu", "aten::relu_": "native:relu",
    "aten::add": "native:add", "aten::add_": "native:add",
    "aten::sub": "native:sub", "aten::mul": "native:mul",
    "aten::mul_": "native:mul", "aten::div": "native:div",
    "aten::exp": "native:exp", "aten::sqrt": "native:sqrt",
    "aten::embedding": "native:take", "aten::index_select": "native:take",
    "aten::sum": "native:sum", "aten::amax": "native:max",
    "aten::max_pool2d_with_indices": "native:maxpool2",
    "aten::argmax": "native:argmax", "aten::_to_copy": "native:astype",
    "aten::rsqrt": "composed:div(1,sqrt)", "aten::neg": "composed:mul(-1)",
    "aten::mean": "composed:sum*1/n",
    "aten::adaptive_avg_pool2d": "composed:matmul(ones/n)",
    "aten::_native_batch_norm_legit_no_training": "composed:mul+add",
    "aten::cudnn_batch_norm": "composed:mul+add",
    "aten::native_batch_norm": "composed:mul+add",
    "aten::_scaled_dot_product_efficient_attention":
        "composed:bmm+softmax+bmm",
    "aten::_scaled_dot_product_flash_attention": "composed:bmm+softmax+bmm",
    "aten::_scaled_dot_product_cudnn_attention": "composed:bmm+softmax+bmm",
    "aten::dropout": "composed:identity(eval)",
    "aten::native_dropout": "composed:identity(eval)",
}
for _op in ("view", "_unsafe_view", "reshape", "_reshape_alias", "transpose",
            "t", "permute", "expand", "clone", "unsqueeze", "squeeze", "alias",
            "detach", "as_strided", "copy_", "lift_fresh", "contiguous",
            "flatten"):
    SUPPORT["aten::" + _op] = "layout"
# Allocation and host-scalar ops: no device work of their own.
for _op in ("empty", "empty_strided", "empty_like", "zeros", "zeros_like",
            "ones", "ones_like", "full", "full_like", "arange",
            "scalar_tensor", "_local_scalar_dense", "fill_", "new_empty",
            "new_zeros", "new_ones", "new_full", "zero_", "sym_size",
            "sym_stride", "sym_numel", "is_same_size", "resolve_conj",
            "resolve_neg"):
    SUPPORT["aten::" + _op] = "alloc"


def variant_of(name, args):
    if name != "aten::convolution":
        return ""
    stride, pad, dilation, transposed, _, groups = args[3:9]
    tags = []
    if transposed:
        tags.append("transposed")
    if groups != 1:
        tags.append("groups")
    if any(d != 1 for d in dilation):
        tags.append("dilation")
    if len(stride) != 2:
        tags.append("conv%dd" % len(stride))
    return "+".join(tags)


def make_runner(suite, name):
    """The dashboard's runner for SUITE, set up as its inference performance
    run of NAME in float32 is."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path[:0] = [DYNAMO, TORCHBENCH]
    os.chdir(TORCHBENCH)
    # huggingface.py pip-installs transformers from git main whenever an
    # import fails, for any reason; a census or an export on that is not the
    # pinned suite
    import transformers
    want = os.environ.get("SUITES_TRANSFORMERS", "5.13.0")
    if transformers.__version__ != want:
        raise SystemExit("transformers %s, not the pinned %s"
                         % (transformers.__version__, want))
    import common
    if suite == "torchbench":
        from torchbench import TorchBenchmarkRunner as R
    elif suite == "huggingface":
        from huggingface import HuggingfaceRunner as R
    else:
        from timm_models import TimmRunner as R
    runner = R()
    args = common.parse_args(["--inference", "--performance", "--float32",
                              "--backend", "eager", "--device", "cuda",
                              "--only", name])
    runner.args = args
    runner.model_iter_fn = runner.forward_pass
    return runner, args


def worker(suite, name):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    runner, args = make_runner(suite, name)
    skips = set()
    for attr in ("skip_models", "skip_models_for_cuda",
                 "skip_models_for_freezing_cuda",
                 "skip_models_due_to_control_flow"):
        try:
            skips |= set(getattr(runner, attr))
        except Exception:
            pass
    out = {"suite": suite, "model": name, "dashboard_skip": int(name in skips)}
    try:
        device, name2, model, inputs, batch = runner.load_model(
            "cuda", name, batch_size=args.batch_size)
    except Exception as e:
        out.update(status="load_fail", detail=repr(e)[:300])
        return out
    model, inputs = runner.cast_based_on_args(model, inputs)
    out["batch"] = batch

    ops = collections.Counter()
    dtypes = collections.Counter()

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, a=(), kw=None):
            nm = func._overloadpacket._qualified_op_name
            v = variant_of(nm, a)
            ops[nm + ("[%s]" % v if v else "")] += 1
            for t in a:
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    dtypes[str(t.dtype).replace("torch.", "")] += 1
                    break
            return func(*a, **(kw or {}))

    try:
        with torch.no_grad():
            runner.model_iter_fn(model, inputs)
            torch.cuda.synchronize()
            with Record():
                runner.model_iter_fn(model, inputs)
            torch.cuda.synchronize()
    except Exception as e:
        out.update(status="run_fail", detail=repr(e)[:300])
        return out
    out.update(status="ok", ops=dict(ops),
               dtype=",".join(d for d, _ in dtypes.most_common()))
    return out


def names(suite):
    if suite == "torchbench":
        d = os.path.join(TORCHBENCH, "torchbenchmark", "models")
        return sorted(n for n in os.listdir(d)
                      if os.path.isdir(os.path.join(d, n))
                      and not n.startswith(("_", ".")))
    f = {"huggingface": "huggingface_models_list.txt",
         "timm": "timm_models_list.txt"}[suite]
    out = []
    for line in open(os.path.join(DYNAMO, f)):
        line = line.strip()
        if line:
            out.append(line.split(",")[0].split()[0].strip())
    return out


def classify(ops):
    missing, layout = [], []
    for key in ops:
        base, _, var = key.partition("[")
        kind = SUPPORT.get(base)
        if kind is None or var:
            missing.append(key)
        elif kind == "layout":
            layout.append(key)
    return sorted(missing), sorted(layout)


def run(argv):
    suite = flag(argv, "--suite", "all")
    only = flag(argv, "--only", None)
    out = flag(argv, "--out", "suite_census.tsv")
    timeout = int(flag(argv, "--timeout", "900"))
    done = set()
    if os.path.exists(out):
        for line in open(out).readlines()[1:]:
            f = line.split("\t")
            done.add((f[0], f[1]))
    else:
        open(out, "w").write("\t".join(COLUMNS) + "\n")
    for s in SUITES if suite == "all" else (suite,):
        for n in [only] if only else names(s):
            if (s, n) in done:
                continue
            try:
                p = subprocess.run([sys.executable, __file__, "--one", s, n],
                                   capture_output=True, text=True,
                                   timeout=timeout)
                rows = [l for l in p.stdout.splitlines()
                        if l.startswith("CENSUS ")]
                row = json.loads(rows[-1][7:]) if rows else {
                    "suite": s, "model": n, "status": "crash",
                    "detail": (p.stderr.strip().splitlines() or ["?"])[-1][:300]}
            except subprocess.TimeoutExpired:
                row = {"suite": s, "model": n, "status": "timeout"}
            row["ops"] = json.dumps(row.get("ops", {}), sort_keys=True)
            with open(out, "a") as f:
                f.write("\t".join(str(row.get(c, "")).replace("\t", " ")
                                  .replace("\n", " ") for c in COLUMNS) + "\n")
            print("%-12s %-40s %s" % (s, n, row["status"]), flush=True)


def report(path):
    import csv
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    need = collections.Counter()
    tally = collections.Counter()
    print("| suite | model | batch | class | missing ops |")
    print("|---|---|---|---|---|")
    for r in rows:
        miss = "-"
        if r["status"] != "ok":
            cls = r["status"]
        else:
            m, lay = classify(json.loads(r["ops"]))
            need.update(m)
            cls = ("covered" if not lay else "covered, layout") if not m \
                else "missing 1-2" if len(m) <= 2 else "missing 3+"
            miss = ", ".join(k.replace("aten::", "") for k in m) or "-"
        tally[(r["suite"], cls)] += 1
        print("| %s | %s | %s | %s | %s |" % (
            r["suite"], r["model"], r["batch"], cls, miss))
    print("\n| suite | class | models |\n|---|---|---|")
    for (s, c), n in sorted(tally.items()):
        print("| %s | %s | %d |" % (s, c, n))
    print("\n| missing op | models needing it |\n|---|---|")
    for k, n in need.most_common():
        print("| %s | %d |" % (k.replace("aten::", ""), n))


def flag(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--one"]:
        print("CENSUS " + json.dumps(worker(a[1], a[2])), flush=True)
    elif a[:1] == ["run"]:
        run(a[1:])
    elif a[:1] == ["report"]:
        report(a[1])
    else:
        print(__doc__.strip())
