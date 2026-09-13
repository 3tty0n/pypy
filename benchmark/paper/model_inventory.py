#!/usr/bin/env python3
"""What each model actually computes, independent of the output check.

    benchmark/paper/bench.sh inventory       # writes $OUT/model_inventory.tsv
    benchmark/paper/model_inventory.py $OUT [MODEL...]

The models figure compares nine workloads across systems, and the only
evidence so far that the two systems run the same architecture is that their
logits agree to a tolerance.  That is an output check: it cannot separate "the
same computation" from "a different computation that happens to agree here".
This table is the structural evidence next to it - parameters, multiply-add
work, and how many cuBLAS/cuDNN calls a forward costs - with the differences
that remain named one by one.

Our side is analytic: the exported checkpoint (index.json) plus the model
config fix every GEMM shape in tensorpypy/models.py, so no GPU, no pypy-c and
no measurement is involved.  There is no runtime matmul counter in
_metatensor (kernel_count/kernel_compile_count/launch_count are the only
counters, and launch_count counts generated kernels too), so the call counts
are derived from the model structure and cross-checked against the nsys kernel
mix in gap.tsv for the five models that has.

Torch's side is measured on one eager forward: FlopCounterMode for the work,
torch.profiler for the calls, p.numel() for the parameters.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "applevel")

# model -> (family, weights dir, batch).  Same mapping as run_models.sh.
MODELS = [
    ("distilgpt2", "gpt2", "distilgpt2", 1),
    ("tiny-gpt2", "gpt2", "tiny-gpt2", 1),
    ("smollm2-135m", "llama", "smollm2-135m", 1),
    ("bert-tiny", "bert", "bert-tiny", 1),
    ("bert-mini", "bert", "bert-mini", 1),
    ("resnet18-b1", "resnet", "resnet18", 1),
    ("resnet18-b8", "resnet", "resnet18", 8),
    ("mixer_b16", "mixer", "mixer_b16", 1),
    ("vit-tiny", "vit", "vit-tiny", 1),
]

COLUMNS = ["model", "system", "params", "gflops", "gemm_calls", "bmm_calls",
           "conv_calls", "sdpa_calls", "notes"]

# index.json entries that are not model parameters: the input image, and the
# rotary tables, which HF rebuilds as a buffer rather than storing.
NOT_PARAMS = ("image", "rope.cos", "rope.sin", "rope.p")


# --- ours: analytic, from index.json ----------------------------------------

class Ops(object):
    """The GEMM-shaped work of one forward.  `conv` is a GEMM too - Conv2d in
    tensorpypy is im2col followed by one cuBLAS call - but it is counted apart
    so the column lines up with torch's aten::convolution."""

    def __init__(self):
        self.flops = 0
        self.calls = {"gemm": 0, "bmm": 0, "conv": 0, "sdpa": 0}

    def add(self, kind, m, k, n, count=1, batch=1):
        self.flops += 2 * m * k * n * count * batch
        self.calls[kind] += count
        return self


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def params_of(cfg):
    return sum(numel(shape) for name, (_, shape) in cfg["index"].items()
               if name not in NOT_PARAMS)


def cols(cfg, name):
    """Trailing dimension of an exported weight, so the feed-forward width is
    read off the checkpoint rather than assumed to be 4d."""
    return cfg["index"][name][1][-1]


def attn_block(ops, t, d, ff, heads, gelu_proj=True):
    """One pre-norm transformer block as models.py builds it: the packed QKV
    GEMM, the two batched attention calls, the output projection and the two
    feed-forward GEMMs."""
    ops.add("gemm", t, d, 3 * d)          # packed qkv, one call
    ops.add("bmm", t, d // heads, t, batch=heads)   # attn_scores
    ops.add("bmm", t, t, d // heads, batch=heads)   # attn_context
    ops.add("gemm", t, d, d)              # attn out projection
    ops.add("gemm", t, d, ff)             # mlp fc
    ops.add("gemm", t, ff, d)             # mlp proj


def ours_gpt2(cfg, batch):
    ops = Ops()
    t, d, L = cfg["seq"] * batch, cfg["n_embd"], cfg["n_layer"]
    ff = cols(cfg, "h.0.mlp.fc.w")
    for _ in range(L):
        attn_block(ops, t, d, ff, cfg["n_head"])
    ops.add("gemm", t, d, cfg["vocab"])   # lm head, tied to wte
    return ops, "tied lm head (wte reused, 1 GEMM)"


def ours_bert(cfg, batch):
    ops = Ops()
    t, d, L = cfg["seq"] * batch, cfg["n_embd"], cfg["n_layer"]
    ff = cols(cfg, "h.0.mlp.fc.w")
    for _ in range(L):
        attn_block(ops, t, d, ff, cfg["n_head"])
    ops.add("gemm", t, d, d)              # mlm dense
    ops.add("gemm", t, d, cfg["vocab"])   # mlm decoder, tied to wte
    return ops, ("position+token-type embeddings folded into one [seq,d] "
                 "table at export; tied mlm decoder")


def ours_llama(cfg, batch):
    ops = Ops()
    t, d, L = cfg["seq"], cfg["n_embd"], cfg["n_layer"]
    ff = cols(cfg, "h.0.mlp.gate.w")
    for _ in range(L):
        ops.add("gemm", t, d, 3 * d)      # packed qkv (kv heads pre-expanded)
        ops.add("bmm", t, d // cfg["n_head"], t, batch=cfg["n_head"])
        ops.add("bmm", t, t, d // cfg["n_head"], batch=cfg["n_head"])
        ops.add("gemm", t, d, d)
        ops.add("gemm", t, d, ff)         # gate
        ops.add("gemm", t, d, ff)         # up
        ops.add("gemm", t, ff, d)         # down
    ops.add("gemm", t, d, cfg["vocab"])
    return ops, ("GQA expanded at export: k/v are stored as %d full heads, so "
                 "the packed QKV GEMM is [d,3d] where torch's k/v are "
                 "[d,%d]" % (cfg["n_head"], cfg["head_dim"] * cfg["n_kv_head"]))


def ours_vit(cfg, batch):
    ops = Ops()
    t, d, L = cfg["tokens"], cfg["n_embd"], cfg["n_layer"]
    ff = cols(cfg, "h.0.mlp.fc.w")
    ops.add("gemm", t, cfg["patch"], d)   # patch embedding as a GEMM
    for _ in range(L):
        attn_block(ops, t, d, ff, cfg["n_head"])
    ops.add("gemm", 1, d, cfg["classes"])  # cls token only
    return ops, ("patch embedding is a gather + one GEMM, not a conv; "
                 "classifier reads the cls row only")


def ours_mixer(cfg, batch):
    ops = Ops()
    d, L = cfg["n_embd"], cfg["n_layer"]
    t = (cfg["image_size"] // cfg["patch_size"]) ** 2
    ff = cols(cfg, "h.0.mlp.fc.w")
    tok = cfg["index"]["h.0.tok.fc.w"][1][0]
    ops.add("gemm", t, cfg["index"]["wp"][1][0], d)   # im2col patch GEMM
    for _ in range(L):
        ops.add("gemm", tok, t, d)        # token mixing fc
        ops.add("gemm", t, tok, d)        # token mixing proj
        ops.add("gemm", t, d, ff)         # channel mixing fc
        ops.add("gemm", t, ff, d)         # channel mixing proj
    ops.add("gemm", 1, t, d)              # mean pool as a GEMM
    ops.add("gemm", 1, d, cfg["classes"])
    return ops, ("patch embedding is im2col + one GEMM; global average pool "
                 "is a [1,tokens] GEMM, not a reduction")


def resnet_convs(cfg, batch):
    """(m, k, n) of every conv GEMM, in forward order, following resnet.py's
    geometry: stem stride 2, max pool stride 2, then the block chain."""
    idx = cfg["index"]
    s = cfg["image_size"]
    k = cfg["stem"][1]
    oh = (s + 2 * (k // 2) - k) // 2 + 1
    out = [(batch * oh * oh,) + tuple(idx["conv1.w"][1])]
    h = (oh + 2 * 1 - 3) // 2 + 1               # max pool
    for li, n, shape, kk, stride, down in cfg["layers"]:
        p = "l%d.%d." % (li, n)
        o1 = (h + 2 * (kk // 2) - kk) // stride + 1
        out.append((batch * o1 * o1,) + tuple(idx[p + "conv1.w"][1]))
        o2 = (o1 + 2 * (kk // 2) - kk) // 1 + 1
        out.append((batch * o2 * o2,) + tuple(idx[p + "conv2.w"][1]))
        if down:
            od = (h + 0 - 1) // stride + 1
            out.append((batch * od * od,) + tuple(idx[p + "down.w"][1]))
        h = o2
    return out, h


def ours_resnet(cfg, batch):
    ops = Ops()
    convs, h = resnet_convs(cfg, batch)
    for m, k, n in convs:
        ops.add("conv", m, k, n)
    chan = cfg["index"]["fc.w"][1][0]
    ops.add("gemm", batch, batch * h * h, chan)   # average pool as a GEMM
    ops.add("gemm", batch, chan, cfg["classes"])
    return ops, ("every conv is im2col + one cuBLAS GEMM (TF32), where torch "
                 "calls cuDNN; global average pool is a GEMM, not a reduction")


OURS = {"gpt2": ours_gpt2, "bert": ours_bert, "llama": ours_llama,
        "vit": ours_vit, "mixer": ours_mixer, "resnet": ours_resnet}


def ours_row(name, family, cfg, batch):
    ops, notes = OURS[family](cfg, batch)
    return dict(model=name, system="ours", params=str(params_of(cfg)),
                gflops="%.3f" % (ops.flops / 1e9),
                gemm_calls=str(ops.calls["gemm"]),
                bmm_calls=str(ops.calls["bmm"]),
                conv_calls=str(ops.calls["conv"]),
                sdpa_calls="0", notes=notes)


# --- torch: measured, one eager forward -------------------------------------

GEMM_OPS = ("aten::mm", "aten::addmm")
BMM_OPS = ("aten::bmm", "aten::baddbmm")
# One convolution appears three times in the profile - aten::convolution,
# aten::_convolution, aten::cudnn_convolution are the same call seen at three
# dispatch levels - so the column counts the outermost and the notes keep all
# three, which is what makes the triple visible.
CONV_OPS = ("aten::convolution",)
SDPA_PREFIX = "aten::_scaled_dot_product"
REPORTED = (GEMM_OPS + BMM_OPS + CONV_OPS +
            ("aten::cudnn_convolution", "aten::_convolution",
             "aten::linear", "aten::matmul"))


def build_torch(family, cfg, weights, batch, dev, dtype):
    """The same module each *_torch.py builds, and the same input."""
    import torch
    import torch_common
    if family in ("gpt2", "bert", "llama"):
        idx = torch.tensor([cfg["tokens"]] * batch, device=dev,
                           dtype=torch.long)
        if family == "gpt2":
            from transformers import GPT2LMHeadModel
            hf = GPT2LMHeadModel.from_pretrained(cfg["source"])
        elif family == "bert":
            from transformers import BertForMaskedLM
            hf = BertForMaskedLM.from_pretrained(cfg["source"])
        else:
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(cfg["source"],
                                                      dtype=dtype)
        hf = hf.to(dev, dtype).eval()
        if family == "llama":
            return hf, (lambda i: hf(i, use_cache=False).logits), (idx,)
        return hf, (lambda i: hf(i).logits), (idx,)
    a = torch_common.Args()
    a.cfg, a.outdir = cfg, weights
    px = torch_common.image(a).to(dev, dtype)
    if family == "vit":
        from transformers import ViTForImageClassification
        hf = ViTForImageClassification.from_pretrained(cfg["source"])
        hf = hf.to(dev, dtype).eval()
        return hf, (lambda x: hf(x).logits), (px,)
    import timm
    m = timm.create_model(cfg["source"], pretrained=True).to(dev, dtype).eval()
    if family == "resnet":
        px = px.permute(0, 3, 1, 2).contiguous().expand(
            batch, -1, -1, -1).contiguous()
    return m, m, (px,)


def torch_worker(name, family, weights, batch):
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils.flop_counter import FlopCounterMode
    cfg = json.load(open(os.path.join(weights, "index.json")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, os.environ.get("RTENSOR_DTYPE", "float32"))
    module, fwd, args = build_torch(family, cfg, weights, batch, dev, dtype)
    with torch.no_grad():
        fwd(*args)                                   # allocate, warm cudnn
        with FlopCounterMode(display=False) as fc:
            fwd(*args)
        flops = fc.get_total_flops()
        acts = [ProfilerActivity.CPU]
        if dev == "cuda":
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts) as prof:
            fwd(*args)
    counts = {}
    for ev in prof.key_averages():
        if ev.key in REPORTED or ev.key.startswith(SDPA_PREFIX):
            counts[ev.key] = counts.get(ev.key, 0) + ev.count
    print(json.dumps(dict(
        params=sum(p.numel() for p in module.parameters()),
        flops=flops, counts=counts)))


def torch_row(name, family, weights, batch):
    py = os.environ.get("TORCH_PYTHON") or os.environ.get("RTENSOR_PYTHON")
    if not py:
        return None
    env = dict(os.environ, PYTHONPATH=os.path.abspath(APP))
    cmd = [py, os.path.abspath(__file__), "--torch-worker", name, family,
           weights, str(batch)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)
    out, _ = p.communicate()
    if p.returncode != 0 or not out.strip():
        print("model_inventory: torch failed for %s" % name, file=sys.stderr)
        return None
    data = json.loads(out.decode("utf-8").strip().splitlines()[-1])
    c = data["counts"]
    sdpa = sum(v for k, v in c.items() if k.startswith(SDPA_PREFIX))
    per_op = " ".join("%s=%d" % (k.replace("aten::", ""), c[k])
                      for k in sorted(c))
    return dict(model=name, system="torch", params=str(data["params"]),
                gflops="%.3f" % (data["flops"] / 1e9),
                gemm_calls=str(sum(c.get(k, 0) for k in GEMM_OPS)),
                bmm_calls=str(sum(c.get(k, 0) for k in BMM_OPS)),
                conv_calls=str(sum(c.get(k, 0) for k in CONV_OPS)),
                sdpa_calls=str(sdpa), notes=per_op)


# --- the difference, named --------------------------------------------------

def gap_gemm(out):
    """{(model, system): {category: kernels per forward}} from gap.tsv's mix
    column.  Kernel counts, not call counts: cuBLAS finishes a split-K GEMM
    with a separate reduction kernel, which nsys sees and a call count does
    not."""
    path = os.path.join(out, "gap.tsv")
    mix = {}
    if not os.path.exists(path):
        return mix
    import csv
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for part in (r.get("notes") or "").split():
                key, sep, value = part.partition("=")
                if sep and value.isdigit() and key in ("gemm", "splitk"):
                    mix.setdefault((r["model"], r["system"]), {})[key] = int(value)
    return mix


# What torch does differently, per architecture, filled in with the model's own
# layer count.  This is the (d) column: every remaining call-count difference
# named, so the two systems can be compared without trusting either's timing.
DELTA = {
    "gpt2": lambda c: (
        "HF GPT2 packs q/k/v into one Conv1D GEMM, as ours does; SDPA is one "
        "fused kernel where ours issues 2 batched GEMMs per layer (%d)"
        % (2 * c["n_layer"])),
    "bert": lambda c: (
        "q/k/v are 3 separate projections (3x%d=%d GEMMs) where ours packs "
        "them into %d; SDPA is %d calls where ours issues 2 batched GEMMs per "
        "layer (%d)" % (c["n_layer"], 3 * c["n_layer"], c["n_layer"],
                        c["n_layer"], 2 * c["n_layer"])),
    "vit": lambda c: (
        "q/k/v are 3 separate projections (3x%d=%d GEMMs) where ours packs "
        "them into %d; SDPA is %d calls where ours issues 2 batched GEMMs per "
        "layer (%d); patch embedding is a cuDNN conv"
        % (c["n_layer"], 3 * c["n_layer"], c["n_layer"], c["n_layer"],
           2 * c["n_layer"])),
    "llama": lambda c: (
        "q/k/v are 3 separate projections (3x%d=%d GEMMs, k/v narrow: %d "
        "cols for %d kv heads) where ours packs them into %d full-width ones; "
        "SDPA is %d calls where ours issues 2 batched GEMMs per layer (%d)"
        % (c["n_layer"], 3 * c["n_layer"], c["head_dim"] * c["n_kv_head"],
           c["n_kv_head"], c["n_layer"], c["n_layer"], 2 * c["n_layer"])),
    "mixer": lambda c: (
        "patch embedding is a cuDNN conv, not im2col + GEMM; global average "
        "pool is a reduction, not a GEMM"),
    "resnet": lambda c: (
        "every conv is a cuDNN call, not im2col + GEMM; global average pool "
        "is a reduction, not a GEMM"),
}


def reconcile(ours, torch_, family, cfg, gemm_mix):
    """Append to each row's notes what is left over: the call-count gap named
    piece by piece, the parameter and FLOP deltas, and what nsys saw."""
    seen = gemm_mix.get((ours["model"], "ours"))
    if seen:
        # The analytic count is exact, so the residual against the nsys gemm
        # bucket is the cuBLAS-internal kernels - split-K reduce and the
        # cublasLt epilogue - which are kernels, not calls.  A gap run made
        # after gap_analysis started separating them records splitk directly.
        calls = sum(int(ours[k]) for k in ("gemm_calls", "bmm_calls",
                                           "conv_calls"))
        splitk = seen.get("splitk", seen.get("gemm", 0) - calls)
        ours["notes"] += ("; nsys gemm=%d kernels = %d cuBLAS calls + %d "
                          "split-K reduce/epilogue" %
                          (seen.get("gemm", 0), calls, splitk))
    if not torch_:
        return
    o_calls = sum(int(ours[k]) for k in ("gemm_calls", "bmm_calls",
                                         "conv_calls"))
    # SDPA counted as the two batched GEMMs it stands for, which is what the
    # attention actually costs on either side; where the math backend already
    # decomposed it, its bmm children are in the profile and counting them
    # twice would be wrong, hence max() rather than a sum.
    t_calls = (int(torch_["gemm_calls"]) + int(torch_["conv_calls"]) +
               max(int(torch_["bmm_calls"]), 2 * int(torch_["sdpa_calls"])))
    ours["notes"] += ("; %d cuBLAS calls vs torch's %d (SDPA counted as its "
                      "2 batched GEMMs)" % (o_calls, t_calls))
    dp = int(ours["params"]) - int(torch_["params"])
    if dp:
        ours["notes"] += "; params %+d vs torch" % dp
    og, tg = float(ours["gflops"]), float(torch_["gflops"])
    if tg and abs(og - tg) / tg > 0.005:
        ours["notes"] += "; %+.1f%% FLOPs" % (100.0 * (og - tg) / tg)
    torch_["notes"] += "; " + DELTA[family](cfg)
    seen = gemm_mix.get((ours["model"], "torch-compile"))
    if seen:
        # Not reconciled against the eager call counts above: torch.compile
        # lowers SDPA per shape - one fused fmha kernel for distilgpt2, two
        # bmm for the small BERTs and ViT - so the compiled call count is a
        # measurement, not a property of the module.  The kernel total is
        # reported as nsys saw it, split-K reduces included.
        torch_["notes"] += ("; nsys gemm=%d kernels under torch.compile "
                            "(split-K reduce included)" % seen.get("gemm", 0))


def selfcheck(weights_root):
    """The analytic side has no measurement to disagree with, so it gets one
    runnable check: the closed-form call counts, and ResNet-18's published
    1.81 GMAC at batch 1, which also pins the conv geometry."""
    def row(name, family, wdir, batch):
        cfg = json.load(open(os.path.join(weights_root, wdir, "index.json")))
        return ours_row(name, family, cfg, batch), cfg
    r, cfg = row("distilgpt2", "gpt2", "distilgpt2", 1)
    assert int(r["gemm_calls"]) == 4 * cfg["n_layer"] + 1, r
    assert int(r["bmm_calls"]) == 2 * cfg["n_layer"], r
    r, cfg = row("vit-tiny", "vit", "vit-tiny", 1)
    assert int(r["gemm_calls"]) == 4 * cfg["n_layer"] + 2, r
    r, cfg = row("resnet18-b1", "resnet", "resnet18", 1)
    assert int(r["conv_calls"]) == 20, r
    assert abs(float(r["gflops"]) / 3.63 - 1.0) < 0.01, r
    r8, _ = row("resnet18-b8", "resnet", "resnet18", 8)
    assert abs(float(r8["gflops"]) / (8 * float(r["gflops"])) - 1) < 0.01, r8
    print("selfcheck ok")


def main(argv):
    if argv[1:2] == ["--torch-worker"]:
        torch_worker(argv[2], argv[3], argv[4], int(argv[5]))
        return 0
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default=os.environ.get("OUT"))
    p.add_argument("models", nargs="*")
    p.add_argument("--no-torch", action="store_true",
                   help="our rows only (no torch venv needed)")
    p.add_argument("--selfcheck", action="store_true",
                   help="assert the analytic call counts and ResNet FLOPs")
    args = p.parse_args(argv[1:])
    if not args.out:
        p.error("give a result directory (or set OUT)")
    weights_root = os.environ.get("WEIGHTS", os.path.join(HERE, "weights"))
    if args.selfcheck:
        selfcheck(weights_root)
        return 0
    gemm_mix = gap_gemm(args.out)

    rows = []
    for name, family, wdir, batch in MODELS:
        if args.models and name not in args.models:
            continue
        weights = os.path.join(weights_root, wdir)
        cfg = json.load(open(os.path.join(weights, "index.json")))
        ours = ours_row(name, family, cfg, batch)
        torch_ = None if args.no_torch else torch_row(name, family, weights,
                                                      batch)
        reconcile(ours, torch_, family, cfg, gemm_mix)
        for row in (ours, torch_):
            if row:
                rows.append(row)
                print("\t".join(row[c] for c in COLUMNS))

    path = os.path.join(args.out, "model_inventory.tsv")
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(row[c] for c in COLUMNS) + "\n")
    print("wrote " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
