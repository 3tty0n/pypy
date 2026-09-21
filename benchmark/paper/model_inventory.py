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
import collections
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "applevel")

# model -> (family, weights dir, batch).  Same mapping as run_models.sh.
MODELS = [
    ("gpt2", "gpt2", "gpt2", 1),
    ("gpt2-medium", "gpt2", "gpt2-medium", 1),
    ("distilgpt2", "gpt2", "distilgpt2", 1),
    ("smollm2-135m", "llama", "smollm2-135m", 1),
    ("smollm2-360m", "llama", "smollm2-360m", 1),
    ("smollm2-1.7b", "llama", "smollm2-1.7b", 1),
    ("qwen2.5-0.5b", "llama", "qwen2.5-0.5b", 1),
    ("bert-base", "bert", "bert-base", 1),
    ("bert-mini", "bert", "bert-mini", 1),
    ("resnet18-b1", "resnet", "resnet18", 1),
    ("resnet18-b8", "resnet", "resnet18", 8),
    ("mixer_b16", "mixer", "mixer_b16", 1),
    ("vit-base", "vit", "vit-base", 1),
    ("deit-tiny", "vit", "deit-tiny", 1),
    ("vit-tiny", "vit", "vit-tiny", 1),
    ("tiny-gpt2", "gpt2", "tiny-gpt2", 1),
    ("bert-tiny", "bert", "bert-tiny", 1),
]

COLUMNS = ["model", "system", "params_only", "params_plus_buffers", "gflops",
           "gemm_calls", "bmm_calls", "conv_calls", "sdpa_calls",
           "hf_id", "revision", "weights", "notes"]

# Where a checkpoint came from and whether every weight the benchmark reads is
# a trained one.  A reader cannot check a number against the wrong checkpoint,
# and "untrained" is a property to declare, not to hide: two of the rows are
# fixtures whose weights were never trained, and they are reported apart from
# the population for that reason.
#
#   trained      every tensor the forward reads comes from the checkpoint
#   head-random  the encoder is pretrained, the task head is not in the
#                checkpoint and transformers initialises it randomly
#   random       the whole checkpoint is a randomly initialised test fixture
WEIGHT_PROVENANCE = {
    "sshleifer/tiny-gpt2": "random",
    "prajjwal1/bert-tiny": "head-random",
}

HF_HUB = os.path.join(os.environ.get("HF_HOME",
                                     os.path.expanduser("~/.cache/huggingface")),
                      "hub")


def hf_revision(source, *also):
    """The repository and snapshot the export actually read, as "repo@sha".

    timm names are not hub ids; the hub repo is timm/<name>, and torchvision's
    plain "resnet18" resolves to the timm default weights.  `also` carries the
    name the export recorded, for a checkpoint whose id the hub has since
    redirected: the cache is keyed by the name that was downloaded.
    """
    for repo in (source,) + also + ("timm/" + source,
                                    "timm/" + source + ".a1_in1k"):
        d = os.path.join(HF_HUB, "models--" + repo.replace("/", "--"),
                         "snapshots")
        if os.path.isdir(d):
            names = os.listdir(d)
            if names:
                # A repo can have several snapshots cached; the export read
                # the one that was downloaded last.
                newest = max(names,
                             key=lambda n: os.path.getmtime(os.path.join(d, n)))
                return repo + "@" + newest[:12]
    return source + "@unknown"

# One definition of "parameter", applied to both sides.
#
# params_only        the learned weights
# params_plus_buffers  those plus the registered non-learned tensors that are
#                    still part of the trained model - batch-norm running mean
#                    and variance, and nothing else in these nine models
#
# Excluded from both columns on both sides, because they are inputs or tables
# rebuilt from the config rather than anything the training produced:
#
#   ours   image (the input image), rope.cos/rope.sin/rope.p (rotary tables)
#   torch  position_ids, token_type_ids (position index), inv_freq and
#          original_inv_freq (rotary tables), num_batches_tracked (a counter,
#          not a tensor of data), and the causal mask where a module keeps one
#
# The input token ids live in the config, not in index.json, so they are
# outside the count on our side by construction.
NOT_PARAMS = ("image", "rope.cos", "rope.sin", "rope.p")
# Our exported batch-norm running statistics: <layer>.m and <layer>.v.
BUFFER_SUFFIX = (".m", ".v")
# torch buffers that are inputs or config-derived tables, not model data.
NOT_TORCH_BUFFERS = ("position_ids", "token_type_ids", "inv_freq",
                     "original_inv_freq", "num_batches_tracked",
                     "attn.bias", "masked_bias", "causal_mask",
                     "cos_cached", "sin_cached")


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


def is_buffer(name):
    return name.endswith(BUFFER_SUFFIX)


def params_of(cfg, buffers=False):
    return sum(numel(shape) for name, (_, shape) in cfg["index"].items()
               if name not in NOT_PARAMS
               and (buffers or not is_buffer(name)))


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


# A checkpoint the export recorded under a name the Hub now redirects.
# The weights are the same bytes; the provenance table should name the
# repository as it is addressed today.
CANONICAL_ID = {"distilgpt2": "distilbert/distilgpt2"}


def ours_row(name, family, cfg, batch):
    ops, notes = OURS[family](cfg, batch)
    recorded = cfg.get("source", "unknown")
    source = CANONICAL_ID.get(recorded, recorded)
    # The id column names the repository the snapshot was actually found in,
    # so a timm short name reads as the hub repo that holds those weights
    # rather than as the string the export happened to pass to timm.
    revision = hf_revision(source, recorded)
    # A name with no slash is a timm or torchvision short name, not a hub id,
    # so the id column takes the repository the snapshot was found in.  A name
    # that already is a hub id keeps its canonical spelling even when the
    # cache is keyed by the one the hub redirects from.
    repo = revision.rpartition("@")[0]
    hf_id = source if "/" in source else (repo or source)
    return dict(model=name, system="ours",
                hf_id=hf_id, revision=revision,
                weights=WEIGHT_PROVENANCE.get(source, "trained"),
                params_only=str(params_of(cfg)),
                params_plus_buffers=str(params_of(cfg, True)),
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
    bufs = {n: b.numel() for n, b in module.named_buffers()
            if not n.endswith(NOT_TORCH_BUFFERS)}
    aten = {}
    for ev in prof.key_averages():
        if ev.key.startswith("aten::"):
            aten[ev.key] = aten.get(ev.key, 0) + ev.count
    print(json.dumps(dict(
        params=sum(p.numel() for p in module.parameters()),
        buffers=sum(bufs.values()), buffer_names=sorted(bufs),
        flops=flops, counts=counts, aten=aten)))


def torch_data(name, family, weights, batch):
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
    return json.loads(out.decode("utf-8").strip().splitlines()[-1])


def torch_row(data):
    c = data["counts"]
    sdpa = sum(v for k, v in c.items() if k.startswith(SDPA_PREFIX))
    per_op = " ".join("%s=%d" % (k.replace("aten::", ""), c[k])
                      for k in sorted(c))
    # Provenance is a property of the checkpoint, not of the system that
    # loads it, so it is carried on the ours row alone and left blank here;
    # the provenance table reads the ours rows.
    return dict(model=data["model"], system="torch",
                hf_id="", revision="", weights="",
                params_only=str(data["params"]),
                params_plus_buffers=str(data["params"] + data["buffers"]),
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
        if "splitk" in seen:
            # gap_analysis separates them, so the buckets add up as they are.
            ours["notes"] += ("; nsys gemm=%d kernels for %d cuBLAS calls, "
                              "plus splitk=%d reduce/epilogue kernels" %
                              (seen.get("gemm", 0), calls, seen["splitk"]))
        else:
            ours["notes"] += ("; nsys gemm=%d kernels = %d cuBLAS calls + %d "
                              "split-K reduce/epilogue" %
                              (seen.get("gemm", 0), calls,
                               seen.get("gemm", 0) - calls))
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
    for col in ("params_only", "params_plus_buffers"):
        dp = int(ours[col]) - int(torch_[col])
        if dp:
            ours["notes"] += "; %s %+d vs torch" % (col, dp)
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


# --- one vocabulary, both sides ---------------------------------------------

VOCAB = ["matmul", "bmm", "conv", "layernorm", "rmsnorm", "softmax", "gelu",
         "silu", "relu", "add", "mul", "residual_add", "embedding_gather",
         "rotary", "reshape_transpose", "reduce", "other"]
OP_COLUMNS = ["model", "system"] + VOCAB + ["notes"]

# aten name -> vocabulary word.  One dispatch level only: torch's profiler
# reports aten::linear *and* the aten::addmm it dispatches to, aten::softmax
# *and* aten::_softmax, aten::convolution *and* aten::_convolution *and*
# aten::cudnn_convolution, so counting both levels would count one call twice.
# The wrappers are listed here as ALIAS and are not counted; the raw per-op
# profile is kept in the notes column so the choice is checkable.
ATEN_MAP = collections.OrderedDict([
    ("aten::mm", "matmul"), ("aten::addmm", "matmul"),
    ("aten::bmm", "bmm"), ("aten::baddbmm", "bmm"),
    ("aten::convolution", "conv"),
    ("aten::native_layer_norm", "layernorm"),
    ("aten::_fused_rms_norm", "rmsnorm"), ("aten::rms_norm", "rmsnorm"),
    ("aten::_softmax", "softmax"),
    ("aten::gelu", "gelu"), ("aten::silu", "silu"),
    ("aten::relu", "relu"), ("aten::relu_", "relu"),
    ("aten::add", "add"), ("aten::add_", "add"),
    ("aten::mul", "mul"), ("aten::mul_", "mul"),
    ("aten::embedding", "embedding_gather"),
    ("aten::index_select", "embedding_gather"),
    ("aten::view", "reshape_transpose"),
    ("aten::reshape", "reshape_transpose"),
    ("aten::transpose", "reshape_transpose"),
    ("aten::permute", "reshape_transpose"),
    ("aten::contiguous", "reshape_transpose"),
    ("aten::expand", "reshape_transpose"),
    ("aten::sum", "reduce"), ("aten::mean", "reduce"),
    ("aten::amax", "reduce"),
])
# Wrappers: the same call seen at an outer dispatch level as something already
# in ATEN_MAP.  Dropped, not counted, not "other".
ALIAS = ("aten::linear", "aten::matmul", "aten::conv2d",
         "aten::_convolution", "aten::cudnn_convolution",
         "aten::convolution_overrideable",
         "aten::softmax", "aten::_safe_softmax",
         "aten::layer_norm", "aten::_unsafe_view",
         "aten::detach", "aten::to", "aten::_to_copy", "aten::copy_",
         "aten::empty", "aten::empty_like", "aten::empty_strided",
         "aten::resolve_conj", "aten::resolve_neg", "aten::t",
         "aten::flatten", "aten::unsqueeze", "aten::squeeze",
         "aten::slice", "aten::select", "aten::size", "aten::item",
         "aten::_has_compatible_shallow_copy_type", "aten::is_same_size",
         # inner dispatch levels of ops already counted above
         "aten::as_strided", "aten::_reshape_alias", "aten::alias",
         "aten::clamp_min_", "aten::clamp_min",
         "aten::max_pool2d_with_indices", "aten::adaptive_avg_pool2d",
         "aten::_batch_norm_impl_index", "aten::cudnn_batch_norm",
         "aten::scaled_dot_product_attention",
         "aten::_efficient_attention_forward",
         "aten::_flash_attention_forward")
# One SDPA call stands for the maths it fused.  Taken as a floor, not a sum:
# the math backend decomposes SDPA into real aten::bmm/softmax/mul, which are
# already in the profile, and adding the implied ones on top would count the
# same work twice.
SDPA_AS = {"bmm": 2, "softmax": 1, "mul": 1}


def vocab_counts():
    return collections.OrderedDict((w, 0) for w in VOCAB)


# --- ours: analytic, the same walk of models.py as the FLOP side ------------

def _attn(v, masked):
    """models.CausalSelfAttention.forward."""
    v["matmul"] += 1                       # packed qkv
    v["add"] += 1                          # qkv bias
    v["bmm"] += 1                          # attn_scores
    v["mul"] += 1                          # 1/sqrt(dh)
    if masked:
        v["add"] += 1                      # causal mask
    v["softmax"] += 1
    v["bmm"] += 1                          # attn_context
    v["matmul"] += 1                       # out projection
    v["add"] += 1                          # out bias


def _mlp(v, act):
    """models.GPT2MLP.forward; act is "gelu" (tanh form) or "gelu_erf"."""
    v["matmul"] += 2
    v["add"] += 2
    v["gelu"] += 1


def ops_gpt2(v, cfg, batch):
    v["embedding_gather"] += 1             # wte.take(idx)
    v["add"] += 1                          # + position table
    for _ in range(cfg["n_layer"]):
        v["layernorm"] += 1
        _attn(v, True)
        v["residual_add"] += 1
        v["layernorm"] += 1
        _mlp(v, "gelu")
        v["residual_add"] += 1
    v["layernorm"] += 1
    v["matmul"] += 1                       # tied lm head


def ops_bert(v, cfg, batch):
    v["embedding_gather"] += 1
    v["add"] += 1                          # + folded position/token-type table
    v["layernorm"] += 1
    for _ in range(cfg["n_layer"]):
        _attn(v, False)                    # bidirectional: no mask
        v["residual_add"] += 1
        v["layernorm"] += 1
        _mlp(v, "gelu_erf")
        v["residual_add"] += 1
        v["layernorm"] += 1
    v["matmul"] += 1                       # mlm dense
    v["add"] += 1
    v["gelu"] += 1                         # gelu_erf
    v["layernorm"] += 1
    v["matmul"] += 1                       # tied mlm decoder
    v["add"] += 1                          # mlm bias


def ops_llama(v, cfg, batch):
    v["embedding_gather"] += 1
    for _ in range(cfg["n_layer"]):
        v["rmsnorm"] += 1
        v["matmul"] += 1                   # packed qkv
        v["rotary"] += 1                   # qkv*cos + rot_half(qkv)*sin
        v["bmm"] += 1
        v["mul"] += 1
        v["add"] += 1                      # causal mask
        v["softmax"] += 1
        v["bmm"] += 1
        v["matmul"] += 1                   # out projection
        v["residual_add"] += 1
        v["rmsnorm"] += 1
        v["matmul"] += 2                   # gate, up
        v["silu"] += 1
        v["mul"] += 1                      # gate * up
        v["matmul"] += 1                   # down
        v["residual_add"] += 1
    v["rmsnorm"] += 1
    v["matmul"] += 1                       # tied lm head


def ops_vit(v, cfg, batch):
    v["embedding_gather"] += 1             # img.take(patch index)
    v["reshape_transpose"] += 1
    v["matmul"] += 1                       # patch projection
    v["add"] += 1                          # + position/cls embedding
    for _ in range(cfg["n_layer"]):
        v["layernorm"] += 1
        _attn(v, False)
        v["residual_add"] += 1
        v["layernorm"] += 1
        _mlp(v, "gelu_erf")
        v["residual_add"] += 1
    v["layernorm"] += 1
    v["embedding_gather"] += 1             # cls row
    v["matmul"] += 1
    v["add"] += 1


def ops_mixer(v, cfg, batch):
    v["reshape_transpose"] += 1            # im2col
    v["matmul"] += 1                       # patch projection
    v["add"] += 1
    for _ in range(cfg["n_layer"]):
        v["layernorm"] += 1
        v["matmul"] += 1                   # token mixing fc
        v["add"] += 1
        v["gelu"] += 1                     # gelu_erf
        v["matmul"] += 1                   # token mixing proj
        v["add"] += 1
        v["residual_add"] += 1
        v["layernorm"] += 1
        _mlp(v, "gelu_erf")
        v["residual_add"] += 1
    v["layernorm"] += 1
    v["matmul"] += 2                       # mean pool GEMM, classifier
    v["add"] += 1


def _conv_bn(v, k, stride, pad):
    """nn.Conv2d(nhwc=True) then nn.BatchNorm2d(nhwc=True)."""
    if not (k == 1 and stride == 1 and pad == 0):
        v["reshape_transpose"] += 1        # im2col_nhwc
    v["conv"] += 1                         # the TF32 GEMM
    v["mul"] += 1                          # batch norm folded to scale
    v["add"] += 1                          # ... and shift


def ops_resnet(v, cfg, batch):
    k = cfg["stem"][1]
    _conv_bn(v, k, 2, k // 2)
    v["relu"] += 1
    v["other"] += 1                        # maxpool2_nhwc
    for _, _, _, kk, stride, down in cfg["layers"]:
        _conv_bn(v, kk, stride, kk // 2)
        v["relu"] += 1
        _conv_bn(v, kk, 1, kk // 2)
        if down:
            _conv_bn(v, 1, stride, 0)
        v["residual_add"] += 1
        v["relu"] += 1
    v["matmul"] += 2                       # mean pool GEMM, fc
    v["add"] += 1


OURS_OPS = {"gpt2": ops_gpt2, "bert": ops_bert, "llama": ops_llama,
            "vit": ops_vit, "mixer": ops_mixer, "resnet": ops_resnet}


# --- the fused-kernel IR, for the expansion counts in the notes -------------

TTIR_OP = re.compile(r"%v\d+ = ([\w.]+)")


def ttir_opcodes(tmpdir, outdir):
    """{model: (kernels, {ttir opcode: nodes})} from fusion_stats.py's dumps.

    The elementwise vocabulary words above are library-level tensor ops; this
    is what they become after fusion, and it is the only place the gelu
    polynomial's size is written down.  Counted per compiled kernel, not per
    launch: nothing in the dump says how often each kernel runs.  The kernels
    belonging to the model are the last `kernels` distinct ones, the ones
    fusion.tsv counts; the earlier numbers are start-up single-op kernels.
    """
    found = {}
    if not tmpdir or not os.path.isdir(tmpdir):
        return found
    sys.path.insert(0, HERE)
    try:
        import fusion_stats
    except ImportError:
        return found
    counts = {}
    fusion = os.path.join(outdir, "fusion.tsv")
    if os.path.exists(fusion):
        with open(fusion) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                counts[r["model"]] = int(r["kernels"])
    for model in sorted(os.listdir(tmpdir)):
        d = os.path.join(tmpdir, model)
        if not os.path.isdir(d):
            continue
        ns = [int(f[9:-5]) for f in os.listdir(d)
              if f.startswith("rtensor_k") and f.endswith(".ttir")]
        if not ns:
            continue
        ks = fusion_stats.kernels_of(d, 0, max(ns) + 1)
        n = counts.get(model)
        if n:
            ks = ks[-n:]
        hist = collections.Counter()
        for k in ks:
            for line in k[4].splitlines():
                m = TTIR_OP.match(line.strip())
                if m and m.group(1) not in ("tt.load", "tt.splat",
                                            "arith.constant"):
                    hist[m.group(1)] += 1
        found[model] = (len(ks), hist)
    return found


def ttir_note(model, ttir):
    got = ttir.get(model)
    if not got:
        return ""
    n, hist = got
    body = " ".join("%s=%d" % (k.split(".")[-1], v)
                    for k, v in hist.most_common())
    return ("fused elementwise IR, per compiled kernel (not per launch): "
            "%d kernels, %s" % (n, body))


# --- rows -------------------------------------------------------------------

# gelu_erf and gelu are one library call on our side and one aten::gelu on
# torch's, but both expand into a polynomial once lowered.  Measured off the
# generated Triton IR (rtensor_k107 on distilgpt2 is the tanh form).
GELU_EXPANSION = (
    "gelu counts 1 per call on both sides, but the two forms are not the same "
    "size once lowered. Measured on the generated Triton IR: the tanh form "
    "(distilgpt2, gpt2) is a 13-node region - 7 addf, 4 mulf, 1 exp, 1 divf; "
    "the erf form (bert, vit, mixer) is an 88-node region - 35 addf, 30 mulf, "
    "16 select, 5 divf, 1 exp, 1 subf. Both regions also carry the fused bias "
    "add, and the pass does no common-subexpression elimination, so a shared "
    "operand is recomputed at every use")

OURS_OP_NOTE = {
    "gpt2": "gelu is the tanh form; the causal mask is one add per layer",
    "bert": "gelu is the erf form; no mask (bidirectional)",
    "llama": "rotary is one row: qkv*cos + rot_half(qkv)*sin, i.e. 2 mul + "
             "1 add + 1 rot_half gather once lowered",
    "vit": "gelu is the erf form; the patch embedding is a gather + reshape "
           "+ one GEMM, not a conv",
    "mixer": "gelu is the erf form; the patch embedding is im2col + one GEMM; "
             "global average pool is a [1,tokens] GEMM, counted as matmul",
    "resnet": "batch norm is folded into one mul + one add per conv (no "
              "batchnorm word in the vocabulary); maxpool is the one 'other'; "
              "global average pool is a GEMM, counted as matmul",
}


def ours_op_row(name, family, cfg, batch, ttir):
    v = vocab_counts()
    OURS_OPS[family](v, cfg, batch)
    notes = [OURS_OP_NOTE[family]]
    t = ttir_note(name, ttir)
    if t:
        notes.append(t)
    row = dict(model=name, system="ours", notes="; ".join(notes))
    row.update((w, str(v[w])) for w in VOCAB)
    return row


def torch_op_row(data):
    v = vocab_counts()
    other, unmapped = 0, collections.Counter()
    sdpa = 0
    for key, n in data["aten"].items():
        if key.startswith(SDPA_PREFIX):
            sdpa += n
            continue
        if key in ALIAS:
            continue
        word = ATEN_MAP.get(key)
        if word:
            v[word] += n
        else:
            other += n
            unmapped[key] += n
    for word, k in SDPA_AS.items():
        v[word] = max(v[word], k * sdpa)
    v["other"] = other
    notes = []
    if sdpa:
        notes.append("%d SDPA call(s) counted as %d bmm + %d softmax + %d mul"
                     % (sdpa, 2 * sdpa, sdpa, sdpa))
    notes.append("add is every aten::add: the profiler cannot tell a residual "
                 "connection from a bias add, so residual_add is 0 here and "
                 "the residuals are inside add")
    if unmapped:
        notes.append("other = " + " ".join(
            "%s=%d" % (k.replace("aten::", ""), n)
            for k, n in unmapped.most_common()))
    row = dict(model=data["model"], system="torch", notes="; ".join(notes))
    row.update((w, str(v[w])) for w in VOCAB)
    return row


def write_tsv(path, columns, rows):
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(row[c] for c in columns) + "\n")
    print("wrote " + path)


# Why a cell differs, keyed by (family or "*", vocabulary word).  Anything not
# listed is reported as an unexplained difference, which is the point of the
# table.
OP_REASON = {
    ("*", "residual_add"): "we count the block's residual connection as its "
        "own word; torch's profiler cannot tell it from a bias add, so torch "
        "carries its residuals inside add and residual_add is 0",
    ("*", "add"): "torch folds every linear's bias into aten::addmm where we "
        "issue a separate add, and torch's residual adds land here too",
    ("*", "mul"): "HF writes gelu, RMSNorm and rotary as Python tensor "
        "expressions, so their muls are individual aten::mul calls; ours are "
        "inside one gelu/rms_norm/rotary op",
    ("*", "reshape_transpose"): "torch materialises a view at every layout "
        "change (head split and merge, NCHW/NHWC, every qkv split); our "
        "kernels index the flat buffer, so only a real data movement "
        "(im2col, the ViT patch gather) is counted",
    ("*", "embedding_gather"): "HF looks up token, position and token-type "
        "tables with one aten::embedding each; we fold them into one table "
        "and one gather, and add the input add separately",
    ("*", "matmul"): "we pack q/k/v into one GEMM per layer where HF issues "
        "three projections",
    ("*", "bmm"): "one SDPA call is counted as the 2 batched products it "
        "stands for; ours issues those 2 batched GEMMs explicitly",
    ("*", "conv"): "the patch embedding is a cuDNN conv on torch's side; ours "
        "is a gather (ViT) or im2col (Mixer) plus a GEMM, counted as matmul",
    ("*", "reduce"): "torch's global average pool is aten::mean; ours is a "
        "[1,tokens] GEMM, counted as matmul",
    ("*", "other"): "aten ops with no word in the vocabulary: dropout in "
        "eval, allocation and fill, batch norm, max pool, and the scalar "
        "pieces of HF's Python gelu/RMSNorm/rotary",
    ("gpt2", "gelu"): "HF GPT-2 uses NewGELUActivation, written in Python, so "
        "the profiler sees pow/tanh/mul/add and never aten::gelu; ours is one "
        "gelu op per MLP",
    ("llama", "rmsnorm"): "HF LlamaRMSNorm is Python (pow, mean, rsqrt, mul), "
        "so there is no aten::rms_norm to count; ours is one row kernel",
    ("llama", "add"): "torch's adds are the residuals, the eps inside HF's "
        "Python RMSNorm and the two halves of its Python rotary; ours are the "
        "causal mask only, the rest being inside rms_norm and rotary",
    ("llama", "rotary"): "HF applies rotary as Python mul/add over q and k, "
        "which lands in mul and add; ours is one rotary op per layer",
    ("llama", "reduce"): "the aten::mean inside HF's Python RMSNorm; our "
        "rms_norm does that reduction inside the row kernel",
    ("resnet", "mul"): "batch norm is cudnn_batch_norm on torch (counted in "
        "other); ours is folded into one mul and one add per conv",
    ("resnet", "matmul"): "our global average pool is a [B, B*hw] GEMM, "
        "torch's is aten::mean",
    ("mixer", "matmul"): "our patch embedding and global average pool are "
        "GEMMs; torch's are a cuDNN conv and aten::mean",
    ("vit", "matmul"): "we pack q/k/v into one GEMM per layer where HF issues "
        "three projections, and our patch embedding is a GEMM where torch's "
        "is a cuDNN conv",
    ("vit", "embedding_gather"): "our patch extraction is a gather over the "
        "image and the classifier reads the cls row with a second gather; "
        "torch does the first with a conv and the second with a slice",
    ("resnet", "add"): "the convolutions carry no bias on either side: our "
        "adds are the batch-norm shift (20) plus the fc bias, torch's are "
        "only the 8 residual connections, with batch norm inside "
        "cudnn_batch_norm",
    ("resnet", "reshape_transpose"): "one im2col per conv on our side; on "
        "torch's, the NCHW/NHWC permutes and the flatten before the fc",
}


FAMILY = dict((name, family) for name, family, _, _ in MODELS)


def write_notes(path, rows):
    """The tex note: the exact aten mapping, and every differing cell named."""
    lines = ["# aten name -> vocabulary word (the mapping the torch rows use)"]
    for k, w in ATEN_MAP.items():
        lines.append("map\t%s\t%s" % (k.replace("aten::", ""), w))
    lines.append("map\t_scaled_dot_product_*\t2 bmm + 1 softmax + 1 mul")
    for k in ALIAS:
        lines.append("alias\t%s\tdropped: outer dispatch level of a call "
                     "already counted" % k.replace("aten::", ""))
    lines.append("note\t\t" + GELU_EXPANSION)
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["system"]] = r
    for model in sorted(by):
        o, t = by[model].get("ours"), by[model].get("torch")
        if not (o and t):
            continue
        for w in VOCAB:
            a, b = int(o[w]), int(t[w])
            if a == b:
                continue
            why = (OP_REASON.get((model, w))
                   or OP_REASON.get((FAMILY.get(model), w))
                   or OP_REASON.get(("*", w))
                   or "NOT EXPLAINED by any rule above")
            lines.append("diff\t%s %s: ours %d, torch %d\t%s"
                         % (model, w, a, b, why))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote " + path)


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
    # The vocabulary walker and the FLOP walker read the same models.py by
    # hand, so they are only evidence if they agree on the calls they share.
    for name, family, wdir, batch in MODELS:
        cfg = json.load(open(os.path.join(weights_root, wdir, "index.json")))
        a = ours_row(name, family, cfg, batch)
        b = ours_op_row(name, family, cfg, batch, {})
        for x, y in (("gemm_calls", "matmul"), ("bmm_calls", "bmm"),
                     ("conv_calls", "conv")):
            assert a[x] == b[y], (name, x, a[x], b[y])
        # batch norm is one mul + one add per conv, nothing else adds a mul
        # in ResNet, so this pins the ResNet walk.
        if family == "resnet":
            assert b["mul"] == a["conv_calls"], b
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
    p.add_argument("--ttir", default=os.path.join(
        tempfile.gettempdir(), "fusion-stats"),
        help="fusion_stats.py's dump directory, read for the fused-kernel "
             "opcode histogram that backs the elementwise notes")
    args = p.parse_args(argv[1:])
    weights_root = os.environ.get("WEIGHTS", os.path.join(HERE, "weights"))
    if args.selfcheck:
        selfcheck(weights_root)
        return 0
    if not args.out:
        p.error("give a result directory (or set OUT)")
    gemm_mix = gap_gemm(args.out)
    ttir = ttir_opcodes(args.ttir, args.out)

    rows, op_rows = [], []
    for name, family, wdir, batch in MODELS:
        if args.models and name not in args.models:
            continue
        weights = os.path.join(weights_root, wdir)
        cfg = json.load(open(os.path.join(weights, "index.json")))
        ours = ours_row(name, family, cfg, batch)
        data = None if args.no_torch else torch_data(name, family, weights,
                                                     batch)
        if data is not None:
            data["model"] = name
        torch_ = torch_row(data) if data else None
        reconcile(ours, torch_, family, cfg, gemm_mix)
        for row in (ours, torch_):
            if row:
                rows.append(row)
                print("\t".join(row[c] for c in COLUMNS))
        op_rows.append(ours_op_row(name, family, cfg, batch, ttir))
        if data:
            op_rows.append(torch_op_row(data))

    write_tsv(os.path.join(args.out, "model_inventory.tsv"), COLUMNS, rows)
    write_tsv(os.path.join(args.out, "op_inventory.tsv"), OP_COLUMNS, op_rows)
    write_notes(os.path.join(args.out, "op_inventory_notes.txt"), op_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
