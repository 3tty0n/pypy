#!/usr/bin/env python3
"""Dump one suite model exactly as the PyTorch 2 dashboard runs it.

    suite_export.py SUITE MODEL OUTDIR [--batch B]

Loads MODEL through the dashboard's runner for SUITE (the same path
suite_census.py takes), then writes OUTDIR/index.json and OUTDIR/data.bin:
every state_dict entry under its torch name and in its torch dtype, the
example inputs, and the eval forward's outputs on those inputs.  Nothing is
renamed, packed, folded or transposed: a port reads the tensors a torch
module holds, so the arithmetic it does can be checked against the source
module op for op.

index.json:
  suite, model, batch, torch, source (the suite's pinned revision)
  params  {name: [byte offset, shape, dtype]}
  inputs  [[byte offset, shape, dtype], ...]   positional, in call order
  kwargs  {name: [byte offset, shape, dtype]}
  outputs [[byte offset, shape, dtype], ...]   flattened in pytree order
  outputs_fp64  the same forward with model and inputs cast to float64 by
                the dashboard's cast_to_fp64, the reference its accuracy
                check measures errors against
  tolerance, cosine, larger_multiplier   what the runner's
                get_tolerance_and_cosine_flag and config give this model
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suite_census  # noqa: E402


def revision(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def config_of(model):
    """A transformers model's config as the dict its constructor reads."""
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "to_dict"):
        return None
    d = cfg.to_dict()
    d["_attn_implementation"] = getattr(cfg, "_attn_implementation", None)
    d["attribute_map"] = dict(getattr(type(cfg), "attribute_map", {}))
    return json.loads(json.dumps(d, default=str))


def main(argv):
    import torch
    from torch.utils._pytree import tree_flatten
    suite, name, out = argv[1], argv[2], os.path.abspath(argv[3])
    batch = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else None
    runner, args = suite_census.make_runner(suite, name)
    _, _, model, inputs, batch = runner.load_model("cuda", name,
                                                   batch_size=batch)
    model, inputs = runner.cast_based_on_args(model, inputs)
    import common
    with torch.no_grad():
        result = runner.forward_pass(model, inputs)
        m64, i64 = common.cast_to_fp64(copy.deepcopy(model),
                                       copy.deepcopy(inputs))
        result64 = runner.forward_pass(m64, i64)
    torch.cuda.synchronize()
    tol, cosine = runner.get_tolerance_and_cosine_flag(False, "cuda", name)
    larger = name in getattr(
        runner, "require_larger_multiplier_for_smaller_tensor", ())

    os.makedirs(out, exist_ok=True)
    blob = open(os.path.join(out, "data.bin"), "wb")

    def put(t):
        t = t.detach().contiguous().cpu()
        off = blob.tell()
        blob.write(t.numpy().tobytes() if t.dtype != torch.bfloat16
                   else t.view(torch.int16).numpy().tobytes())
        return [off, list(t.shape), str(t.dtype).replace("torch.", "")]

    params = {k: put(v) for k, v in model.state_dict(keep_vars=False).items()}
    if isinstance(inputs, dict):
        pos, kw = [], {k: put(v) for k, v in inputs.items()
                       if isinstance(v, torch.Tensor)}
    else:
        pos = [put(v) for v in inputs if isinstance(v, torch.Tensor)]
        kw = {}
    outs = [put(v) for v in tree_flatten(result)[0]
            if isinstance(v, torch.Tensor)]
    outs64 = [put(v) for v in tree_flatten(result64)[0]
              if isinstance(v, torch.Tensor)]
    blob.close()
    src = suite_census.TORCHBENCH if suite == "torchbench" else \
        suite_census.DYNAMO
    json.dump({"suite": suite, "model": name, "batch": batch,
               "torch": torch.__version__, "source": revision(src),
               "params": params, "inputs": pos, "kwargs": kw,
               "outputs": outs, "outputs_fp64": outs64,
               "config": config_of(model),
               "tolerance": tol, "cosine": bool(cosine),
               "larger_multiplier": bool(larger)},
              open(os.path.join(out, "index.json"), "w"), indent=1,
              sort_keys=True)
    print("wrote %s: %d params, %d inputs, %d outputs, %.1f MB" % (
        out, len(params), len(pos) + len(kw), len(outs),
        os.path.getsize(os.path.join(out, "data.bin")) / 1e6))


if __name__ == "__main__":
    main(sys.argv)
