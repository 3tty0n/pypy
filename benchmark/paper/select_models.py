#!/usr/bin/env python3
"""The model selection rule, applied and written out as evidence.

Every model in the evaluation is reimplemented against TensorPyPy's API, so
the population is a ported one and a reader cannot assume it. This applies a
rule that was fixed before the models were measured, one criterion at a time,
and records for each candidate which criterion it met and which it failed.
The population is then every candidate that meets all five - not a selection
made afterwards from the results.

  C1  published by the organisation that trained it, under that
      organisation's own account.  An individual's re-upload of someone
      else's weights is a mirror: it can disappear, it can differ from the
      original, and it is not what a deployment pulls.
  C2  every tensor the forward reads comes from the checkpoint.  A randomly
      initialised fixture, or a checkpoint whose task head is missing and
      is filled in at load time, is not a trained model.
  C3  people run it, not only publish it: at least DOWNLOAD_FLOOR downloads
      in the last 30 days, as the Hub reports them on the query date
      recorded in the output.
  C4  its forward is inside TensorPyPy's operator set at batch 1.
  C5  it occupies a point of the (architecture family, size) grid that no
      other admitted candidate occupies, so the population is a set of
      ladders rather than a pile.

C1 and C2 come first on purpose.  Download count alone would admit
sshleifer/tiny-gpt2, which has over a million downloads a month because it is
the fixture half the ecosystem's test suites instantiate, and whose weights
are random.

C3 has one exemption, and it is named per row rather than left to judgement.
A candidate that meets C1, C2, C4 and C5 and fails only C3 is admitted when
it is the only candidate covering its architecture family, or the only one
covering an end of that family's size ladder: dropping it would not make the
population more honest, it would make it narrower in a way the rule was never
meant to cause.  Every exempted row is marked in the output and in every
table, and the evaluation reports its headline mean twice - over the whole
population, and over the rows that met C3 on their own - so no exemption can
carry a result.

    select_models.py [OUT] [--offline]

Writes OUT/model_selection.tsv and prints the table.
"""
import argparse
import csv
import datetime
import os
import sys

# C1: the accounts we accept, and whose weights each one publishes.  A name
# is here because the organisation that trained the model controls it.
PUBLISHERS = {
    "openai-community": "OpenAI, GPT-2 release",
    "distilbert": "Hugging Face, DistilBERT/DistilGPT-2 release",
    "HuggingFaceTB": "Hugging Face, SmolLM release",
    "Qwen": "Alibaba, Qwen release",
    "google-bert": "Google, BERT release",
    "google": "Google Research",
    "facebook": "Meta AI",
    "timm": "the timm library's weight registry",
    "sentence-transformers": "the sentence-transformers release",
}

# C3: a checkpoint pulled this often in a month is one people run.  Fixed
# before the query; the gap in the observed data is two orders of magnitude
# wide, so the population does not turn on the exact value.
DOWNLOAD_FLOOR = 100000

# C2 verdicts.  "trained" unless stated; the two exceptions are the reason
# the criterion exists.
WEIGHTS = {
    "sshleifer/tiny-gpt2": "random: a test fixture, never trained",
    "prajjwal1/bert-tiny": "partial: pretrained encoder, masked-language "
                           "head absent from the checkpoint and initialised "
                           "randomly at load",
}

# C4 verdicts, from the operator inventory.  Everything listed here has been
# run; a candidate that had not been would say so.
BLOCKED = {}

# The C3 exemptions, with the coverage each one is the only source of.  A row
# is only exempt if it fails C3 alone; failing anything else is not covered
# by this table.
EXEMPT = {
    "mixer_b16": "only attention-free, convolution-free architecture in the "
                 "candidate set; no MLP-only vision checkpoint at this scale "
                 "is served",
    "bert-mini": "only small end of the encoder ladder; the alternative "
                 "(all-MiniLM-L6-v2) has no masked-language head and is not "
                 "ported",
}

# Porting status.  A candidate that is not ported cannot stand in for the
# grid point it would occupy, so it is recorded here with the reason rather
# than dropped from the list: a reader should be able to see that a
# better-used checkpoint was considered, and what admitting it would cost.
#   measured    ported, and in the result set
#   to-port     admitted, port outstanding this round
#   not-ported  declined; the reason is the cost, and it is named
NOT_PORTED = {
    "minilm-l6": "BertModel with no masked-language head: the output is the "
                 "encoder state, which needs a second output path through the "
                 "export, the model and both baseline twins",
}

# name, HuggingFace id, family, parameters, grid point (C5), status
CANDIDATES = [
    # GPT-2 decoder ladder
    ("distilgpt2",    "distilbert/distilgpt2",              "gpt2",   "82M",  "gpt2 small",   "measured"),
    ("gpt2",          "openai-community/gpt2",              "gpt2",   "124M", "gpt2 medium",  "measured"),
    ("gpt2-medium",   "openai-community/gpt2-medium",       "gpt2",   "355M", "gpt2 large",   "to-port"),
    # Llama-style decoder ladder
    ("smollm2-135m",  "HuggingFaceTB/SmolLM2-135M",         "llama",  "135M", "llama small",  "measured"),
    ("smollm2-360m",  "HuggingFaceTB/SmolLM2-360M",         "llama",  "360M", "llama medium", "measured"),
    ("smollm2-1.7b",  "HuggingFaceTB/SmolLM2-1.7B",         "llama",  "1.7B", "llama large",  "to-port"),
    # Llama-style with attention bias
    ("qwen2.5-0.5b",  "Qwen/Qwen2.5-0.5B-Instruct",         "llama",  "494M", "qwen2 bias",   "measured"),
    # Encoder
    ("bert-base",     "google-bert/bert-base-uncased",      "bert",   "109M", "bert base",    "measured"),
    ("bert-mini",     "google/bert_uncased_L-4_H-256_A-4",  "bert",   "11M",  "bert small",   "measured"),
    ("minilm-l6",     "sentence-transformers/all-MiniLM-L6-v2", "bert", "23M", "bert small",  "not-ported"),
    # Vision transformer
    ("vit-base",      "google/vit-base-patch16-224",        "vit",    "87M",  "vit base",     "measured"),
    ("vit-tiny",      "WinKawaks/vit-tiny-patch16-224",     "vit",    "5.7M", "vit small",    "measured"),
    ("deit-tiny",     "facebook/deit-tiny-patch16-224",     "vit",    "5.7M", "vit small",    "to-port"),
    # Convolution
    ("resnet18-b1",   "timm/resnet18.a1_in1k",              "resnet", "11.7M", "resnet b1",   "measured"),
    ("resnet18-b8",   "timm/resnet18.a1_in1k",              "resnet", "11.7M", "resnet b8",   "measured"),
    # Attention-free, GEMM only
    ("mixer_b16",     "timm/mixer_b16_224.goog_in21k_ft_in1k", "mixer", "60M", "mixer base",  "measured"),
    # Fixtures, here to be judged by the same rule as everything else
    ("tiny-gpt2",     "sshleifer/tiny-gpt2",                "gpt2",   "103k", "fixture",      "measured"),
    ("bert-tiny",     "prajjwal1/bert-tiny",                "bert",   "4.4M", "fixture",      "measured"),
]

COLUMNS = ["model", "hf_id", "family", "params", "grid_point", "status",
           "author",
           "downloads_30d", "c1_publisher", "c2_weights", "c3_used",
           "c4_operators", "c5_distinct", "verdict", "reason"]


def downloads(ids, offline):
    """{id: (author, downloads)} from the Hub, or from nothing when offline."""
    if offline:
        return {}
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for i in sorted(set(ids)):
        try:
            m = api.model_info(i)
            out[i] = (m.author or "", m.downloads or 0)
        except Exception as exc:
            out[i] = ("", -1)
            print("%s: %s" % (i, str(exc).splitlines()[0][:80]), file=sys.stderr)
    return out


def judge(rows):
    """C5 last: it can only be decided once the other four have run, because
    a grid point is claimed by the admitted candidate, not by every one."""
    claimed = {}
    for r in sorted(rows, key=lambda r: r["c3_used"] != "pass"):
        if (r["status"] == "not-ported"
                or not (r["c1_publisher"] == "pass"
                        and r["c2_weights"] == "pass"
                        and r["c4_operators"] == "pass")):
            r["c5_distinct"] = "n/a"
            continue
        key = (r["family"], r["grid_point"])
        if key in claimed:
            r["c5_distinct"] = "fail"
            r["reason"] = "same grid point as %s" % claimed[key]
        else:
            r["c5_distinct"] = "pass"
            claimed[key] = r["model"]
    for r in rows:
        failed = [c for c in ("c1_publisher", "c2_weights", "c3_used",
                              "c4_operators", "c5_distinct")
                  if r[c] == "fail"]
        if r["status"] == "not-ported":
            r["verdict"] = "not ported"
            r["reason"] = NOT_PORTED.get(r["model"], "not ported")
        elif not failed:
            r["verdict"] = "population"
        elif failed == ["c3_used"] and r["model"] in EXEMPT:
            r["verdict"] = "population*"
            r["reason"] = "C3 exemption: " + EXEMPT[r["model"]]
        else:
            r["verdict"] = "excluded"
            if not r["reason"]:
                r["reason"] = "fails " + ", ".join(c.split("_")[0].upper()
                                                   for c in failed)
    return rows


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default="")
    p.add_argument("--offline", action="store_true",
                   help="skip the Hub query; C3 is reported as unknown")
    a = p.parse_args(argv[1:])

    info = downloads([c[1] for c in CANDIDATES], a.offline)
    rows = []
    for name, hf_id, family, params, grid, status in CANDIDATES:
        author, dl = info.get(hf_id, ("", -1))
        weights = WEIGHTS.get(hf_id)
        rows.append({
            "model": name, "hf_id": hf_id, "family": family,
            "params": params, "grid_point": grid, "status": status,
            "author": author or "?",
            "downloads_30d": dl if dl >= 0 else "",
            "c1_publisher": "pass" if author in PUBLISHERS else "fail",
            "c2_weights": "fail" if weights else "pass",
            "c3_used": ("unknown" if dl < 0 else
                        ("pass" if dl >= DOWNLOAD_FLOOR else "fail")),
            "c4_operators": "fail" if hf_id in BLOCKED else "pass",
            "c5_distinct": "",
            "verdict": "", "reason": weights or BLOCKED.get(hf_id, ""),
        })
    rows = judge(rows)

    width = max(len(r["model"]) for r in rows)
    print("selection rule applied %s, download floor %s/30d\n"
          % (datetime.date.today().isoformat(), "{:,}".format(DOWNLOAD_FLOOR)))
    print("%-*s %-22s %12s  %s  %-11s %s"
          % (width, "model", "author", "dl/30d", "C1C2C3C4C5", "verdict",
             "reason"))
    for r in rows:
        flags = "".join(
            {"pass": " +", "fail": " -", "unknown": " ?", "n/a": " ."}[r[c]]
            for c in ("c1_publisher", "c2_weights", "c3_used", "c4_operators",
                      "c5_distinct"))
        print("%-*s %-22s %12s %s  %-11s %s"
              % (width, r["model"], r["author"],
                 "{:,}".format(r["downloads_30d"])
                 if r["downloads_30d"] != "" else "-",
                 flags, r["verdict"], r["reason"]))

    keep = [r["model"] for r in rows if r["verdict"].startswith("population")]
    used = [r["model"] for r in rows if r["verdict"] == "population"]
    exempt = [r["model"] for r in rows if r["verdict"] == "population*"]
    drop = [r["model"] for r in rows if r["verdict"] == "excluded"]
    unported = [r["model"] for r in rows if r["verdict"] == "not ported"]
    print("\npopulation (%d): %s" % (len(keep), " ".join(keep)))
    print("  met C3   (%d): %s" % (len(used), " ".join(used)))
    print("  exempt   (%d): %s" % (len(exempt), " ".join(exempt)))
    print("excluded   (%d): %s" % (len(drop), " ".join(drop)))
    print("not ported (%d): %s" % (len(unported), " ".join(unported)))

    todo = [r["model"] for r in rows
            if r["verdict"].startswith("population")
            and r["status"] == "to-port"]
    print("\nadmitted, port outstanding: %s" % (" ".join(todo) or "none"))

    if a.out:
        path = os.path.join(a.out, "model_selection.tsv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, COLUMNS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print("wrote " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
