#!/bin/bash
# Correctness on more than one input.
#
# The models sweep checks each system's logits against ours on the one input
# stored in the checkpoint.  One input cannot separate "the same computation"
# from "a computation that happens to agree on this vector", so this runs the
# same check on five more inputs per model: INPUT_SEED=1..5, derived from the
# stored input by the integer recipe in applevel/inputs.py (text: the id
# sequence rotated by k and shifted by k, mod vocab; vision: additive uniform
# +-0.05 noise on the normalized pixels, keyed by the flat index and the seed).
# All three sides derive the same bits from that recipe.
#
# Ours runs first for each seed and writes logits_pypy_seed<k>.bin; the other
# systems are compared against that file and judged by the same tolerance_for
# table the models sweep uses.  Seed 0 is never run here, so the stored
# logits_pypy.bin every other stage compares against is not touched.
#
#     benchmark/paper/bench.sh correctness             # all nine models
#     benchmark/paper/bench.sh correctness bert-mini   # one
#     SEEDS="1 2" benchmark/paper/bench.sh correctness # fewer inputs
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

SEEDS=${SEEDS:-"1 2 3 4 5"}
# Only the outputs matter here, so the timed loop is as short as it can be
# while still being the same code path.
ITERS=${CORRECTNESS_ITERS:-5}
WARMUP=${CORRECTNESS_WARMUP:-2}

TSV="$OUT/correctness.tsv"
tsv_init "$TSV" "model\tsystem\tseed\tmaxabsdiff\targmax_match\ttol\tpass"

ALL_MODELS="gpt2 distilgpt2 smollm2-135m smollm2-360m qwen2.5-0.5b bert-base bert-mini resnet18-b1 resnet18-b8 mixer_b16 vit-base vit-tiny tiny-gpt2 bert-tiny"

# model -> pypy script, torch script, jax model, weights dir, kind, extra argv
spec() {
  case "$1" in
    gpt2)         echo "gpt2.py gpt2_torch.py gpt2 gpt2 text" ;;
    distilgpt2)   echo "gpt2.py gpt2_torch.py gpt2 distilgpt2 text" ;;
    tiny-gpt2)    echo "gpt2.py gpt2_torch.py gpt2 tiny-gpt2 text" ;;
    smollm2-135m) echo "llama.py llama_torch.py llama smollm2-135m text" ;;
    smollm2-360m) echo "llama.py llama_torch.py llama smollm2-360m text" ;;
    qwen2.5-0.5b) echo "llama.py llama_torch.py llama qwen2.5-0.5b text" ;;
    bert-base)    echo "bert.py bert_torch.py bert bert-base text" ;;
    bert-tiny)    echo "bert.py bert_torch.py bert bert-tiny text" ;;
    bert-mini)    echo "bert.py bert_torch.py bert bert-mini text" ;;
    resnet18-b1)  echo "resnet.py resnet_torch.py resnet resnet18 vision 1" ;;
    resnet18-b8)  echo "resnet.py resnet_torch.py resnet resnet18 vision 8" ;;
    mixer_b16)    echo "mixer.py mixer_torch.py mixer mixer_b16 vision" ;;
    vit-base)     echo "vit.py vit_torch.py vit vit-base vision" ;;
    vit-tiny)     echo "vit.py vit_torch.py vit vit-tiny vision" ;;
    *) return 1 ;;
  esac
}

# The `argmax` line without the diff fields run_models.sh also strips.
argmax_line() { echo "$1" | grep '^argmax' | sed 's/ maxabsdiff=.*//' | head -1; }

# How much of the decision agrees: the fraction of positions for a text model
# (one argmax per token), top-1 for a vision model (the argmax line is the
# top-5 ranking, so its first entry is the predicted class).
argmax_match() {
  awk -v r="$1" -v s="$2" -v kind="$3" 'BEGIN {
    n = split(r, a, " "); m = split(s, b, " ")
    if (n < 2 || m < 2) { print ""; exit }
    if (kind == "vision") { print (a[2] == b[2]) ? "1.0000" : "0.0000"; exit }
    c = 0; t = 0
    for (i = 2; i <= n && i <= m; i++) { t++; if (a[i] == b[i]) c++ }
    if (t == 0) { print ""; exit }
    printf "%.4f\n", c / t
  }'
}

row() {
  local model=$1 system=$2 seed=$3 diff=$4 match=$5 tol=$6 passed=$7
  echo -e "$model\t$system\t$seed\t$diff\t$match\t$tol\t$passed" >> "$TSV"
  bench_record correctness model="$model" system="$system" seed="$seed" \
    maxabsdiff="$diff" argmax_match="$match" tolerance="$tol" pass="$passed"
}

one_model() {
  local model=$1
  set -- $(spec "$model")
  local pyscript=$1 torchscript=$2 jaxmodel=$3 wdir=$4 kind=$5 extra=$6
  local weights="$WEIGHTS/$wdir"
  MODEL_TOL=$(tolerance_for "$model")
  export MODEL_TOL

  for seed in $SEEDS; do
    progress_step "$model seed $seed"
    export INPUT_SEED=$seed
    # Ours first: it writes the reference for this input.
    local out ref
    if ! out=$(INPUT_SEED=$seed "$RUN_PYPY" $JIT_FLAGS "$APP/$pyscript" \
                 "$weights" "$ITERS" "$WARMUP" $extra 2>/dev/null); then
      echo "run_correctness.sh: ours failed for $model seed $seed" >&2
      continue
    fi
    ref=$(argmax_line "$out")
    row "$model" ours "$seed" 0 1.0000 "$MODEL_TOL" 1

    system_row() {
      local system=$1; shift
      local o
      if ! o=$("$@" 2>/dev/null) || ! echo "$o" | grep -q steady_us=; then
        echo "run_correctness.sh: $system failed for $model seed $seed" >&2
        return 0
      fi
      row "$model" "$system" "$seed" \
        "$(diff_of "$o")" \
        "$(argmax_match "$ref" "$(argmax_line "$o")" "$kind")" \
        "$(field_of "$o" tol)" "$(field_of "$o" pass)"
    }

    system_row torch-eager "$TORCH_PYTHON" "$APP/$torchscript" eager \
      "$weights" "$ITERS" "$WARMUP" $extra
    system_row torch-compile "$TORCH_PYTHON" "$APP/$torchscript" compile \
      "$weights" "$ITERS" "$WARMUP" $extra
    if [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ]; then
      system_row jax "$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" jax \
        "$weights" "$ITERS" "$WARMUP" $extra
    fi
    unset INPUT_SEED
  done
}

MODELS=${MODELS:-}
if [ "$#" -gt 0 ]; then MODELS="$*"; fi
MODELS=${MODELS:-$ALL_MODELS}
for m in $MODELS; do
  spec "$m" >/dev/null || { echo "run_correctness.sh: unknown model '$m', valid: $ALL_MODELS" >&2; exit 1; }
done

progress_init correctness $(( $(echo $SEEDS | wc -w) * $(echo $MODELS | wc -w) ))
for m in $MODELS; do one_model "$m"; done
progress_done
echo "wrote $TSV"
