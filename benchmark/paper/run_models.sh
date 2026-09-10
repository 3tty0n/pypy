#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
BASELINES_ONLY=${BASELINES_ONLY:-0}
if [ "$1" = "--baselines-only" ]; then BASELINES_ONLY=1; shift; fi
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/models.tsv"
tsv_init "$TSV" "model\tsystem\tround\tsteady_us\tmaxabsdiff\targmax_match\tcompile_ms\tfirst_run_ms"

run_ours() {
  local script=$1; shift
  "$RUN_PYPY" $JIT_FLAGS "$APP/$script" "$@"
}

record() {
  local model=$1 system=$2 round=$3 out=$4
  local steady=$(steady_of "$out")
  local diff=$(diff_of "$out")
  local argmax=$(echo "$out" | grep '^argmax' | sed 's/ maxabsdiff=.*//')
  echo "$argmax" > "$OUT/.last_argmax_$system"
  local match=""
  if [ -f "$OUT/.last_argmax_ours" ] && [ -f "$OUT/.last_argmax_$system" ] && [ "$system" != "ours" ]; then
    if [ "$(cat "$OUT/.last_argmax_ours")" = "$(cat "$OUT/.last_argmax_$system")" ]; then match=1; else match=0; fi
  fi
  local compile_ms=$(field_of "$out" compile_ms) first_ms=$(field_of "$out" first_run_ms)
  echo -e "$model\t$system\t$round\t${steady:-}\t${diff:-}\t${match}\t${compile_ms:-}\t${first_ms:-}" >> "$TSV"
  bench_record models model="$model" system="$system" round="$round" \
    steady_us="${steady:-}" maxabsdiff="${diff:-}" argmax_match="${match}" \
    compile_ms="${compile_ms:-}" first_run_ms="${first_ms:-}"
}

# JAX/XLA and IREE rows come from jax_models.py in the optional venv; the
# model name is its first argument, and IREE only covers the two models the
# paper's IREE column uses (see README).
IREE_MODELS=" bert-mini vit-tiny "
jax_rows() {
  local model=$1 jaxmodel=$2 weights=$3 round=$4; shift 4
  [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ] || return 0
  [ -n "$jaxmodel" ] || return 0
  local out
  out=$("$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" jax "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
    record "$model" jax "$round" "$out"
  case "$IREE_MODELS" in
    *" $model "*)
      out=$("$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" iree "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
        record "$model" iree "$round" "$out" ;;
  esac
}

model() {
  local model=$1 pyscript=$2 torchscript=$3 jaxmodel=$4 weights=$5; shift 5
  for round in $(seq "$ROUNDS"); do
    progress_step "$model round $round/$ROUNDS"
    if [ "$BASELINES_ONLY" != 1 ]; then
      out=$(run_ours "$pyscript" "$weights" "$ITERS" "$WARMUP" "$@")
      record "$model" ours "$round" "$out"
      out=$("$TORCH_PYTHON" "$APP/$torchscript" eager "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null)
      record "$model" torch-eager "$round" "$out"
      out=$("$TORCH_PYTHON" "$APP/$torchscript" compile "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null)
      record "$model" torch-compile "$round" "$out"
    fi
    jax_rows "$model" "$jaxmodel" "$weights" "$round" "$@"
    if [ -n "${TRT_PYTHON:-}" ]; then
      out=$("$TRT_PYTHON" "$APP/$torchscript" tensorrt "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
        record "$model" torch-tensorrt "$round" "$out"
    fi
  done
}

# fourth column: the jax_models.py model, "" where there is no JAX twin
model_distilgpt2() { model distilgpt2 gpt2.py gpt2_torch.py gpt2 "$WEIGHTS/distilgpt2"; }
model_tiny-gpt2() { model tiny-gpt2 gpt2.py gpt2_torch.py gpt2 "$WEIGHTS/tiny-gpt2"; }
model_smollm2-135m() { model smollm2-135m llama.py llama_torch.py llama "$WEIGHTS/smollm2-135m"; }
model_bert-tiny() { model bert-tiny bert.py bert_torch.py bert "$WEIGHTS/bert-tiny"; }
model_bert-mini() { model bert-mini bert.py bert_torch.py bert "$WEIGHTS/bert-mini"; }
model_resnet18-b1() { model resnet18-b1 resnet.py resnet_torch.py resnet "$WEIGHTS/resnet18" 1; }
model_resnet18-b8() { model resnet18-b8 resnet.py resnet_torch.py resnet "$WEIGHTS/resnet18" 8; }
model_mixer_b16() { model mixer_b16 mixer.py mixer_torch.py mixer "$WEIGHTS/mixer_b16"; }
model_vit-tiny() { model vit-tiny vit.py vit_torch.py vit "$WEIGHTS/vit-tiny"; }

ALL_MODELS="distilgpt2 tiny-gpt2 smollm2-135m bert-tiny bert-mini resnet18-b1 resnet18-b8 mixer_b16 vit-tiny"
MODELS=${MODELS:-}
if [ "$#" -gt 0 ]; then MODELS="$*"; fi
MODELS=${MODELS:-$ALL_MODELS}

progress_init models $((ROUNDS * $(echo $MODELS | wc -w)))

for m in $MODELS; do
  fn="model_$m"
  if ! declare -f "$fn" >/dev/null; then
    echo "run_models.sh: unknown model '$m', valid: $ALL_MODELS" >&2
    exit 1
  fi
  "$fn"
done

progress_done
rm -f "$OUT"/.last_argmax_*
echo "wrote $TSV"
