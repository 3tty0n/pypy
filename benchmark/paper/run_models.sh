#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/models.tsv"
tsv_init "$TSV" "model\tsystem\tround\tsteady_us\tmaxabsdiff\targmax_match"

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
  echo -e "$model\t$system\t$round\t${steady:-}\t${diff:-}\t${match}" >> "$TSV"
}

model() {
  local model=$1 pyscript=$2 torchscript=$3 weights=$4; shift 4
  for round in $(seq "$ROUNDS"); do
    out=$(run_ours "$pyscript" "$weights" "$ITERS" "$WARMUP" "$@")
    record "$model" ours "$round" "$out"
    out=$("$TORCH_PYTHON" "$APP/$torchscript" eager "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null)
    record "$model" torch-eager "$round" "$out"
    out=$("$TORCH_PYTHON" "$APP/$torchscript" compile "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null)
    record "$model" torch-compile "$round" "$out"
  done
}

model_distilgpt2() { model distilgpt2 gpt2.py gpt2_torch.py "$WEIGHTS/distilgpt2"; }
model_tiny-gpt2() { model tiny-gpt2 gpt2.py gpt2_torch.py "$WEIGHTS/tiny-gpt2"; }
model_smollm2-135m() { model smollm2-135m llama.py llama_torch.py "$WEIGHTS/smollm2-135m"; }
model_bert-tiny() { model bert-tiny bert.py bert_torch.py "$WEIGHTS/bert-tiny"; }
model_bert-mini() { model bert-mini bert.py bert_torch.py "$WEIGHTS/bert-mini"; }
model_resnet18-b1() { model resnet18-b1 resnet.py resnet_torch.py "$WEIGHTS/resnet18" 1; }
model_resnet18-b8() { model resnet18-b8 resnet.py resnet_torch.py "$WEIGHTS/resnet18" 8; }
model_mixer_b16() { model mixer_b16 mixer.py mixer_torch.py "$WEIGHTS/mixer_b16"; }
model_vit-tiny() { model vit-tiny vit.py vit_torch.py "$WEIGHTS/vit-tiny"; }

ALL_MODELS="distilgpt2 tiny-gpt2 smollm2-135m bert-tiny bert-mini resnet18-b1 resnet18-b8 mixer_b16 vit-tiny"
MODELS=${MODELS:-}
if [ "$#" -gt 0 ]; then MODELS="$*"; fi
MODELS=${MODELS:-$ALL_MODELS}

for m in $MODELS; do
  fn="model_$m"
  if ! declare -f "$fn" >/dev/null; then
    echo "run_models.sh: unknown model '$m', valid: $ALL_MODELS" >&2
    exit 1
  fi
  "$fn"
done

rm -f "$OUT"/.last_argmax_*
echo "wrote $TSV"
