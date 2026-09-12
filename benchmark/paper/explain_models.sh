#!/bin/bash
# Dynamo graph structure per model, once: graphs/breaks/ops from
# torch._dynamo.explain, no timing. Writes $OUT/explain.tsv.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

TSV="$OUT/explain.tsv"
tsv_init "$TSV" "model\tgraphs\tbreaks\tops"

explain_row() {
  local model=$1 torchscript=$2 weights=$3; shift 3
  local out
  out=$("$TORCH_PYTHON" "$APP/$torchscript" explain "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null)
  local graphs=$(field_of "$out" graphs) breaks=$(field_of "$out" breaks) ops=$(field_of "$out" ops)
  echo -e "$model\t${graphs:-}\t${breaks:-}\t${ops:-}" >> "$TSV"
  bench_record explain model="$model" graphs="${graphs:-}" breaks="${breaks:-}" ops="${ops:-}"
}

explain_row distilgpt2 gpt2_torch.py "$WEIGHTS/distilgpt2"
explain_row tiny-gpt2 gpt2_torch.py "$WEIGHTS/tiny-gpt2"
explain_row smollm2-135m llama_torch.py "$WEIGHTS/smollm2-135m"
explain_row bert-tiny bert_torch.py "$WEIGHTS/bert-tiny"
explain_row bert-mini bert_torch.py "$WEIGHTS/bert-mini"
explain_row resnet18-b1 resnet_torch.py "$WEIGHTS/resnet18" 1
explain_row resnet18-b8 resnet_torch.py "$WEIGHTS/resnet18" 8
explain_row mixer_b16 mixer_torch.py "$WEIGHTS/mixer_b16"
explain_row vit-tiny vit_torch.py "$WEIGHTS/vit-tiny"

echo "wrote $TSV"
