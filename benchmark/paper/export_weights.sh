#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

export_one() {
  local script=$1 name=$2 model=$3
  shift 3
  if [ -f "$WEIGHTS/$name/index.json" ]; then
    echo "skip $name (already exported)"
    return
  fi
  echo "exporting $name <- $model"
  "$RTENSOR_PYTHON" "$APP/$script" "$WEIGHTS/$name" --model "$model" "$@"
}

# The population, as benchmark/paper/select_models.py admits it: every
# checkpoint below is published by the organisation that trained it, has all
# of its weights, and is pulled at least 100,000 times a month.
export_one gpt2_export.py gpt2 openai-community/gpt2
export_one gpt2_export.py gpt2-medium openai-community/gpt2-medium
export_one gpt2_export.py distilgpt2 distilgpt2
export_one llama_export.py smollm2-135m HuggingFaceTB/SmolLM2-135M
export_one llama_export.py smollm2-360m HuggingFaceTB/SmolLM2-360M
export_one llama_export.py smollm2-1.7b HuggingFaceTB/SmolLM2-1.7B
export_one llama_export.py qwen2.5-0.5b Qwen/Qwen2.5-0.5B-Instruct
export_one bert_export.py bert-base google-bert/bert-base-uncased
export_one bert_export.py bert-mini google/bert_uncased_L-4_H-256_A-4
export_one resnet_export.py resnet18 resnet18
export_one mixer_export.py mixer_b16 mixer_b16_224.goog_in21k_ft_in1k
export_one vit_export.py vit-base google/vit-base-patch16-224
export_one vit_export.py deit-tiny facebook/deit-tiny-patch16-224

# The three checkpoints the selection rule excludes are not exported here.
# To reproduce one of those rows, export it by hand and name it:
#   applevel/gpt2_export.py $WEIGHTS/tiny-gpt2 --model sshleifer/tiny-gpt2
#   applevel/bert_export.py $WEIGHTS/bert-tiny --model prajjwal1/bert-tiny
#   applevel/vit_export.py  $WEIGHTS/vit-tiny  --model WinKawaks/vit-tiny-patch16-224
# then: bench.sh models tiny-gpt2
