#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/ablation.tsv"
tsv_init "$TSV" "experiment\tvariant\tmodel\tround\tsteady_us\tmaxabsdiff\tnote"

NOFUSE_OPTS="enable_opts=intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"

run_gpt2() {
  local jitflags=$1 weights=$2
  "$RUN_PYPY" $jitflags "$APP/gpt2.py" "$weights" "$ITERS" "$WARMUP"
}
run_resnet() {
  local jitflags=$1 weights=$2
  "$RUN_PYPY" $jitflags "$APP/resnet.py" "$weights" "$ITERS" "$WARMUP" 1
}

exp_fusion() {
  for round in $(seq "$ROUNDS"); do
    out=$(run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
    echo -e "fusion\ton\tdistilgpt2\t$round\t$(steady_of "$out")\t\t" >> "$TSV"
    out=$(run_gpt2 "-S --jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=60000,$NOFUSE_OPTS" "$WEIGHTS/distilgpt2")
    echo -e "fusion\toff\tdistilgpt2\t$round\t$(steady_of "$out")\t\tenable_opts minus tensor" >> "$TSV"
  done
}

exp_flat_block() {
  for block in 256 4096; do
    for round in $(seq "$ROUNDS"); do
      out=$(RTENSOR_FLAT_BLOCK=$block run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
      echo -e "flat_block\t$block\tdistilgpt2\t$round\t$(steady_of "$out")\t\t" >> "$TSV"
      out=$(RTENSOR_FLAT_BLOCK=$block run_resnet "$JIT_FLAGS" "$WEIGHTS/resnet18")
      echo -e "flat_block\t$block\tresnet18\t$round\t$(steady_of "$out")\t\t" >> "$TSV"
    done
  done
}

exp_budget_mb() {
  for budget in 8 64; do
    for round in $(seq "$ROUNDS"); do
      out=$(RTENSOR_BUDGET_MB=$budget run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
      echo -e "budget_mb\t$budget\tdistilgpt2\t$round\t$(steady_of "$out")\t\t" >> "$TSV"
    done
  done
}

exp_precision() {
  for dtype in float32 float16; do
    for round in $(seq "$ROUNDS"); do
      if out=$(RTENSOR_DTYPE=$dtype run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2" 2>&1); then
        tdiff=$(RTENSOR_DTYPE=$dtype "$TORCH_PYTHON" "$APP/gpt2_torch.py" eager "$WEIGHTS/distilgpt2" "$ITERS" "$WARMUP" 2>/dev/null)
        echo -e "precision\t$dtype\tdistilgpt2\t$round\t$(steady_of "$out")\t$(diff_of "$tdiff")\t" >> "$TSV"
      else
        echo -e "precision\t$dtype\tdistilgpt2\t$round\t\t\tunsupported: run failed" >> "$TSV"
      fi
      if out=$(RTENSOR_DTYPE=$dtype "$RUN_PYPY" $JIT_FLAGS "$APP/llama.py" "$WEIGHTS/smollm2-135m" "$ITERS" "$WARMUP" 2>&1); then
        tdiff=$(RTENSOR_DTYPE=$dtype "$TORCH_PYTHON" "$APP/llama_torch.py" eager "$WEIGHTS/smollm2-135m" "$ITERS" "$WARMUP" 2>/dev/null)
        echo -e "precision\t$dtype\tsmollm2-135m\t$round\t$(steady_of "$out")\t$(diff_of "$tdiff")\t" >> "$TSV"
      else
        echo -e "precision\t$dtype\tsmollm2-135m\t$round\t\t\tunsupported: run failed" >> "$TSV"
      fi
    done
  done
}

ALL_EXPERIMENTS="fusion flat_block budget_mb precision"
EXPERIMENTS=${EXPERIMENTS:-}
if [ "$#" -gt 0 ]; then EXPERIMENTS="$*"; fi
EXPERIMENTS=${EXPERIMENTS:-$ALL_EXPERIMENTS}

for e in $EXPERIMENTS; do
  fn="exp_$e"
  if ! declare -f "$fn" >/dev/null; then
    echo "run_ablation.sh: unknown experiment '$e', valid: $ALL_EXPERIMENTS" >&2
    exit 1
  fi
  "$fn"
done

echo "wrote $TSV"
