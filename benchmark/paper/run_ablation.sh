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

# One ablation row, to both writers: exp variant model round steady diff note
ab_row() {
  echo -e "$1\t$2\t$3\t$4\t$5\t$6\t$7" >> "$TSV"
  bench_record ablation experiment="$1" variant="$2" model="$3" round="$4" \
    steady_us="$5" maxabsdiff="$6" note="$7"
}

exp_fusion() {
  for round in $(seq "$ROUNDS"); do
    out=$(run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
    ab_row fusion on distilgpt2 "$round" "$(steady_of "$out")" "" ""
    out=$(run_gpt2 "-S --jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=60000,$NOFUSE_OPTS" "$WEIGHTS/distilgpt2")
    ab_row fusion off distilgpt2 "$round" "$(steady_of "$out")" "" "enable_opts minus tensor"
  done
}

exp_flat_block() {
  for block in 256 4096; do
    for round in $(seq "$ROUNDS"); do
      out=$(RTENSOR_FLAT_BLOCK=$block run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
      ab_row flat_block "$block" distilgpt2 "$round" "$(steady_of "$out")" "" ""
      out=$(RTENSOR_FLAT_BLOCK=$block run_resnet "$JIT_FLAGS" "$WEIGHTS/resnet18")
      ab_row flat_block "$block" resnet18 "$round" "$(steady_of "$out")" "" ""
    done
  done
}

exp_budget_mb() {
  for budget in 8 64; do
    for round in $(seq "$ROUNDS"); do
      out=$(RTENSOR_BUDGET_MB=$budget run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
      ab_row budget_mb "$budget" distilgpt2 "$round" "$(steady_of "$out")" "" ""
    done
  done
}

exp_precision() {
  for dtype in float32 float16; do
    for round in $(seq "$ROUNDS"); do
      if out=$(RTENSOR_DTYPE=$dtype run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2" 2>&1); then
        tdiff=$(RTENSOR_DTYPE=$dtype "$TORCH_PYTHON" "$APP/gpt2_torch.py" eager "$WEIGHTS/distilgpt2" "$ITERS" "$WARMUP" 2>/dev/null)
        ab_row precision "$dtype" distilgpt2 "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" ""
      else
        ab_row "precision" "$dtype" "distilgpt2" "$round" "" "" "unsupported: run failed"
      fi
      if out=$(RTENSOR_DTYPE=$dtype "$RUN_PYPY" $JIT_FLAGS "$APP/llama.py" "$WEIGHTS/smollm2-135m" "$ITERS" "$WARMUP" 2>&1); then
        tdiff=$(RTENSOR_DTYPE=$dtype "$TORCH_PYTHON" "$APP/llama_torch.py" eager "$WEIGHTS/smollm2-135m" "$ITERS" "$WARMUP" 2>/dev/null)
        ab_row precision "$dtype" smollm2-135m "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" ""
      else
        ab_row "precision" "$dtype" "smollm2-135m" "$round" "" "" "unsupported: run failed"
      fi
    done
  done
}

ALL_EXPERIMENTS="fusion flat_block budget_mb precision"
EXPERIMENTS=${EXPERIMENTS:-}
if [ "$#" -gt 0 ]; then EXPERIMENTS="$*"; fi
EXPERIMENTS=${EXPERIMENTS:-$ALL_EXPERIMENTS}

progress_init ablation $(echo $EXPERIMENTS | wc -w)

for e in $EXPERIMENTS; do
  fn="exp_$e"
  if ! declare -f "$fn" >/dev/null; then
    echo "run_ablation.sh: unknown experiment '$e', valid: $ALL_EXPERIMENTS" >&2
    exit 1
  fi
  progress_step "$e"
  "$fn"
done

progress_done
echo "wrote $TSV"
