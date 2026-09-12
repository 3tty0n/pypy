#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/ablation.tsv"
tsv_init "$TSV" "experiment\tvariant\tmodel\tround\tsteady_us\tmaxabsdiff\tnote\tbinary\tlaunches_per_iter"

# Every row here runs on our own translated pypy-c; identify it by its hash
# the same way run_models.sh identifies the "ours" system.
PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)

NOFUSE_OPTS="enable_opts=intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"

run_gpt2() {
  local jitflags=$1 weights=$2
  "$RUN_PYPY" $jitflags "$APP/gpt2.py" "$weights" "$ITERS" "$WARMUP"
}
run_resnet() {
  local jitflags=$1 weights=$2 batch=${3:-1}
  "$RUN_PYPY" $jitflags "$APP/resnet.py" "$weights" "$ITERS" "$WARMUP" "$batch"
}

# One ablation row, to both writers: exp variant model round steady diff note
# [launches] - launches_per_iter is optional (8th arg), blank where the
# experiment does not measure it.
ab_row() {
  local launches=${8:-}
  echo -e "$1\t$2\t$3\t$4\t$5\t$6\t$7\t${PYPY_SHA:-unknown}\t$launches" >> "$TSV"
  bench_record ablation experiment="$1" variant="$2" model="$3" round="$4" \
    steady_us="$5" maxabsdiff="$6" note="$7" binary="${PYPY_SHA:-unknown}" \
    launches_per_iter="$launches"
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

exp_tf32() {
  for batch in 8 1; do
    for tf32 in 1 0; do
      variant=tf32; [ "$tf32" = 0 ] && variant=fp32
      for round in $(seq "$ROUNDS"); do
        out=$(RTENSOR_TF32=$tf32 run_resnet "$JIT_FLAGS" "$WEIGHTS/resnet18" "$batch")
        torch_tf32=1; [ "$tf32" = 0 ] && torch_tf32=0
        tdiff=$(TORCH_CUDNN_TF32=$torch_tf32 "$TORCH_PYTHON" "$APP/resnet_torch.py" eager "$WEIGHTS/resnet18" "$ITERS" "$WARMUP" "$batch" 2>/dev/null)
        ab_row tf32 "$variant" "resnet18-b$batch" "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" ""
      done
    done
  done
  for round in $(seq "$ROUNDS"); do
    out=$(TORCH_CUDNN_TF32=0 "$TORCH_PYTHON" "$APP/resnet_torch.py" compile "$WEIGHTS/resnet18" "$ITERS" "$WARMUP" 8 2>/dev/null)
    ab_row tf32 "torch-fp32" "resnet18-b8" "$round" "$(steady_of "$out")" "$(diff_of "$out")" ""
  done
}

exp_max_inputs() {
  for mi in 4 6 8; do
    local variant="mi$mi"
    for round in $(seq "$ROUNDS"); do
      out=$(RTENSOR_MAX_INPUTS=$mi run_gpt2 "$JIT_FLAGS" "$WEIGHTS/distilgpt2")
      tdiff=$("$TORCH_PYTHON" "$APP/gpt2_torch.py" eager "$WEIGHTS/distilgpt2" "$ITERS" "$WARMUP" 2>/dev/null)
      ab_row max_inputs "$variant" distilgpt2 "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" "" "$(field_of "$out" launches_per_iter)"

      out=$(RTENSOR_MAX_INPUTS=$mi "$RUN_PYPY" $JIT_FLAGS "$APP/mixer.py" "$WEIGHTS/mixer_b16" "$ITERS" "$WARMUP" 2>&1)
      tdiff=$("$TORCH_PYTHON" "$APP/mixer_torch.py" eager "$WEIGHTS/mixer_b16" "$ITERS" "$WARMUP" 2>/dev/null)
      ab_row max_inputs "$variant" mixer_b16 "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" "" "$(field_of "$out" launches_per_iter)"

      out=$(RTENSOR_MAX_INPUTS=$mi "$RUN_PYPY" $JIT_FLAGS "$APP/bert.py" "$WEIGHTS/bert-mini" "$ITERS" "$WARMUP" 2>&1)
      tdiff=$("$TORCH_PYTHON" "$APP/bert_torch.py" eager "$WEIGHTS/bert-mini" "$ITERS" "$WARMUP" 2>/dev/null)
      ab_row max_inputs "$variant" bert-mini "$round" "$(steady_of "$out")" "$(diff_of "$tdiff")" "" "$(field_of "$out" launches_per_iter)"
    done
  done
}

ALL_EXPERIMENTS="fusion flat_block budget_mb precision tf32 max_inputs"
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
