#!/bin/bash
# Batch-size sweep (paper EVAL_PLAN item E): where the systems converge as the
# GEMMs grow.  B independent sequences of the same length, the same token ids
# repeated, so every output row block must come back identical - that identity
# is the correctness check for the batching itself, and it is recorded per row
# as batch_rows_identical.
#
# steady_us stays per forward (per batch); per_seq_us = steady_us / B is the
# number the figure plots, because that is what converges.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

BATCHES=${BATCHES:-"1 2 4 8 16 32"}
BATCH_MODELS=${BATCH_MODELS:-"bert-mini distilgpt2"}
BATCH_SYSTEMS=${BATCH_SYSTEMS:-"ours torch-eager torch-compile torch-compile-ro jax"}

TSV="$OUT/batch.tsv"
tsv_init "$TSV" "model\tsystem\tbatch\tround\tsteady_us\tper_seq_us\tmaxabsdiff\tbatch_rows_identical\tstatus\tbinary\treference"

PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
pkg_version() {
  local python=$1 module=$2 name=$3
  [ -n "$python" ] && [ -x "$python" ] || { echo unknown; return; }
  "$python" -c "import $module; print('$name-' + $module.__version__)" 2>/dev/null || echo unknown
}
TORCH_VER=$(pkg_version "$TORCH_PYTHON" torch torch)
JAX_VER=$(pkg_version "$JAX_PYTHON" jax jax)
binary_of() {
  case "$1" in
    ours) echo "${PYPY_SHA:-unknown}" ;;
    jax) echo "$JAX_VER" ;;
    *) echo "$TORCH_VER" ;;
  esac
}
reference_file() {
  local dt=${RTENSOR_DTYPE:-float32}
  if [ "$dt" = float32 ]; then echo logits_pypy.bin
  else echo "logits_pypy_$dt.bin"; fi
}

has_system() { case " $BATCH_SYSTEMS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# $1 model $2 system $3 batch $4 round $5 status $6 output
record() {
  local model=$1 system=$2 batch=$3 round=$4 status=$5 out=$6
  local steady="" per_seq="" diff="" ident=""
  if [ "$status" = ok ]; then
    steady=$(steady_of "$out")
    per_seq=$(field_of "$out" per_seq_us)
    diff=$(diff_of "$out")
    ident=$(field_of "$out" batch_rows_identical)
  fi
  local binary=$(binary_of "$system")
  # Taken now, not at startup: an ours run rewrites the reference file, and the
  # hash has to be the one the baselines were actually compared against.
  local refpath="${MODEL_WEIGHTS:-}/$(reference_file)"
  local reference=unknown
  [ -f "$refpath" ] && reference=$(sha256sum "$refpath" | cut -c1-12)
  echo -e "$model\t$system\t$batch\t$round\t${steady:-}\t${per_seq:-}\t${diff:-}\t${ident:-}\t$status\t${binary}\t${reference}" >> "$TSV"
  bench_record batch model="$model" system="$system" batch="$batch" \
    round="$round" steady_us="${steady:-}" per_seq_us="${per_seq:-}" \
    maxabsdiff="${diff:-}" batch_rows_identical="${ident:-}" \
    status="$status" binary="${binary}" reference="${reference}"
}

# A system that cannot fit a batch (or, for ours, cannot express it on this
# pypy-c yet) records a failed row and the sweep goes on: the point of the
# sweep is the shape of the curve where it exists.
try() {
  local model=$1 system=$2 batch=$3 round=$4; shift 4
  local out status=ok
  if ! out=$("$@" 2>>"$OUT/batch_errors.log"); then
    status=failed
    echo "batch: $model $system B=$batch round $round FAILED (see batch_errors.log)" >&2
  fi
  record "$model" "$system" "$batch" "$round" "$status" "$out"
}

sweep() {
  local model=$1 pyscript=$2 torchscript=$3 jaxmodel=$4 weights=$5
  MODEL_TOL=$(tolerance_for "$model")
  export MODEL_TOL
  MODEL_WEIGHTS=$weights
  for b in $BATCHES; do
    for round in $(seq "$ROUNDS"); do
      progress_step "$model B=$b round $round/$ROUNDS"
      # ours first: it writes logits_pypy.bin, the reference the others are
      # checked against.
      if has_system ours; then
        try "$model" ours "$b" "$round" \
          "$RUN_PYPY" $JIT_FLAGS "$APP/$pyscript" "$weights" "$ITERS" "$WARMUP" "$b"
      fi
      for m in eager compile compile-ro; do
        has_system "torch-$m" || continue
        try "$model" "torch-$m" "$b" "$round" \
          "$TORCH_PYTHON" "$APP/$torchscript" "$m" "$weights" "$ITERS" "$WARMUP" "$b"
      done
      if has_system jax && [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ]; then
        try "$model" jax "$b" "$round" \
          "$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" jax "$weights" "$ITERS" "$WARMUP" "$b"
      fi
    done
  done
}

model_bert-mini() { sweep bert-mini bert.py bert_torch.py bert "$WEIGHTS/bert-mini"; }
model_distilgpt2() { sweep distilgpt2 gpt2.py gpt2_torch.py gpt2 "$WEIGHTS/distilgpt2"; }

if [ "$#" -gt 0 ]; then BATCH_MODELS="$*"; fi

progress_init batch $((ROUNDS * $(echo $BATCHES | wc -w) * $(echo $BATCH_MODELS | wc -w)))
for m in $BATCH_MODELS; do
  fn="model_$m"
  if ! declare -f "$fn" >/dev/null; then
    echo "run_batch.sh: unknown model '$m', valid: bert-mini distilgpt2" >&2
    exit 1
  fi
  "$fn"
done
progress_done
echo "wrote $TSV"
