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
tsv_init "$TSV" "model\tsystem\tround\tsteady_us\tmaxabsdiff\targmax_match\tcompile_ms\tfirst_run_ms\tlaunches_per_iter\ttolerance\tpass\tbinary\treference"

# Per-row provenance: which build produced the row and which reference file it
# was checked against.  Ours is a binary we build, so it is identified by its
# hash; the baselines come out of a wheel, so the version string is the thing
# that identifies them.
PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
pkg_version() {
  local python=$1 module=$2 name=$3
  [ -n "$python" ] && [ -x "$python" ] || { echo unknown; return; }
  "$python" -c "import $module; print('$name-' + $module.__version__)" 2>/dev/null || echo unknown
}
TORCH_VER=$(pkg_version "$TORCH_PYTHON" torch torch)
JAX_VER=$(pkg_version "$JAX_PYTHON" jax jax)
TRT_VER=$(pkg_version "$TRT_PYTHON" torch_tensorrt torch_tensorrt)
binary_of() {
  case "$1" in
    ours) echo "${PYPY_SHA:-unknown}" ;;
    jax|iree) echo "$JAX_VER" ;;
    torch-tensorrt) echo "$TRT_VER" ;;
    *) echo "$TORCH_VER" ;;
  esac
}

# applevel/common.py's reference_name(): float32 keeps the plain name.
reference_file() {
  local dt=${RTENSOR_DTYPE:-float32}
  if [ "$dt" = float32 ]; then echo logits_pypy.bin
  else echo "logits_pypy_$dt.bin"; fi
}

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
  # launches_per_iter is the dead-code guard (kernels the forward actually
  # launches); tol/pass come from the shared tolerance_for table.
  local launches=$(field_of "$out" launches_per_iter)
  local tol=$(field_of "$out" tol) passed=$(field_of "$out" pass)
  local binary=$(binary_of "$system")
  # Taken now, not at startup: the ours run rewrites the reference file, and
  # the hash has to be the one the baselines were actually compared against.
  local refpath="${MODEL_WEIGHTS:-}/$(reference_file)"
  local reference=unknown
  [ -f "$refpath" ] && reference=$(sha256sum "$refpath" | cut -c1-12)
  echo -e "$model\t$system\t$round\t${steady:-}\t${diff:-}\t${match}\t${compile_ms:-}\t${first_ms:-}\t${launches:-}\t${tol:-}\t${passed:-}\t${binary}\t${reference}" >> "$TSV"
  bench_record models model="$model" system="$system" round="$round" \
    steady_us="${steady:-}" maxabsdiff="${diff:-}" argmax_match="${match}" \
    compile_ms="${compile_ms:-}" first_run_ms="${first_ms:-}" \
    launches_per_iter="${launches:-}" tolerance="${tol:-}" pass="${passed:-}" \
    binary="${binary}" reference="${reference}"
}

# JAX/XLA and IREE rows come from jax_models.py in the optional venv; the
# model name is its first argument. IREE_MODELS excludes:
# - resnet18 (b1/b8): IREE's CUDA backend rejects the conv2d lowering
#   ("linalg.conv_2d_nhwc_hwcf ... strides failed to satisfy constraint:
#   64-bit signless int elements") for a StableHLO conv from jax.export, the
#   same compiler limitation as micro variant 9 (CNN).
IREE_MODELS=" distilgpt2 tiny-gpt2 bert-tiny bert-mini vit-tiny mixer_b16 smollm2-135m "
BASELINES=${BASELINES:-"triton tensorrt compile-ro compile-mat jax iree"}
has_baseline() { case " $BASELINES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
jax_rows() {
  local model=$1 jaxmodel=$2 weights=$3 round=$4; shift 4
  [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ] || return 0
  [ -n "$jaxmodel" ] || return 0
  local out
  if has_baseline jax; then
    out=$("$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" jax "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
      record "$model" jax "$round" "$out"
  fi
  if has_baseline iree; then
    case "$IREE_MODELS" in
      *" $model "*)
        out=$("$JAX_PYTHON" "$APP/jax_models.py" "$jaxmodel" iree "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
          record "$model" iree "$round" "$out" ;;
    esac
  fi
}

model() {
  local model=$1 pyscript=$2 torchscript=$3 jaxmodel=$4 weights=$5; shift 5
  # One tolerance per (workload class, dtype), read by every system's script.
  MODEL_TOL=$(tolerance_for "$model")
  export MODEL_TOL
  MODEL_WEIGHTS=$weights
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
    if has_baseline compile-ro; then
      out=$("$TORCH_PYTHON" "$APP/$torchscript" compile-ro "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
        record "$model" torch-compile-ro "$round" "$out"
    fi
    if has_baseline compile-mat; then
      out=$("$TORCH_PYTHON" "$APP/$torchscript" compile-mat "$weights" "$ITERS" "$WARMUP" "$@" 2>/dev/null) &&
        record "$model" torch-compile-mat "$round" "$out"
    fi
    jax_rows "$model" "$jaxmodel" "$weights" "$round" "$@"
    if has_baseline tensorrt && [ -n "${TRT_PYTHON:-}" ]; then
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
