#!/bin/bash
# Per-forward warm-up trace: every forward's latency from a fresh process
# until steady state, for tiny-gpt2/distilgpt2, on ours/torch-eager/
# torch-compile/torch-compile-ro/jax, cold and warm kernel caches, ROUNDS
# times.  Replaces "first_run_ms" as the measure of start-up cost.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
# A private symlink rather than config.sh's shared $REPO/pypy-c-paper: this
# run can take hours and another concurrent bench.sh invocation deleting the
# shared link out from under it (its own EXIT trap) breaks "ours" mid-sweep.
WARMUP_PYPY_LINK="$REPO/.pypy-c-warmup-$$"
if [ -n "$PYPY" ] && [ "$(dirname "$PYPY")" != "$REPO" ]; then
  ln -sf "$PYPY" "$WARMUP_PYPY_LINK"
  RUN_PYPY="$WARMUP_PYPY_LINK"
else
  RUN_PYPY="$PYPY"
fi
trap 'rm -f "$WARMUP_PYPY_LINK"' EXIT

N=${N:-300}
# compiled/launches: per-forward _metatensor.kernel_compile_count()/
# launch_count() deltas, the "ours" driver only (blank for other systems) -
# compiled>0 means that forward spawned the Triton compile subprocess (a
# disk-cache miss); launches counts every kernel dispatch, hit or miss.
TSV="$OUT/warmup.tsv"
tsv_init "$TSV" "model\tsystem\tcache\tround\titer\tus\tcompiled\tlaunches\tbinary"
SERIES_DIR="$OUT/.warmup_series"
mkdir -p "$SERIES_DIR"

# Per-row provenance, same convention as run_models.sh.
PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
TORCH_VER=$([ -n "$TORCH_PYTHON" ] && [ -x "$TORCH_PYTHON" ] && \
  "$TORCH_PYTHON" -c "import torch; print('torch-' + torch.__version__)" 2>/dev/null || echo unknown)
JAX_VER=$([ -n "${JAX_PYTHON:-}" ] && [ -x "${JAX_PYTHON:-}" ] && \
  "$JAX_PYTHON" -c "import jax; print('jax-' + jax.__version__)" 2>/dev/null || echo unknown)
binary_of() { case "$1" in ours) echo "${PYPY_SHA:-unknown}" ;; jax) echo "$JAX_VER" ;; *) echo "$TORCH_VER" ;; esac; }

record_one() {
  # model system cache round <<< trace stdout on stdin
  local model=$1 system=$2 cache=$3 round=$4 eager=$5
  local series="$SERIES_DIR/${model}_${cache}_${round}_${system}"
  local binary=$(binary_of "$system")
  local out
  if [ -n "$eager" ]; then
    out=$("$RTENSOR_PYTHON" "$HERE/warmup_record.py" \
      "$model" "$system" "$cache" "$round" "$TSV" "$series" "$binary" "$eager")
  else
    out=$("$RTENSOR_PYTHON" "$HERE/warmup_record.py" \
      "$model" "$system" "$cache" "$round" "$TSV" "$series" "$binary")
  fi
  local first=$(field_of "$out" first_us) total=$(field_of "$out" total_us)
  local steady=$(field_of "$out" steady_us)
  local steady_at=$(echo "$out" | grep -o 'steady_at=[a-z0-9]*' | cut -d= -f2)
  local crossover=$(echo "$out" | grep -o 'crossover=[a-z0-9]*' | cut -d= -f2)
  local cold_compiles=$(field_of "$out" cold_compiles)
  local first_compiles=$(field_of "$out" first_forward_compiles)
  bench_record warmup model="$model" system="$system" cache="$cache" \
    round="$round" total_us="${total:-}" steady_us="${steady:-}" \
    steady_at="${steady_at:-}" first_us="${first:-}" crossover="${crossover:-}" \
    cold_compiles="${cold_compiles:-}" first_forward_compiles="${first_compiles:-}" \
    binary="$binary"
  echo "$series"
}

# Empty the on-disk kernel caches so the next process starts cold: ours
# keys kernels by $TMPDIR/rtensor-k-<hash> (rpython/metatensor/kernels.py),
# torch.compile/triton by TORCHINDUCTOR_CACHE_DIR/TRITON_CACHE_DIR, and
# JAX's persistent compile cache (off unless JAX_COMPILATION_CACHE_DIR is
# set) by that directory.
ORIG_TMPDIR=${TMPDIR:-/tmp}
cold_env() {
  # -p pins the base dir explicitly: mktemp otherwise reads $TMPDIR itself,
  # which this function just replaced, so round 2's mktemp would try to
  # create a directory inside round 1's already-deleted one.
  TMPDIR=$(mktemp -d -p "$ORIG_TMPDIR")
  TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p "$ORIG_TMPDIR")
  TRITON_CACHE_DIR=$(mktemp -d -p "$ORIG_TMPDIR")
  unset JAX_COMPILATION_CACHE_DIR
}

warm_env() {
  # Persistent per-model directories, reused across rounds so the cache
  # really is warm by the time it's measured.
  local model=$1
  TMPDIR="$OUT/.warmup_cache/$model/tmpdir"
  TORCHINDUCTOR_CACHE_DIR="$OUT/.warmup_cache/$model/inductor"
  TRITON_CACHE_DIR="$OUT/.warmup_cache/$model/triton"
  JAX_COMPILATION_CACHE_DIR="$OUT/.warmup_cache/$model/jax"
  mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$JAX_COMPILATION_CACHE_DIR"
}

export_cache_env() {
  export TMPDIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
  if [ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]; then
    export JAX_COMPILATION_CACHE_DIR
  else
    unset JAX_COMPILATION_CACHE_DIR
  fi
}

prime() { "$@" >/dev/null 2>&1 || true; }

run_ours()   { WARMUP_TRACE=$N "$RUN_PYPY" $JIT_FLAGS "$APP/gpt2.py" "$1" "$N" 1; }
run_torch()  { WARMUP_TRACE=$N "$TORCH_PYTHON" "$APP/gpt2_torch.py" "$1" "$2" "$N" 1; }
run_jax()    { WARMUP_TRACE=$N "$JAX_PYTHON" "$APP/jax_models.py" gpt2 jax "$1" "$N" 1; }

one_model() {
  local model=$1 weights="$WEIGHTS/$1"
  for cache in cold warm; do
    for round in $(seq "$ROUNDS"); do
      progress_step "$model $cache round $round/$ROUNDS"
      if [ "$cache" = cold ]; then cold_env; else warm_env "$model"; fi
      export_cache_env
      if [ "$cache" = warm ]; then
        prime run_ours "$weights"
        prime run_torch eager "$weights"
        prime run_torch compile "$weights"
        prime run_torch compile-ro "$weights"
        [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ] && prime run_jax "$weights"
      fi

      eager=$(run_torch eager "$weights" | \
        record_one "$model" torch-eager "$cache" "$round" "")
      run_ours "$weights" | record_one "$model" ours "$cache" "$round" "$eager" >/dev/null
      run_torch compile "$weights" | \
        record_one "$model" torch-compile "$cache" "$round" "$eager" >/dev/null
      run_torch compile-ro "$weights" | \
        record_one "$model" torch-compile-ro "$cache" "$round" "$eager" >/dev/null
      if [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ]; then
        run_jax "$weights" | \
          record_one "$model" jax "$cache" "$round" "$eager" >/dev/null
      fi
      if [ "$cache" = cold ]; then
        rm -rf "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
      fi
    done
  done
}

MODELS=${MODELS:-"tiny-gpt2 distilgpt2"}
progress_init warmup $((2 * ROUNDS * $(echo $MODELS | wc -w)))
for m in $MODELS; do one_model "$m"; done
progress_done
rm -rf "$OUT/.warmup_cache"
echo "wrote $TSV"
