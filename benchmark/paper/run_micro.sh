#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"

# --applevel runs the same models through the translated pypy-c instead of the
# standalone metatensor-bench, so the comparison against torch_bench.py (which
# pays for CPython) is like for like. It only adds "app" rows; run it alongside
# a normal micro run, not instead of one.
APPLEVEL=${APPLEVEL:-0}
if [ "$1" = "--applevel" ]; then APPLEVEL=1; shift; fi
# --baselines-only adds just the triton/tensorrt/jax/iree rows to an existing
# result set, for a grid whose ours/torch rows were measured earlier.
BASELINES_ONLY=${BASELINES_ONLY:-0}
if [ "$1" = "--baselines-only" ]; then BASELINES_ONLY=1; shift; fi
if [ "$APPLEVEL" = 1 ]; then
  paper_setup_pypy
  trap paper_cleanup_pypy EXIT
  # A pypy-c older than the _metatensor module fails on every single point, and
  # app_point would otherwise swallow all of it and leave an empty app column.
  if ! "$RUN_PYPY" -c 'import _metatensor; _metatensor.mem_total()' >/dev/null 2>&1; then
    echo "run_micro.sh: $RUN_PYPY predates the _metatensor module it needs;" \
         "run 'bench.sh setup' to retranslate it" >&2
    exit 1
  fi
fi

# Every driver takes WARMUP as its optional 6th argument and warms up exactly
# that many iterations, so no two systems are timed after a different warm-up;
# the count is recorded as `warmup` in results.jsonl.
HEADER="mode\tvariant\tk\tn\titers\twarm_s\tsteady_us\tkernels\tacc\tcompiled_in_timed\tlaunches_per_iter\tgraphs\tbreaks\tcompile_ms\tfirst_run_ms"
TSV="$OUT/micro.tsv"
tsv_init "$TSV" "$HEADER"

# metatensor-bench fits its working set to the GPU and reports the element
# count it actually ran with as field 4, so torch has to be given that same
# count rather than the requested one.
ours() { "$BENCH" "$@"; }
torchrun() {
  local line
  line=$("$TORCH_PYTHON" "$HERE/../torch_bench.py" "$@" 2>/dev/null | tail -1)
  [ -n "$line" ] || return 0
  echo "$line" | tr ' ' '\t' >> "$TSV"
  record_micro_line "$line"
}

# The other baselines print the torch line shape (plus compile_ms and
# first_run_ms), so they land in the same tsv and jsonl as another mode.
# A missing venv or an unsupported variant just leaves the row out.
baseline_run() {
  local python=$1 script=$2 line; shift 2
  [ -n "$python" ] && [ -x "$python" ] || return 0
  line=$("$python" "$HERE/../$script" "$@" 2>/dev/null | tail -1)
  [ -n "$line" ] || return 0
  echo "$line" | tr ' ' '\t' >> "$TSV"
  record_micro_line "$line"
}
# variant 9 (CNN) is the only one excluded: IREE's CUDA backend rejects the
# conv2d lowering ("linalg.conv_2d_nhwc_hwcf ... strides failed to satisfy
# constraint: 64-bit signless int elements") for a StableHLO conv exported by
# jax.export, a compiler limitation, not something this harness controls.
# Every other variant, at every dtype in the precision sweep, compiles and
# matches jax to the printed digits (fp16 attention, variant 13, cancels to
# ~1e-4 like every other backend's fp16 attention).
IREE_VARIANTS=" 0 1 2 3 4 5 6 7 8 10 11 12 13 "
BASELINES=${BASELINES:-"triton tensorrt compile-ro compile-mat jax iree"}
has_baseline() { case " $BASELINES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
baselines() {
  local variant=$1 k=$2 n=$3
  has_baseline triton && baseline_run "$TORCH_PYTHON" triton_bench.py triton "$variant" "$k" "$n" "$ITERS" "$WARMUP"
  has_baseline tensorrt && baseline_run "$TRT_PYTHON" torch_bench.py tensorrt "$variant" "$k" "$n" "$ITERS" "$WARMUP"
  # Reachable both from the normal torch section below (run_point calls
  # baselines() at the end of every point) and from --baselines-only, which
  # calls only baselines().
  has_baseline compile-ro && torchrun compile-ro "$variant" "$k" "$n" "$ITERS" "$WARMUP"
  has_baseline compile-mat && torchrun compile-mat "$variant" "$k" "$n" "$ITERS" "$WARMUP"
  has_baseline jax && baseline_run "$JAX_PYTHON" jax_bench.py jax "$variant" "$k" "$n" "$ITERS" "$WARMUP"
  if has_baseline iree; then
    case "$IREE_VARIANTS" in
      *" $variant "*) baseline_run "$JAX_PYTHON" jax_bench.py iree "$variant" "$k" "$n" "$ITERS" "$WARMUP" ;;
    esac
  fi
}

app_point() {
  local variant=$1 k=$2 n=$3 line
  if ! line=$("$RUN_PYPY" $JIT_FLAGS "$HERE/../applevel/micro.py" \
                app "$variant" "$k" "$n" "$ITERS" "$WARMUP") || [ -z "$line" ]; then
    echo "run_micro.sh: app-level variant $variant k $k n $n failed" >&2
    return 0
  fi
  line=$(echo "$line" | tail -1)
  echo "$line" | tr ' ' '\t' >> "$TSV"
  record_micro_line "$line"
}

run_point() {
  local variant=$1 k=$2 n=$3 line eff=""
  if [ "$BASELINES_ONLY" = 1 ]; then
    baselines "$variant" "$k" "$n"
    return 0
  fi
  if [ "$APPLEVEL" = 1 ]; then
    app_point "$variant" "$k" "$n"
    return 0
  fi
  for mode in fused eager nojit; do
    if ! line=$(ours $mode $variant $k $n $ITERS $WARMUP); then
      echo "run_micro.sh: standalone variant $variant k $k n $n mode $mode failed" >&2
      return 0
    fi
    echo "$line" | tr ' ' '\t' >> "$TSV"
    record_micro_line "$line"
    if [ "$mode" = fused ]; then eff=$(echo "$line" | awk '{print $4}'); fi
  done
  [ -n "$eff" ] || eff=$n
  if [ "$eff" != "$n" ]; then
    echo "run_micro: variant $variant n $n -> $eff (fitted to GPU memory)" >&2
  fi
  torchrun compile $variant $k $eff $ITERS $WARMUP
  torchrun eager $variant $k $eff $ITERS $WARMUP
  baselines $variant $k $eff
}

if [ -n "$1" ]; then
  v=$1 k=$2 n=$3
  for rep in $(seq "$ROUNDS"); do run_point "$v" "$k" "$n"; done
  echo "wrote $TSV"
  exit 0
fi

# The grid lives in benchmark/benchmarks.toml; grid.py is the only thing that
# parses it, so a new point is one entry there rather than a loop here and a
# label in plot.py.
GRID=$("${RTENSOR_PYTHON:-python3}" "$HERE/grid.py" points) || exit 1
SWEEP=$("${RTENSOR_PYTHON:-python3}" "$HERE/grid.py" precision) || exit 1

grid_count=$(echo "$GRID" | awk 'NF{c++} END{print c+0}')
if [ "$APPLEVEL" = 1 ]; then
  sweep_count=0
else
  sweep_count=$(echo "$SWEEP" | awk 'NF{c++} END{print c+0}')
fi
progress_init micro $((ROUNDS * (grid_count + sweep_count)))

for rep in $(seq "$ROUNDS"); do
  while read -r v k n; do
    [ -n "$v" ] || continue
    progress_step "variant $v k $k n $n"
    run_point "$v" "$k" "$n"
  done <<< "$GRID"
done

# precision sweep is app-level N/A: micro.py rejects non-float64, so skip it entirely there.
if [ "$APPLEVEL" != 1 ]; then
  for rep in $(seq "$ROUNDS"); do
    while read -r v k n dt; do
      [ -n "$v" ] || continue
      progress_step "variant $v k $k n $n dtype $dt"
      RTENSOR_DTYPE=$dt TORCH_DTYPE=$dt run_point "$v" "$k" "$n"
    done <<< "$SWEEP"
  done
fi

progress_done
echo "wrote $TSV"
