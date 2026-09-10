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
IREE_VARIANTS=" 0 6 11 12 13 "
baselines() {
  local variant=$1 k=$2 n=$3
  baseline_run "$TORCH_PYTHON" triton_bench.py triton "$variant" "$k" "$n" "$ITERS"
  baseline_run "$JAX_PYTHON" jax_bench.py jax "$variant" "$k" "$n" "$ITERS"
  case "$IREE_VARIANTS" in
    *" $variant "*)
      if [ "${RTENSOR_DTYPE:-float64}" = float64 ]; then
        baseline_run "$JAX_PYTHON" jax_bench.py iree "$variant" "$k" "$n" "$ITERS"
      fi ;;
  esac
}

app_point() {
  local variant=$1 k=$2 n=$3 line
  if ! line=$("$RUN_PYPY" $JIT_FLAGS "$HERE/../applevel/micro.py" \
                app "$variant" "$k" "$n" "$ITERS") || [ -z "$line" ]; then
    echo "run_micro.sh: app-level variant $variant k $k n $n failed" >&2
    return 0
  fi
  line=$(echo "$line" | tail -1)
  echo "$line" | tr ' ' '\t' >> "$TSV"
  record_micro_line "$line"
}

run_point() {
  local variant=$1 k=$2 n=$3 line eff=""
  if [ "$APPLEVEL" = 1 ]; then
    app_point "$variant" "$k" "$n"
    return 0
  fi
  for mode in fused eager nojit; do
    line=$(ours $mode $variant $k $n $ITERS)
    echo "$line" | tr ' ' '\t' >> "$TSV"
    record_micro_line "$line"
    if [ "$mode" = fused ]; then eff=$(echo "$line" | awk '{print $4}'); fi
  done
  [ -n "$eff" ] || eff=$n
  if [ "$eff" != "$n" ]; then
    echo "run_micro: variant $variant n $n -> $eff (fitted to GPU memory)" >&2
  fi
  torchrun compile $variant $k $eff $ITERS
  torchrun eager $variant $k $eff $ITERS
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
