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
APP_VARIANTS="0 8 9 13"
if [ "$APPLEVEL" = 1 ]; then
  paper_setup_pypy
  trap paper_cleanup_pypy EXIT
fi

HEADER="mode\tvariant\tk\tn\titers\twarm_s\tsteady_us\tkernels\tacc\tcompiled_in_timed\tlaunches_per_iter\tgraphs\tbreaks"
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

app_point() {
  local variant=$1 k=$2 n=$3 line
  case " $APP_VARIANTS " in
    *" $variant "*) ;;
    *) return 0 ;;
  esac
  line=$("$RUN_PYPY" $JIT_FLAGS "$HERE/../applevel/micro.py" app "$variant" "$k" "$n" "$ITERS" | tail -1) || return 0
  [ -n "$line" ] || return 0
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

for rep in $(seq "$ROUNDS"); do
  while read -r v k n; do
    [ -n "$v" ] && run_point "$v" "$k" "$n"
  done <<< "$GRID"
done

for rep in $(seq "$ROUNDS"); do
  while read -r v k n dt; do
    [ -n "$v" ] || continue
    RTENSOR_DTYPE=$dt TORCH_DTYPE=$dt run_point "$v" "$k" "$n"
  done <<< "$SWEEP"
done

echo "wrote $TSV"
