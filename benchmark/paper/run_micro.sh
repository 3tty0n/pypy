#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"

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

run_point() {
  local variant=$1 k=$2 n=$3 line eff=""
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
