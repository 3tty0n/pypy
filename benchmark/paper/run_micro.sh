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
torchrun() { "$TORCH_PYTHON" "$HERE/../torch_bench.py" "$@" 2>/dev/null | tail -1 | tr ' ' '\t' >> "$TSV"; }

run_point() {
  local variant=$1 k=$2 n=$3 line eff=""
  for mode in fused eager nojit; do
    line=$(ours $mode $variant $k $n $ITERS)
    echo "$line" | tr ' ' '\t' >> "$TSV"
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

for rep in $(seq "$ROUNDS"); do
  for k in 1 4 8; do run_point 0 $k 1000000; done
  for n in 10000 100000 1000000 10000000; do run_point 0 4 $n; done
  for v in 1 2 3 4 5; do run_point $v 4 1000000; done
  for v in 11 12 13; do for n in 25600 256000; do run_point $v 1 $n; done; done
  for v in 6 7 8 9 10; do for n in 25600 256000; do run_point $v 1 $n; done; done
done

for rep in $(seq "$ROUNDS"); do
  for v in 8 13; do
    for dt in float64 float32 float16; do
      RTENSOR_DTYPE=$dt TORCH_DTYPE=$dt run_point $v 1 256000
    done
  done
done

echo "wrote $TSV"
