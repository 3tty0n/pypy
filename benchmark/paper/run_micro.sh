#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"

HEADER="mode\tvariant\tk\tn\titers\twarm_s\tsteady_us\tkernels\tacc\tcompiled_in_timed\tlaunches_per_iter\tgraphs\tbreaks"
TSV="$OUT/micro.tsv"
echo -e "$HEADER" > "$TSV"

ours() { "$BENCH" "$@" | tr ' ' '\t' >> "$TSV"; }
torchrun() { "$TORCH_PYTHON" "$HERE/../torch_bench.py" "$@" 2>/dev/null | tail -1 | tr ' ' '\t' >> "$TSV"; }

run_point() {
  local variant=$1 k=$2 n=$3
  for mode in fused eager nojit; do ours $mode $variant $k $n $ITERS; done
  torchrun compile $variant $k $n $ITERS
  torchrun eager $variant $k $n $ITERS
}

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
