#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/rtensor-bench}
TORCH_PYTHON=${TORCH_PYTHON:-$RTENSOR_PYTHON}
export RTENSOR_BUDGET_MB=${RTENSOR_BUDGET_MB:-64}
OUT=${OUT:-$HERE/results/$(date +%F)-$(hostname)}
REPS=${REPS:-3}
ITERS=${ITERS:-200}
mkdir -p "$OUT"
HEADER="mode\tvariant\tk\tn\titers\twarm_s\tsteady_us\tkernels\tacc\tcompiled_in_timed\tlaunches_per_iter\tgraphs\tbreaks"
ours() { "$BIN" "$@" | tr ' ' '\t'; }
torch() { "$TORCH_PYTHON" "$HERE/torch_bench.py" "$@" 2>/dev/null | tail -1 | tr ' ' '\t'; }

echo -e "$HEADER" > "$OUT/rtensor_chain.tsv"
echo -e "$HEADER" > "$OUT/torch.tsv"
for k in 1 2 4 8; do for rep in $(seq $REPS); do
  for mode in fused eager nojit; do ours $mode 0 $k 1000000 $ITERS >> "$OUT/rtensor_chain.tsv"; done
  for mode in eager compile; do torch $mode 0 $k 1000000 $ITERS >> "$OUT/torch.tsv"; done
done; done
for n in 1000 10000 100000 1000000 10000000; do for rep in $(seq $REPS); do
  for mode in fused eager nojit; do ours $mode 0 4 $n $ITERS >> "$OUT/rtensor_chain.tsv"; done
  for mode in eager compile; do torch $mode 0 4 $n $ITERS >> "$OUT/torch.tsv"; done
done; done

echo -e "$HEADER" > "$OUT/rtensor_branch.tsv"
for rep in $(seq $REPS); do for v in 1 2 3 4 5; do for n in 10000 1000000; do
  for mode in fused eager; do ours $mode $v 4 $n $ITERS >> "$OUT/rtensor_branch.tsv"; done
  torch compile $v 4 $n $ITERS >> "$OUT/torch.tsv"
  torch eager $v 4 $n $ITERS >> "$OUT/torch.tsv"
done; done; done
echo -e "$HEADER" > "$OUT/rtensor_mlp.tsv"
for n in 25600 256000; do for rep in $(seq $REPS); do
  for v in 6 7 8 9 10; do
    for mode in fused eager nojit; do ours $mode $v 1 $n $ITERS >> "$OUT/rtensor_mlp.tsv"; done
    torch compile $v 1 $n $ITERS >> "$OUT/torch.tsv"
    torch eager $v 1 $n $ITERS >> "$OUT/torch.tsv"
  done
done; done
python3 "$HERE/summarize.py" "$OUT"/*.tsv | tee "$OUT/summary.txt"
