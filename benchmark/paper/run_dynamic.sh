#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

SERIES="$OUT/dynamic_series.tsv"
SUMMARY="$OUT/dynamic_summary.tsv"
tsv_init "$SERIES" "system\tround\tlength\tus"
tsv_init "$SUMMARY" "system\tround\tlength\tmedian_us\tloops\tbridges\trecompiles"

median() { python3 -c "import sys,statistics;print(statistics.median([float(x) for x in sys.stdin if x.strip()]))"; }

run_ours() {
  local round=$1
  local jitlog="$OUT/.dyn_jit_$round.log"
  out=$(PYPYLOG=jit-summary:"$jitlog" "$RUN_PYPY" $JIT_FLAGS "$HERE/dynamic_gpt2.py" "$WEIGHTS/distilgpt2" "$ITERS")
  loops=$(grep 'Total # of loops:' "$jitlog" | grep -o '[0-9]*$' | tail -1)
  bridges=$(grep 'Total # of bridges:' "$jitlog" | grep -o '[0-9]*$' | tail -1)
  echo "$out" | tail -n +2 | while IFS=$'\t' read -r length us; do
    echo -e "ours\t$round\t$length\t$us" >> "$SERIES"
  done
  for length in 32 48 64 96 128; do
    m=$(echo "$out" | tail -n +2 | awk -F'\t' -v l="$length" '$1==l{print $2}' | median)
    echo -e "ours\t$round\t$length\t$m\t${loops:-0}\t${bridges:-0}\t0" >> "$SUMMARY"
  done
  rm -f "$jitlog"
}

run_torch() {
  local mode=$1 system=$2 round=$3
  out=$("$TORCH_PYTHON" "$HERE/dynamic_gpt2_torch.py" "$mode" "$WEIGHTS/distilgpt2" "$ITERS" 2>/dev/null)
  recompiles=$(echo "$out" | grep '^recompiles' | cut -f2)
  echo "$out" | grep -v '^length\|^recompiles' | while IFS=$'\t' read -r length us; do
    echo -e "$system\t$round\t$length\t$us" >> "$SERIES"
  done
  for length in 32 48 64 96 128; do
    m=$(echo "$out" | grep -v '^length\|^recompiles' | awk -F'\t' -v l="$length" '$1==l{print $2}' | median)
    echo -e "$system\t$round\t$length\t$m\t0\t0\t${recompiles:-0}" >> "$SUMMARY"
  done
}

for round in $(seq "$ROUNDS"); do
  run_ours "$round"
  run_torch eager torch-eager "$round"
  run_torch compile torch-compile-static "$round"
  run_torch compile-dynamic torch-compile-dynamic "$round"
done

echo "wrote $SERIES and $SUMMARY"
