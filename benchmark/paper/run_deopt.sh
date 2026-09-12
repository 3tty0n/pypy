#!/bin/bash
# EVAL_PLAN item C: what a guard failure costs when a trace's assumption
# breaks, against the steady state where it holds.
#
# Four value-guard schedules (deopt_probe.py: never, alternate, both-hot,
# fresh) on ours and on the torch equivalents of the same loop, plus the two
# recovery_probe cases whose guards really fail, (a) and (e).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/deopt.tsv"
tsv_init "$TSV" "system\tpattern\tround\tsteady_us\tfirst_fail_us\tafter_fail_us\tlaunches_per_iter\tloops\tbridges\tpeak_us\tcold_us"

PATTERNS=${PATTERNS:-"never alternate both-hot fresh"}

dp_row() {
  local system=$1 pattern=$2 round=$3 line=$4
  local steady first after launches loops bridges peak cold
  steady=$(field_of "$line" steady_us)
  first=$(field_of "$line" first_fail_us)
  after=$(field_of "$line" after_fail_us)
  launches=$(field_of "$line" launches_per_iter)
  loops=$(field_of "$line" loops)
  bridges=$(field_of "$line" bridges)
  peak=$(field_of "$line" peak_us)
  cold=$(field_of "$line" cold_us)
  echo -e "$system\t$pattern\t$round\t$steady\t$first\t$after\t$launches\t$loops\t$bridges\t$peak\t$cold" >> "$TSV"
  bench_record deopt system="$system" pattern="$pattern" round="$round" \
    steady_us="$steady" first_fail_us="$first" after_fail_us="$after" \
    launches_per_iter="$launches" loops="$loops" bridges="$bridges" \
    peak_us="$peak" cold_us="$cold" iters="$ITERS" warmup="$WARMUP"
}

progress_init deopt $(( ROUNDS * ($(echo $PATTERNS | wc -w) * 4 + 1) ))

for round in $(seq "$ROUNDS"); do
  for p in $PATTERNS; do
    progress_step "ours $p"
    out=$("$RUN_PYPY" $JIT_FLAGS "$APP/deopt_probe.py" "$p" "$ITERS" "$WARMUP" 2>/dev/null | grep '^deopt ' || true)
    [ -n "$out" ] && dp_row ours "$p" "$round" "$out"

    for mode in eager compile compile-ro; do
      progress_step "torch-$mode $p"
      [ -n "$TORCH_PYTHON" ] || continue
      out=$("$TORCH_PYTHON" "$APP/deopt_probe.py" "$p" "$ITERS" "$WARMUP" \
            --mode="$mode" 2>/dev/null | grep '^deopt ' || true)
      if [ -n "$out" ]; then
        dp_row "torch-$mode" "$p" "$round" "$out"
      else
        echo "run_deopt.sh: torch-$mode $p did not run" >&2
      fi
    done
  done

  # recovery_probe (a) and (e): the same question on the semantic probe, where
  # (e)'s guard failure also brings a new shape with it.
  progress_step "recovery_probe"
  out=$("$RUN_PYPY" $JIT_FLAGS "$APP/recovery_probe.py" --times 2>/dev/null | grep '^deopt-probe ' || true)
  echo "$out" | while read -r line; do
    [ -n "$line" ] || continue
    case=$(echo "$line" | grep -o 'case=[a-z]*' | cut -d= -f2)
    dp_row ours "probe-$case" "$round" "$line"
  done
done

progress_done
echo "wrote $TSV"
