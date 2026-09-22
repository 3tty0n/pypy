#!/bin/bash
# The generated leave transition, checked end to end.
#
#   1. audit    the change rule speaks only motion_iface; its contaminated
#               copy is rejected (or nothing below is measured)
#   2. runs     every history (h1 h2 h3 h2post none), four ways:
#                 ref       interpreter only (--jit off): the unspecialised
#                           reference execution
#                 motion    the generated transition, JIT on
#                 adapter   the hand-written adapter, JIT on
#                 erase / late   the two mutants of the generated side
#   3. check    benchmark/motion/check_leave.py, which shares no code with
#               any of them, on every output; motion and adapter must also
#               agree with ref field by field; each mutant must be caught
#   4. count    benchmark/motion/count_lines.py
#   5. perf     (LEAVE_PERF=1) steady-state step time with and without
#               retention and for the adapter, and the time of the step the
#               leave lands in, LEAVE_RUNS fresh processes each
#
# Everything goes to $OUT/leave/, and $OUT/leave.tsv has one row per run.
#   LEAVE_PYPY=/path/to/pypy-c bash benchmark/paper/run_leave.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
MOT="$HERE/../motion"
REPO_ROOT=$(cd "$HERE/../.." && pwd)

P=${LEAVE_PYPY:-$PYPY}
[ -x "$P" ] || { echo "run_leave.sh: no interpreter at $P" >&2; exit 1; }
D="$OUT/leave"
mkdir -p "$D"
BINARY=$(binary_sha "$P")
STEPS=${LEAVE_STEPS:-200}
N=${LEAVE_N:-64}
HISTORIES=${LEAVE_HISTORIES:-"h1 h2 h3 h2post none"}
STATUS=0

echo "== audit" | tee "$D/audit.log"
python3 "$HERE/audit_rules.py" "$REPO_ROOT/pypy/module/_metatensor/rule_leave.py" \
  | tee -a "$D/audit.log" || { echo "run_leave.sh: the rule failed the audit" >&2; exit 1; }
if python3 "$HERE/audit_rules.py" "$MOT/rule_leave_leaky.py" >> "$D/audit.log" 2>&1; then
  echo "run_leave.sh: the audit accepted the contaminated rule; it proves nothing" >&2
  exit 1
fi
echo "   contaminated control rejected" | tee -a "$D/audit.log"
git -C "$REPO_ROOT" log --format='%h %ad %s' --date=iso -- \
  pypy/module/_metatensor/rule_leave.py >> "$D/audit.log"

TSV="$OUT/leave.tsv"
tsv_init "$TSV" "history\timpl\tjit\tsteps\tat\tcommits\tretains\tsteady_launches\tlaunches_per_step\tloops\tbridges\tlog\tcheck\tvs_ref\tbinary"

probe() {  # probe NAME HISTORY [env...] -- [probe args...]
  local name=$1 hist=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local jitflags=$JIT_FLAGS
  [ "$name" = ref ] && jitflags="--jit off"
  env "${envs[@]}" "$P" $jitflags "$APP/leave_probe.py" "$hist" \
    --steps "$STEPS" --n "$N" --save "$D/${name}_$hist" "$@" \
    > "$D/${name}_$hist.out" 2>&1 || {
      echo "run_leave.sh: $name $hist failed: $(tail -1 "$D/${name}_$hist.out")" >&2
      STATUS=1; return 1; }
}

echo "== runs"
for h in $HISTORIES; do
  probe ref "$h" -- --nostats
  probe motion "$h" --
  probe adapter "$h" -- --impl adapter
  probe erase "$h" MOTION_ERASE=1 --
  probe late "$h" MOTION_ORDER=late --
done

echo "== check" | tee "$D/checker.log"
row() {  # row NAME HISTORY CHECK VSREF
  local f="$D/${1}_$2" line
  line=$(cat "$f")
  echo -e "$2\t$1\t$([ "$1" = ref ] && echo off || echo on)\t$(field_of "$line" steps)\t$(field_of "$line" at)\t$(field_of "$line" commits)\t$(field_of "$line" retains)\t$(field_of "$line" steady_launches)\t$(field_of "$line" launches_per_step)\t$(field_of "$line" loops)\t$(field_of "$line" bridges)\t$(echo "$line" | grep -o 'log=[^ ]*' | cut -d= -f2)\t$3\t$4\t$BINARY" >> "$TSV"
}
for h in $HISTORIES; do
  for name in ref motion adapter; do
    if python3 "$MOT/check_leave.py" "$D/${name}_$h" >> "$D/checker.log"; then c=ok; else c=MISMATCH; STATUS=1; fi
    v=-
    if [ "$name" != ref ]; then
      if python3 "$MOT/check_leave.py" --pair "$D/ref_$h" "$D/${name}_$h" >> "$D/checker.log"; then v=same; else v=DIFFER; STATUS=1; fi
    fi
    row "$name" "$h" "$c" "$v"
  done
  for name in erase late; do
    if python3 "$MOT/check_leave.py" "$D/${name}_$h" >> "$D/checker.log"; then c=ok; else c=MISMATCH; fi
    row "$name" "$h" "$c" -
  done
done
# A mutant is caught when the checker rejects it on some history; both must
# be caught on h2, the history they are built to break.
for name in erase late; do
  if grep -q "^MISMATCH h2 motion.*${name}_h2" "$D/checker.log"; then
    echo "   mutant $name: caught on h2" | tee -a "$D/checker.log"
  else
    echo "   mutant $name: NOT caught on h2" | tee -a "$D/checker.log"; STATUS=1
  fi
done
cat "$D/checker.log"

echo "== count"
python3 "$MOT/count_lines.py" | tee "$D/counts.txt"
python3 "$MOT/count_lines.py" --json > "$D/counts.json"

if [ "${LEAVE_PERF:-0}" = 1 ]; then
  echo "== perf"
  PTSV="$OUT/leave_perf.tsv"
  tsv_init "$PTSV" "config\trun\tn\tsteps\tstep_us\tbefore_us\tevent_us\tsteady_launches\tlaunches_per_step\tbinary"
  PN=${LEAVE_PERF_N:-65536}
  RUNS=${LEAVE_RUNS:-10}
  progress_init leave $((RUNS * 5))
  for run in $(seq "$RUNS"); do
    for cfg in "retained none" "unretained none --policy retain-only" \
               "adapter none --impl adapter" "retained-h2 h2" \
               "adapter-h2 h2 --impl adapter"; do
      set -- $cfg
      name=$1 hist=$2; shift 2
      progress_step "$name run $run/$RUNS"
      line=$("$P" $JIT_FLAGS "$APP/leave_probe.py" "$hist" --steps "$STEPS" \
             --n "$PN" --times "$@" 2>/dev/null | grep '^leave ' | tail -1) || true
      [ -n "$line" ] || { echo "run_leave.sh: perf $name run $run produced no row" >&2; STATUS=1; continue; }
      echo -e "$name\t$run\t$PN\t$STEPS\t$(field_of "$line" step_us)\t$(field_of "$line" before_us)\t$(field_of "$line" event_us)\t$(field_of "$line" steady_launches)\t$(field_of "$line" launches_per_step)\t$BINARY" >> "$PTSV"
    done
  done
  progress_done
fi
echo "wrote $TSV and $D/"
exit $STATUS
