#!/bin/bash
# A declared change rule against the hand-written guard-recovery path.
#
# The rule is benchmark/motion/layout_rule.py, and benchmark/paper/
# audit_rules.py is run first: if the rule mentions anything belonging to the
# target representation the comparison is meaningless, so the stage refuses
# to measure before the firewall has been checked.  The audit is also run
# against a deliberately contaminated copy of the same rule, which it must
# reject - an audit that accepts everything establishes nothing.
#
#   derived      the rule applied to the value where it is, so the region
#                the meta-tracer is specialising produces the changed form
#   handwritten  the value brought to canonical form through the existing
#                recovery path first, and the rule applied to that
#
# RUNS fresh processes per arm, each with an empty kernel cache, because the
# question is what a transition costs and a warm cache answers it once.
#
#   MOTION_PYPY=/path/to/pypy-c benchmark/paper/bench.sh motion
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
RULES="$HERE/../motion"

# Which change class to demonstrate.  axis: the same vector addressed along
# the other axis, carried in a fusion node's broadcast parameter.  layout: the
# same values addressed as blocks of heads - a different shape and element
# order - which needs a binary whose gathers are fusion nodes; on an older one
# the derived and handwritten arms are the same program.
CASE=${MOTION_CASE:-axis}
case "$CASE" in
  axis)   RULE="$RULES/axis_rule.py"; PROBE="axis_probe.py"; TAG=axis ;;
  layout) RULE="$RULES/layout_rule.py"; PROBE="motion_probe.py"; TAG=layout ;;
  *) echo "run_motion.sh: unknown case $CASE (axis, layout)" >&2; exit 2 ;;
esac

PROBE_PYPY=${MOTION_PYPY:-$PYPY}
[ -x "$PROBE_PYPY" ] || { echo "run_motion.sh: no interpreter at $PROBE_PYPY" >&2; exit 1; }

echo "== firewall =="
python3 "$HERE/audit_rules.py" "$RULE" || {
  echo "run_motion.sh: the rule failed the audit; not measuring" >&2; exit 1; }
if python3 "$HERE/audit_rules.py" "$RULES/layout_rule_leaky.py" >/dev/null 2>&1; then
  echo "run_motion.sh: the audit accepted the contaminated control, so it" \
       "proves nothing; not measuring" >&2
  exit 1
fi
echo "   the contaminated control is rejected, as it must be"

RUNS=${MOTION_RUNS:-10}
# A list: the retention comparison changes direction with loop length, so a
# single length is a choice, not a result.  The transition is always halfway.
STEPS_LIST=${MOTION_STEPS:-200}
ARMS=${MOTION_ARMS:-"derived handwritten"}
# cold: an empty kernel cache per run, so the transition step pays for every
# compile it causes.  warm: one primed cache per arm, for sweeps where only
# the steady state and retention are read.
CACHE=${MOTION_CACHE:-cold}

TSV="$OUT/motion.tsv"
tsv_init "$TSV" "case\tarm\trun\tsteps\tswitch\tstep_us\tat_us\tretained_bytes\tlaunches_before\tlaunches_after\tkernels_before\tkernels_after\tpass\tcache\tbinary"
BINARY=$(binary_sha "$PROBE_PYPY")
ORIG_TMPDIR=${TMPDIR:-/tmp}
WARM_DIR="$OUT/.motion_cache"
STATUS=0

drain_for() { [ "$1" = drain ] && echo 1 || echo ""; }

progress_init motion $((RUNS * $(echo $ARMS | wc -w) * $(echo $STEPS_LIST | wc -w)))
for STEPS in $STEPS_LIST; do
 SWITCH=$((STEPS / 2))
 for run in $(seq "$RUNS"); do
  for arm in $ARMS; do
    progress_step "$arm steps=$STEPS run $run/$RUNS"
    if [ "$CACHE" = warm ]; then
      cache="$WARM_DIR/$TAG-$arm"
      if [ ! -d "$cache" ]; then
        mkdir -p "$cache/triton"
        TMPDIR="$cache" TRITON_CACHE_DIR="$cache/triton" METATENSOR_DRAIN=$(drain_for "$arm") \
          "$PROBE_PYPY" $JIT_FLAGS "$APP/$PROBE" "$arm" 40 20 >/dev/null 2>&1 || true
      fi
    else
      cache=$(mktemp -d -p "$ORIG_TMPDIR")
    fi
    err=$(mktemp)
    row=$(TMPDIR="$cache" TRITON_CACHE_DIR="$cache/triton" METATENSOR_DRAIN=$(drain_for "$arm") \
          "$PROBE_PYPY" $JIT_FLAGS "$APP/$PROBE" "$arm" "$STEPS" \
          "$SWITCH" 2>"$err" | grep -E '^(motion|axis|blocked) ' | tail -1) || true
    if [ -z "$row" ]; then
      echo "run_motion.sh: $arm run $run produced no row:" \
           "$(grep -m1 -E 'Error|error|Exception|says|needs' "$err" || tail -1 "$err")" >&2
      STATUS=1
    else
      passed=$(field_of "$row" pass)
      [ "$passed" = 1 ] || { echo "run_motion.sh: $arm run $run disagreed with the rule" >&2; STATUS=1; }
      echo -e "$TAG\t$arm\t$run\t$STEPS\t$SWITCH\t$(field_of "$row" step_us)\t$(field_of "$row" at_us)\t$(field_of "$row" retained_bytes)\t$(field_of "$row" launches_before)\t$(field_of "$row" launches_after)\t$(field_of "$row" kernels_before)\t$(field_of "$row" kernels_after)\t$passed\t$CACHE\t$BINARY" >> "$TSV"
      bench_record motion case="$TAG" arm="$arm" run="$run" steps="$STEPS" switch="$SWITCH" \
        step_us="$(field_of "$row" step_us)" at_us="$(field_of "$row" at_us)" \
        retained_bytes="$(field_of "$row" retained_bytes)" \
        launches_before="$(field_of "$row" launches_before)" \
        launches_after="$(field_of "$row" launches_after)" \
        kernels_before="$(field_of "$row" kernels_before)" \
        kernels_after="$(field_of "$row" kernels_after)" pass="$passed" \
        cache="$CACHE" binary="$BINARY"
    fi
    rm -f "$err"
    [ "$CACHE" = warm ] || rm -rf "$cache"
  done
 done
done
progress_done
echo "wrote $TSV"
exit $STATUS
