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

PROBE_PYPY=${MOTION_PYPY:-$PYPY}
[ -x "$PROBE_PYPY" ] || { echo "run_motion.sh: no interpreter at $PROBE_PYPY" >&2; exit 1; }

echo "== firewall =="
python3 "$HERE/audit_rules.py" "$RULES/layout_rule.py" || {
  echo "run_motion.sh: the rule failed the audit; not measuring" >&2; exit 1; }
if python3 "$HERE/audit_rules.py" "$RULES/layout_rule_leaky.py" >/dev/null 2>&1; then
  echo "run_motion.sh: the audit accepted the contaminated control, so it" \
       "proves nothing; not measuring" >&2
  exit 1
fi
echo "   the contaminated control is rejected, as it must be"

RUNS=${MOTION_RUNS:-10}
STEPS=${MOTION_STEPS:-200}
SWITCH=${MOTION_SWITCH:-$((STEPS / 2))}

TSV="$OUT/motion.tsv"
tsv_init "$TSV" "arm\trun\tsteps\tswitch\tstep_us\tat_us\tretained_bytes\tlaunches_before\tlaunches_after\tkernels_before\tkernels_after\tpass\tbinary"
BINARY=$(sha256sum "$PROBE_PYPY" | cut -c1-12)
ORIG_TMPDIR=${TMPDIR:-/tmp}
STATUS=0

progress_init motion $((RUNS * 2))
for run in $(seq "$RUNS"); do
  for arm in derived handwritten; do
    progress_step "$arm run $run/$RUNS"
    cache=$(mktemp -d -p "$ORIG_TMPDIR")
    err=$(mktemp)
    row=$(TMPDIR="$cache" TRITON_CACHE_DIR="$cache/triton" \
          "$PROBE_PYPY" $JIT_FLAGS "$APP/motion_probe.py" "$arm" "$STEPS" \
          "$SWITCH" 2>"$err" | grep '^motion ' | tail -1) || true
    if [ -z "$row" ]; then
      echo "run_motion.sh: $arm run $run produced no row:" \
           "$(grep -m1 -E 'Error|error|Exception|says' "$err" || tail -1 "$err")" >&2
      STATUS=1
    else
      passed=$(field_of "$row" pass)
      [ "$passed" = 1 ] || { echo "run_motion.sh: $arm run $run disagreed with the rule" >&2; STATUS=1; }
      echo -e "$arm\t$run\t$STEPS\t$SWITCH\t$(field_of "$row" step_us)\t$(field_of "$row" at_us)\t$(field_of "$row" retained_bytes)\t$(field_of "$row" launches_before)\t$(field_of "$row" launches_after)\t$(field_of "$row" kernels_before)\t$(field_of "$row" kernels_after)\t$passed\t$BINARY" >> "$TSV"
      bench_record motion arm="$arm" run="$run" steps="$STEPS" switch="$SWITCH" \
        step_us="$(field_of "$row" step_us)" at_us="$(field_of "$row" at_us)" \
        retained_bytes="$(field_of "$row" retained_bytes)" \
        launches_before="$(field_of "$row" launches_before)" \
        launches_after="$(field_of "$row" launches_after)" \
        kernels_before="$(field_of "$row" kernels_before)" \
        kernels_after="$(field_of "$row" kernels_after)" pass="$passed" \
        binary="$BINARY"
    fi
    rm -rf "$err" "$cache"
  done
done
progress_done
echo "wrote $TSV"
exit $STATUS
