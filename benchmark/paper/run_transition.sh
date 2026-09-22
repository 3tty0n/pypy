#!/bin/bash
# What a fused value costs to carry across a guard, measured both ways.
#
# A fused region that has already run holds a descriptor for its result: the
# device buffer, the input set the kernel was compiled for, and the layout it
# writes.  When a consumer for a value inside that region appears only after a
# guard has failed, that descriptor is either kept or discarded:
#
#   direct  keep it.  The launched kernel gains one output and is recompiled
#           for the new signature; inputs, buffers and layout are unchanged.
#   drain   discard it.  Walk the operation graph back to its leaves and
#           record a fresh kernel - canonical form, re-recorded, which is what
#           a runtime that carries no descriptor across the boundary must do.
#
# One binary, one emitter, one kernel cache, one allocator, one program, one
# input; METATENSOR_DRAIN is the only difference between the arms.  See
# applevel/transition_probe.py for the program and where the guard fails.
#
# The binary needs the drain knob, which is newer than the one the model
# sweep was measured with, so this stage takes its own:
#   TRANSITION_PYPY=/path/to/pypy-c benchmark/paper/bench.sh transition
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

PROBE_PYPY=${TRANSITION_PYPY:-$PYPY}
if [ ! -x "$PROBE_PYPY" ]; then
  echo "run_transition.sh: no interpreter at $PROBE_PYPY" >&2
  exit 1
fi

ITERS=${TRANSITION_ITERS:-400}
SWITCH=${TRANSITION_SWITCH:-$((ITERS / 2))}
SHAPES=${TRANSITION_SHAPES:-"256x64 1024x64 256x256"}

TSV="$OUT/transition.tsv"
SERIES="$OUT/transition_series.tsv"
tsv_init "$TSV" "arm\tcache\trows\tcols\tround\tbefore_us\tat_us\tsettling_us\tafter_us\ttotal_ms\tloops\tbridges\tlaunches\tcompiles\tkernels\tpass\tbinary"
tsv_init "$SERIES" "arm\tcache\trows\tcols\tround\titer\tus\tbinary"

BINARY=$(binary_sha "$PROBE_PYPY")
STATUS=0

# Both cache states, because they answer different questions and the answers
# differ by a factor of twenty.  Cold: every kernel this program needs is
# compiled, out of process, inside the measured window - which is what a first
# run costs and what the extra kernel costs.  Warm: the cubins are on disk, so
# what is left is the work itself.
ORIG_TMPDIR=${TMPDIR:-/tmp}
CACHES=${TRANSITION_CACHES:-"cold warm"}
WARM_DIR="$OUT/.transition_cache"

one() {
  local arm=$1 cachestate=$2 rows=$3 cols=$4 round=$5 drain=$6
  local out err rc=0 cache
  if [ "$cachestate" = warm ]; then
    # One directory per arm and shape, reused across rounds, and primed by an
    # untimed run so that round 1 is as warm as round 3.
    cache="$WARM_DIR/$arm-${rows}x${cols}"
    if [ ! -d "$cache" ]; then
      mkdir -p "$cache/triton"
      TMPDIR="$cache" TRITON_CACHE_DIR="$cache/triton" METATENSOR_DRAIN=$drain \
        "$PROBE_PYPY" $JIT_FLAGS "$APP/transition_probe.py" 40 20 \
        --rows "$rows" --cols "$cols" >/dev/null 2>&1 || true
    fi
  else
    cache=$(mktemp -d -p "$ORIG_TMPDIR")
  fi
  err=$(mktemp)
  out=$(TMPDIR="$cache" TRITON_CACHE_DIR="$cache/triton" \
        METATENSOR_DRAIN=$drain "$PROBE_PYPY" $JIT_FLAGS \
        "$APP/transition_probe.py" "$ITERS" "$SWITCH" \
        --rows "$rows" --cols "$cols" --series 2>"$err") || rc=$?
  local row
  row=$(echo "$out" | grep '^transition ' | tail -1)
  if [ "$rc" != 0 ] || [ -z "$row" ]; then
    echo "run_transition.sh: $arm ${rows}x${cols} round $round failed:" \
         "$(grep -m1 -E 'Error|error:|Exception|expected' "$err" || tail -1 "$err")" >&2
    STATUS=1
    rm -f "$err"
    [ "$cachestate" = cold ] && rm -rf "$cache"
    return
  fi
  rm -f "$err"
  [ "$cachestate" = cold ] && rm -rf "$cache"
  echo "$out" | grep '^series' | while IFS=$'\t' read -r _ i us; do
    [ -n "$us" ] || continue
    echo -e "$arm\t$cachestate\t$rows\t$cols\t$round\t$i\t$us\t$BINARY" >> "$SERIES"
  done
  local passed; passed=$(field_of "$row" pass)
  [ "$passed" = 1 ] || { echo "run_transition.sh: $arm ${rows}x${cols} disagreed with the reference" >&2; STATUS=1; }
  echo -e "$arm\t$cachestate\t$rows\t$cols\t$round\t$(field_of "$row" before_us)\t$(field_of "$row" at_us)\t$(field_of "$row" settling_us)\t$(field_of "$row" after_us)\t$(field_of "$row" total_ms)\t$(field_of "$row" loops)\t$(field_of "$row" bridges)\t$(field_of "$row" launches)\t$(field_of "$row" compiles)\t$(field_of "$row" kernels)\t$passed\t$BINARY" >> "$TSV"
  bench_record transition arm="$arm" cache="$cachestate" rows="$rows" cols="$cols" round="$round" \
    before_us="$(field_of "$row" before_us)" at_us="$(field_of "$row" at_us)" \
    settling_us="$(field_of "$row" settling_us)" \
    after_us="$(field_of "$row" after_us)" \
    total_ms="$(field_of "$row" total_ms)" loops="$(field_of "$row" loops)" \
    bridges="$(field_of "$row" bridges)" launches="$(field_of "$row" launches)" \
    compiles="$(field_of "$row" compiles)" kernels="$(field_of "$row" kernels)" \
    iters="$ITERS" switch="$SWITCH" pass="$passed" binary="$BINARY"
}

progress_init transition \
  $((ROUNDS * $(echo $SHAPES | wc -w) * $(echo $CACHES | wc -w) * 2))
for round in $(seq "$ROUNDS"); do
  for cachestate in $CACHES; do
    for shape in $SHAPES; do
      rows=${shape%x*}; cols=${shape#*x}
      progress_step "direct/$cachestate ${rows}x${cols} round $round/$ROUNDS"
      one direct "$cachestate" "$rows" "$cols" "$round" 0
      progress_step "drain/$cachestate ${rows}x${cols} round $round/$ROUNDS"
      one drain "$cachestate" "$rows" "$cols" "$round" 1
    done
  done
done
progress_done
echo "wrote $TSV and $SERIES"
exit $STATUS
