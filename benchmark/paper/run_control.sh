#!/bin/bash
# Host-dependent control flow inside a real model: data-dependent early exit
# (DeeBERT / CALM style) on distilgpt2.  After each transformer block a
# confidence is computed on the device and read back to the host with
# .item(); a Python branch on that value decides whether the remaining blocks
# run at all.  See applevel/earlyexit.py for the technique and the timing
# boundaries, applevel/inputs.py (CONTROL) for the schedule.
#
# Two regimes:
#   stable   one input, one threshold - every iteration exits at layer 4
#   varying  a rotating request mix - exits at layers 2, 3, 5, 6, so guards
#            fail, bridges are compiled, and Dynamo's frame count is
#            compared against the stable regime's
#
# total_ms spans the whole timed phase including the iterations that compile,
# so no system gets to warm its compile cost away; p50/p95/max are over the
# iterations after WARMUP, so a compilation in the steady state lands in the
# tail rather than in the cold start.
#
# Every system must produce the same exit-layer sequence for a regime.  If one
# does not it is doing a different amount of work and the comparison is void,
# so the mismatch is reported and the script exits non-zero.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/control.tsv"
SERIES="$OUT/control_series.tsv"
tsv_init "$TSV" "system\tregime\tround\ttotal_ms\tp50_us\tp95_us\tmax_us\tloops\tbridges\tkernels\tgraphs\tbreaks\tframe_compiles\texit_hash\texit_seq\tmaxabsdiff\ttolerance\tpass\tbinary"
tsv_init "$SERIES" "system\tregime\tround\titer\tus\texit_layer\tbinary"

# Per-row provenance, same convention as run_models.sh.
PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
TORCH_VER=$(pkg_version "$TORCH_PYTHON" torch torch)
JAX_VER=$(pkg_version "$JAX_PYTHON" jax jax)
binary_of() {
  case "$1" in
    ours) echo "${PYPY_SHA:-unknown}" ;;
    jax-*) echo "$JAX_VER" ;;
    *) echo "$TORCH_VER" ;;
  esac
}

WEIGHTS_DIR="$WEIGHTS/distilgpt2"
MODEL_TOL=$(tolerance_for distilgpt2)
export MODEL_TOL

REGIMES=${REGIMES:-"stable varying"}
STATUS=0

# $1 system, $2 regime, $3 round, $4 the driver's whole stdout.
# The per-iteration series goes to control_series.tsv only; one bench_record
# per iteration would be one python process per iteration.
emit() {
  local system=$1 regime=$2 round=$3 out=$4
  local binary row hash expect
  binary=$(binary_of "$system")
  row=$(echo "$out" | grep '^control ' | tail -1)
  if [ -z "$row" ]; then
    echo "run_control.sh: $system/$regime produced no row" >&2
    STATUS=1
    return
  fi
  echo "$out" | grep '^series' | while IFS=$'\t' read -r _ i us layer; do
    [ -n "$layer" ] || continue
    echo -e "$system\t$regime\t$round\t$i\t$us\t$layer\t$binary" >> "$SERIES"
  done

  local total=$(field_of "$row" total_ms) p50=$(field_of "$row" p50_us)
  local p95=$(field_of "$row" p95_us) mx=$(field_of "$row" max_us)
  local loops=$(field_of "$row" loops) bridges=$(field_of "$row" bridges)
  local kernels=$(field_of "$row" kernels) graphs=$(field_of "$row" graphs)
  local breaks=$(field_of "$row" breaks) recomp=$(field_of "$row" frame_compiles)
  local diff=$(field_of "$row" maxabsdiff) tol=$(field_of "$row" tol)
  local passed=$(field_of "$row" pass)
  hash=$(field_of "$row" exit_hash)
  local seq=$(echo "$row" | grep -o 'exit_seq=[0-9]*' | cut -d= -f2)

  # The same work on every system, or the comparison means nothing.
  expect=$(cat "$OUT/.control_hash_$regime" 2>/dev/null || true)
  if [ "$system" = ours ] || [ -z "$expect" ]; then
    echo "$hash" > "$OUT/.control_hash_$regime"
  elif [ "$hash" != "$expect" ]; then
    echo "run_control.sh: $system/$regime exit sequence $hash != ours $expect;" \
         "the systems ran different amounts of work, the row is not comparable" >&2
    STATUS=1
  fi
  if [ "$passed" = 0 ]; then
    echo "run_control.sh: $system/$regime failed the tolerance check" \
         "(maxabsdiff=$diff tol=$tol)" >&2
    STATUS=1
  fi

  echo -e "$system\t$regime\t$round\t$total\t$p50\t$p95\t$mx\t$loops\t$bridges\t$kernels\t$graphs\t$breaks\t$recomp\t$hash\t$seq\t$diff\t$tol\t$passed\t$binary" >> "$TSV"
  bench_record control system="$system" regime="$regime" round="$round" \
    total_ms="$total" p50_us="$p50" p95_us="$p95" max_us="$mx" \
    loops="$loops" bridges="$bridges" kernels="$kernels" graphs="$graphs" \
    breaks="$breaks" frame_compiles="$recomp" exit_hash="$hash" exit_seq="$seq" \
    maxabsdiff="$diff" tolerance="$tol" pass="$passed" iters="$ITERS" \
    warmup="$WARMUP" binary="$binary"
}

# One driver run.  A system that cannot run here is one absent row with a
# reason on stderr, never a silent omission.
run_one() {
  local system=$1 regime=$2 round=$3; shift 3
  local out err rc=0
  err=$(mktemp)
  out=$("$@" 2>"$err") || rc=$?
  if [ "$rc" = 0 ]; then
    emit "$system" "$regime" "$round" "$out"
  else
    echo "run_control.sh: $system/$regime failed:" \
         "$(grep -m1 -E 'Error|error:|Exception|AssertionError' "$err" || tail -1 "$err")" >&2
    STATUS=1
  fi
  rm -f "$err"
}

TORCH_MODES="eager compile"
has_baseline compile-ro && TORCH_MODES="$TORCH_MODES compile-ro"
has_baseline compile-mat && TORCH_MODES="$TORCH_MODES compile-mat"
JAX_MODES=""
if [ -n "${JAX_PYTHON:-}" ] && [ -x "$JAX_PYTHON" ] && has_baseline jax; then
  JAX_MODES="perlayer while"
fi

n_systems=$(( 1 + $(echo $TORCH_MODES | wc -w) + $(echo $JAX_MODES | wc -w) ))
progress_init control $(( ROUNDS * $(echo $REGIMES | wc -w) * n_systems ))

for round in $(seq "$ROUNDS"); do
  for regime in $REGIMES; do
    rm -f "$OUT/.control_hash_$regime"
    # ours first: it writes logits_earlyexit.bin, the reference the others
    # are checked against.
    progress_step "ours $regime round $round/$ROUNDS"
    run_one ours "$regime" "$round" \
      "$RUN_PYPY" $JIT_FLAGS "$APP/earlyexit.py" "$WEIGHTS_DIR" "$regime" \
      "$ITERS" "$WARMUP"
    for mode in $TORCH_MODES; do
      progress_step "torch-$mode $regime round $round/$ROUNDS"
      [ -n "$TORCH_PYTHON" ] || continue
      run_one "torch-$mode" "$regime" "$round" \
        "$TORCH_PYTHON" "$APP/earlyexit_torch.py" "$mode" "$WEIGHTS_DIR" \
        "$regime" "$ITERS" "$WARMUP"
    done
    for mode in $JAX_MODES; do
      progress_step "jax-$mode $regime round $round/$ROUNDS"
      run_one "jax-$mode" "$regime" "$round" \
        "$JAX_PYTHON" "$APP/earlyexit_jax.py" "$mode" "$WEIGHTS_DIR" \
        "$regime" "$ITERS" "$WARMUP"
    done
  done
done

progress_done
rm -f "$OUT"/.control_hash_*
echo "wrote $TSV and $SERIES"
exit $STATUS
