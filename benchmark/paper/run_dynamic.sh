#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"

paper_setup_pypy
trap paper_cleanup_pypy EXIT

SERIES="$OUT/dynamic_series.tsv"
SUMMARY="$OUT/dynamic_summary.tsv"
tsv_init "$SERIES" "system\tround\tpass\tlength\tus\tbuild_us\tbinary"
tsv_init "$SUMMARY" "system\tround\tpass\tlength\tmedian_us\ttotal_us\tloops\tbridges\tkernels\tlaunches\tcache_hits\tgraphs\trecompiles\tcompile_ms\tbinary"

# Per-row provenance, same convention as run_models.sh: ours is identified by
# the pypy-c hash, torch by its version string.
PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
TORCH_VER=$([ -n "$TORCH_PYTHON" ] && [ -x "$TORCH_PYTHON" ] && \
  "$TORCH_PYTHON" -c "import torch; print('torch-' + torch.__version__)" 2>/dev/null || echo unknown)
binary_of() { case "$1" in ours) echo "${PYPY_SHA:-unknown}" ;; *) echo "$TORCH_VER" ;; esac; }

# Both drivers print the same per-iteration table; this folds it into one row
# per (pass, length) window: median of the times, sum of the counter deltas.
# cache_hits is launches minus newly compiled kernels, because the kernel cache
# has no hit counter of its own and adding one would mean retranslating pypy-c.
agg() {
  "${RTENSOR_PYTHON:-python3}" -c "
import sys, statistics, collections
g = collections.OrderedDict()
for line in sys.stdin:
    f = line.split('\t')
    if len(f) != 11 or f[0] == 'pass':
        continue
    g.setdefault((int(f[0]), int(f[1])), []).append([float(x) for x in f[2:]])
for (p, L), rows in g.items():
    step = [r[0] for r in rows]
    tot = [r[0] + r[1] for r in rows]
    s = [sum(r[i] for r in rows) for i in range(2, 9)]
    loops, bridges, kernels, launches, _graphs, recomp, cms = s
    graphs = max(r[6] for r in rows)
    # torch reports compile_ms as -1 (not separable from the iteration that
    # triggers it); summing that would turn it into a count of iterations.
    if min(r[8] for r in rows) < 0:
        cms = -1.0
    print('%d\t%d\t%.1f\t%.1f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f' % (
        p, L, statistics.median(step), statistics.median(tot),
        loops, bridges, kernels, launches, launches - kernels,
        graphs, recomp, cms))
"
}

# $1 system, $2 round, stdin the driver's per-iteration table.
emit() {
  local system=$1 round=$2 out=$3 binary=$(binary_of "$1")
  echo "$out" | tail -n +2 | while IFS=$'\t' read -r p length step build rest; do
    [ -n "$length" ] || continue
    echo -e "$system\t$round\t$p\t$length\t$step\t$build\t$binary" >> "$SERIES"
    bench_record dynamic_series system="$system" round="$round" pass="$p" \
      length="$length" us="$step" build_us="$build" binary="$binary"
  done
  echo "$out" | agg | while IFS=$'\t' read -r p length med tot loops bridges kernels launches hits graphs recomp cms; do
    echo -e "$system\t$round\t$p\t$length\t$med\t$tot\t$loops\t$bridges\t$kernels\t$launches\t$hits\t$graphs\t$recomp\t$cms\t$binary" >> "$SUMMARY"
    bench_record dynamic system="$system" round="$round" pass="$p" \
      length="$length" median_us="$med" total_us="$tot" loops="$loops" \
      bridges="$bridges" kernels="$kernels" launches="$launches" \
      cache_hits="$hits" graphs="$graphs" recompiles="$recomp" \
      compile_ms="$cms" binary="$binary"
  done
}

run_ours() {
  emit ours "$1" "$("$RUN_PYPY" $JIT_FLAGS "$APP/dynamic_gpt2.py" "$WEIGHTS/distilgpt2" "$ITERS")"
}

run_torch() {
  emit "$2" "$3" "$("$TORCH_PYTHON" "$APP/dynamic_gpt2_torch.py" "$1" "$WEIGHTS/distilgpt2" "$ITERS" 2>/dev/null)"
}

progress_init dynamic "$ROUNDS"

for round in $(seq "$ROUNDS"); do
  progress_step "round $round/$ROUNDS"
  run_ours "$round"
  run_torch eager torch-eager "$round"
  run_torch compile torch-compile-static "$round"
  run_torch compile-dynamic torch-compile-dynamic "$round"
done

progress_done
echo "wrote $SERIES and $SUMMARY"
