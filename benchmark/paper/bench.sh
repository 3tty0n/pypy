#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)

ALL_MODELS="distilgpt2 gpt2 gpt2-medium smollm2-135m smollm2-360m smollm2-1.7b qwen2.5-0.5b bert-mini bert-base resnet18-b1 resnet18-b8 mixer_b16 deit-tiny vit-base"
ALL_EXPERIMENTS="fusion flat_block budget_mb precision tf32 max_inputs gather"

usage() {
  cat <<EOF
usage: bench.sh <command> [args] [-- <command> [args] ...]

commands:
  setup                  run setup.sh
  export                 export weight checkpoints
  micro [V K N]          microbenchmarks; no args = full grid, or one point
                         (--dtype float32 sets RTENSOR_DTYPE/TORCH_DTYPE;
                          --applevel runs variants 0/8/9/13 through pypy-c
                          instead of metatensor-bench, adding "app" rows;
                          triton rows always, torch-compile-ro/-mat rows
                          (CUDA graphs / max-autotune) always, jax/iree rows
                          when JAX_PYTHON is set - see README "Other backends";
                          --baselines-only adds only those rows)
  models [--baselines-only] [NAME...]
                         end-to-end models; no args = all
                         names: $ALL_MODELS
  ablation [EXP...]      ablations; no args = all
  lazy [PHASE|MODEL...]  where the DAG lives: the fusion pass against a
                         deferred library on the same runtime and the same
                         kernels; phases models warmup recovery
                         names: $ALL_EXPERIMENTS
  dynamic                dynamic sequence-length experiment
  control [REGIME...]    host-dependent control flow in a real model:
                         data-dependent early exit (DeeBERT/CALM style) on
                         distilgpt2, ours/torch-eager/torch-compile(-ro/-mat)/
                         jax-perlayer/jax-while, into \$OUT/control.tsv and
                         \$OUT/control_series.tsv (regimes: stable varying)
  batch [MODEL...]       batch-size sweep (1 2 4 8 16 32) for bert-mini and
                         distilgpt2 across ours/torch-eager/torch-compile/
                         torch-compile-ro/jax, into \$OUT/batch.tsv
                         (BATCHES and BATCH_SYSTEMS override the grid)
  motion                 a declared change rule against the hand-written
                         guard-recovery path: the rule is audited for target
                         knowledge first, then both arms are measured over
                         ten fresh processes, into \$OUT/motion.tsv
                         (MOTION_PYPY=... for a binary with live_bytes;
                         MOTION_CASE=axis|layout, MOTION_ARMS, MOTION_STEPS
                         as a list for the retention curve, MOTION_CACHE)
  decode                 next-token latency: greedy decode against a key/value
                         cache for the GPT-2 and SmolLM2 ladders and Qwen2.5,
                         ours against torch
                         eager/compile/compile-ro, same token stream required,
                         into \$OUT/decode.tsv (DECODE_PYPY=... for a binary
                         with decode_scores)
  transition             what a fused value costs to carry across a guard:
                         keeping its descriptor against draining it to
                         canonical form and re-recording, one binary and one
                         bit apart, into \$OUT/transition.tsv (needs a binary
                         with the drain knob: TRANSITION_PYPY=...)
  deopt [PATTERN...]     cost of a guard failure against the steady state
                         (patterns: never alternate both-hot fresh; plus the
                          recovery_probe (a)/(e) rows)
  warmup [--n N]         per-forward warm-up trace (distilgpt2, gpt2;
                         ours/torch-eager/torch-compile/torch-compile-ro/jax;
                         cold+warm kernel caches; N forwards, default 300)
  explain                torch._dynamo.explain graph structure per model,
                         into \$OUT/explain.tsv
  inventory [NAME...]    what each model computes - parameters (weights only
                         and weights plus buffers), GEMM/bmm/conv FLOPs and
                         call counts, plus the per-forward operator counts in
                         one vocabulary, ours against torch - into
                         \$OUT/model_inventory.tsv, \$OUT/op_inventory.tsv and
                         \$OUT/op_inventory_notes.txt (--no-torch for our rows
                         only, which needs no GPU)
  fusion [MODEL...]      fusion-region statistics per model forward (kernels,
                         nodes per kernel, why each region was cut) into
                         \$OUT/fusion.tsv; no args = all eleven paper models
  gap [MODEL...]         launches, kernel granularity and GPU utilisation per
                         system, into \$OUT/gap.tsv (needs nsys)
  correctness [NAME...]  each system against ours on five derived inputs per
                         model (INPUT_SEED=1..5), into \$OUT/correctness.tsv;
                         no args = all fourteen models (SEEDS overrides the seeds)
  summarize              render \$OUT/summary.md from the tsv files
  size                   render \$OUT/figures/impl_size.tex, print the table
  check [GROUP...]       smallest run that exercises every mode; groups are
                         prereq micro dtypes torch baselines models ablation
                         dynamic report
  plot [ARGS]            render the paper figures into \$OUT/figures
                         (--only NAME, --format pdf,png, --column single|double,
                          --texture for grayscale print, --titles for slides)
  compare DIR...         cross-machine figures: \$OUT against one or two other
                         result directories, comparing ratios rather than
                         microseconds
  all                    one shot: setup, export, micro, models, ablation,
                         dynamic, summarize, plot
                         (clears \$OUT/*.tsv first; SKIP_SETUP=1 to reuse an
                          existing build)
  list                   print model/experiment names and the micro grid
  help                   this message

several commands chain on a literal --, each keeping its own arguments:

    bench.sh micro --applevel -- models distilgpt2 -- summarize -- plot

one that fails while running is reported but does not abandon the rest;
a bad argument stops the whole chain before anything runs.

individual commands append to \$OUT/*.tsv (header written only if missing),
so several runs plus summarize compose one result set. 'all' starts clean.
EOF
}

cmd_list() {
  echo "models: $ALL_MODELS"
  echo "ablation experiments: $ALL_EXPERIMENTS"
  echo "micro grid: see run_micro.sh, or 'bench.sh micro V K N' for one point"
}

run_cmd() {
  local cmd=$1
  [ $# -gt 0 ] && shift

  case "$cmd" in
    setup)
      bash "$HERE/setup.sh" "$@"
      ;;
    export)
      bash "$HERE/export_weights.sh" "$@"
      ;;
    micro)
      dtype=""
      applevel=()
      while [ $# -gt 0 ]; do
        case "$1" in
          --dtype)    dtype=$2; shift 2 ;;
          --applevel) applevel=(--applevel); shift ;;
          --baselines-only) applevel=(--baselines-only); shift ;;
          *)          break ;;
        esac
      done
      if [ -n "$dtype" ]; then
        RTENSOR_DTYPE=$dtype TORCH_DTYPE=$dtype bash "$HERE/run_micro.sh" "${applevel[@]}" "$@"
      else
        bash "$HERE/run_micro.sh" "${applevel[@]}" "$@"
      fi
      ;;
    models)
      bonly=()
      if [ "$1" = "--baselines-only" ]; then bonly=(--baselines-only); shift; fi
      for m in "$@"; do
        case " $ALL_MODELS " in
          *" $m "*) ;;
          *) echo "bench.sh: unknown model '$m', valid: $ALL_MODELS" >&2; exit 1 ;;
        esac
      done
      bash "$HERE/run_models.sh" "${bonly[@]}" "$@"
      ;;
    ablation)
      for e in "$@"; do
        case " $ALL_EXPERIMENTS " in
          *" $e "*) ;;
          *) echo "bench.sh: unknown experiment '$e', valid: $ALL_EXPERIMENTS" >&2; exit 1 ;;
        esac
      done
      bash "$HERE/run_ablation.sh" "$@"
      ;;
    dynamic)
      bash "$HERE/run_dynamic.sh" "$@"
      ;;
    lazy)
      bash "$HERE/run_lazy.sh" "$@"
      ;;
    control)
      for r in "$@"; do
        case " stable varying " in
          *" $r "*) ;;
          *) echo "bench.sh: unknown regime '$r', valid: stable varying" >&2; exit 1 ;;
        esac
      done
      if [ $# -gt 0 ]; then REGIMES="$*"; export REGIMES; fi
      bash "$HERE/run_control.sh"
      ;;
    batch)
      bash "$HERE/run_batch.sh" "$@"
      ;;
    motion)
      bash "$HERE/run_motion.sh" "$@"
      ;;
    decode)
      bash "$HERE/run_decode.sh" "$@"
      ;;
    transition)
      bash "$HERE/run_transition.sh" "$@"
      ;;
    deopt)
      if [ $# -gt 0 ]; then PATTERNS="$*"; export PATTERNS; fi
      bash "$HERE/run_deopt.sh"
      ;;
    warmup)
      if [ "$1" = "--n" ]; then N=$2; shift 2; fi
      N="$N" bash "$HERE/run_warmup.sh" "$@"
      ;;
    explain)
      bash "$HERE/explain_models.sh" "$@"
      ;;
    inventory)
      source "$HERE/config.sh"
      WEIGHTS="$WEIGHTS" python3 "$HERE/model_inventory.py" "$OUT" "$@"
      ;;
    fusion)
      source "$HERE/config.sh"
      paper_setup_pypy
      trap paper_cleanup_pypy EXIT
      PYPY="$RUN_PYPY" WEIGHTS="$WEIGHTS" ITERS="$ITERS" WARMUP="$WARMUP" \
        JIT_FLAGS="$JIT_FLAGS" \
        "${RTENSOR_PYTHON:-python3}" "$HERE/fusion_stats.py" "$OUT" "$@"
      ;;
    gap)
      source "$HERE/config.sh"
      paper_setup_pypy
      trap paper_cleanup_pypy EXIT
      PYPY="$RUN_PYPY" WEIGHTS="$WEIGHTS" ITERS="$ITERS" WARMUP="$WARMUP" \
        ROUNDS="$ROUNDS" JIT_FLAGS="$JIT_FLAGS" \
        "${RTENSOR_PYTHON:-python3}" "$HERE/gap_analysis.py" "$OUT" "$@"
      ;;
    correctness)
      bash "$HERE/run_correctness.sh" "$@"
      ;;
    summarize)
      source "$HERE/config.sh"
      python3 "$HERE/summarize.py" "$OUT"
      ;;
    size)
      source "$HERE/config.sh"
      mkdir -p "$OUT/figures"
      "${RTENSOR_PYTHON:-python3}" "$HERE/impl_size.py" --tex "$OUT/figures/impl_size.tex"
      ;;
    check)
      bash "$HERE/check.sh" "$@"
      ;;
    plot)
      source "$HERE/config.sh"
      "${RTENSOR_PYTHON:-python3}" "$HERE/plot.py" "$OUT" "$@"
      ;;
    compare)
      if [ $# -lt 1 ]; then
        echo "bench.sh: compare needs at least one other result directory" >&2
        exit 1
      fi
      source "$HERE/config.sh"
      # Leading positionals are the other result directories; everything from
      # the first flag onward belongs to plot.py, values included.
      args=()
      while [ $# -gt 0 ]; do
        case "$1" in
          -*) break ;;
          *)  args+=(--compare "$1"); shift ;;
        esac
      done
      "${RTENSOR_PYTHON:-python3}" "$HERE/plot.py" "$OUT" "${args[@]}" "$@"
      ;;
    all)
      if [ "${SKIP_SETUP:-0}" != "1" ]; then
        echo "== setup =="
        bash "$HERE/setup.sh" || { echo "setup FAILED" >&2; exit 1; }
      fi
      source "$HERE/config.sh"
      rm -f "$OUT"/*.tsv
      LOG="$OUT/log.txt"
      exec 3>&2
      exec > >(tee -a "$LOG") 2>&1
      STEP_TOTAL=7
      STEP_N=0
      step() {
        local name=$1 skipvar=$2
        shift 2
        STEP_N=$((STEP_N + 1))
        if [ "${!skipvar}" = "1" ]; then
          echo "skipping $name ($skipvar=1)"
          return
        fi
        echo "== $name ($STEP_N/$STEP_TOTAL) =="
        "$@" || echo "$name FAILED (see log)"
      }
      step export SKIP_EXPORT bash "$HERE/export_weights.sh"
      step micro SKIP_MICRO bash "$HERE/run_micro.sh"
      step models SKIP_MODELS bash "$HERE/run_models.sh"
      step ablation SKIP_ABLATION bash "$HERE/run_ablation.sh"
      step dynamic SKIP_DYNAMIC bash "$HERE/run_dynamic.sh"
      step summarize SKIP_SUMMARIZE python3 "$HERE/summarize.py" "$OUT"
      step plot SKIP_PLOT "${RTENSOR_PYTHON:-python3}" "$HERE/plot.py" "$OUT"
      ;;
    list)
      cmd_list
      ;;
    help|"")
      usage
      ;;
    *)
      echo "bench.sh: unknown command '$cmd'" >&2
      usage >&2
      exit 1
      ;;
  esac
}

# Commands chain on a literal --, so each one still gets its own arguments:
#   bench.sh micro --applevel -- models distilgpt2 -- summarize
# A segment that fails while running does not abandon the ones after it - hours
# of measurement should not be lost to a summarize that could not find
# matplotlib - while a bad argument still stops everything before it starts.
total=1
for arg in "$@"; do
  [ "$arg" = "--" ] && total=$((total + 1))
done
status=0
n=0
seg=()
run_seg() {
  n=$((n + 1))
  [ "$total" -gt 1 ] && echo "== ${seg[0]} ($n/$total) =="
  run_cmd "${seg[@]}" || { status=1; echo "${seg[0]} FAILED" >&2; }
  seg=()
}
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    run_seg
  else
    seg+=("$arg")
  fi
done
run_seg
exit $status

