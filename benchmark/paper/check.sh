#!/bin/bash
# Smallest run that still exercises every mode, so a broken setup fails in
# minutes rather than after a full grid. Writes to a scratch $OUT and leaves
# the real results alone.
#
#     bench.sh check              # everything
#     bench.sh check micro        # one group: micro, dtypes, torch, models,
#                                 # baselines, ablation, dynamic, report
#
# The check that matters most is launches_per_iter: a kernel that fails to load
# falls back to the CPU silently, and the run still produces plausible numbers.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

CHECK_OUT=${CHECK_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/metatensor-check-XXXXXX")}
OUT="$CHECK_OUT"
export OUT
ITERS=${CHECK_ITERS:-2}
ROUNDS=1
export ITERS ROUNDS
source "$HERE/config.sh"

PASS=0 FAIL=0 SKIP=0
FAILED=()

ok()   { printf '  \033[32mok\033[0m    %-34s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %-34s %s\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); FAILED+=("$1"); }
skip() { printf '  --    %-34s %s\n' "$1" "${2:-}"; SKIP=$((SKIP+1)); }

# field N of a bench line: 7 steady_us, 9 acc, 11 launches/iter
field() { echo "$1" | awk -v i="$2" '{print $i}'; }
positive() { awk -v v="$1" 'BEGIN{exit !(v+0 > 0)}'; }
close_enough() { awk -v a="$1" -v b="$2" 'BEGIN{d=a-b; if(d<0)d=-d; m=(a<0?-a:a); exit !(d <= 1e-6 * (m>1?m:1))}'; }

# --- prerequisites ---------------------------------------------------------
group_prereq() {
  echo "== prerequisites =="
  if [ -x "$BENCH" ]; then ok "metatensor-bench built"; else
    bad "metatensor-bench built" "$BENCH missing - run bench.sh setup"; return; fi
  if [ -x "$PYPY" ]; then ok "pypy-c built"; else
    bad "pypy-c built" "$PYPY missing - run bench.sh setup"; fi
  if [ -n "${RTENSOR_PYTHON:-}" ] && [ -x "$RTENSOR_PYTHON" ]; then
    ok "python3 venv" "$("$RTENSOR_PYTHON" --version 2>&1)"
  else
    bad "python3 venv" "RTENSOR_PYTHON unset or missing"; fi
  if "$RTENSOR_PYTHON" -c "import torch, triton" 2>/dev/null; then
    ok "torch and triton import"
  else
    bad "torch and triton import"; fi
  if [ -n "${RTENSOR_CUBLAS:-}" ] && [ -e "${RTENSOR_CUBLAS:-}" ]; then
    ok "cuBLAS resolved" "$(basename "$RTENSOR_CUBLAS")"
  else
    bad "cuBLAS resolved" "RTENSOR_CUBLAS does not point at a library"; fi
}

# --- our three execution modes --------------------------------------------
group_micro() {
  echo "== execution modes (variant 8, n=25600) =="
  local acc0="" line mode
  for mode in fused eager nojit; do
    if ! line=$("$BENCH" "$mode" 8 1 25600 "$ITERS" 2>&1); then
      bad "$mode runs" "$(echo "$line" | tail -1)"; continue
    fi
    local steady launches acc
    steady=$(field "$line" 7); launches=$(field "$line" 11); acc=$(field "$line" 9)
    if ! positive "$steady"; then bad "$mode steady_us" "got $steady"; continue; fi
    # A kernel that would not load runs on the CPU and reports zero launches.
    if ! positive "$launches"; then
      bad "$mode runs on the GPU" "launches/iter=$launches, kernels fell back to the CPU"
      continue
    fi
    ok "$mode" "steady=${steady}us launches/iter=$launches"
    if [ -z "$acc0" ]; then acc0=$acc
    elif close_enough "$acc" "$acc0"; then ok "$mode agrees with fused" "acc=$acc"
    else bad "$mode agrees with fused" "acc=$acc vs $acc0"; fi
  done
  ACC_OURS=$acc0
}

group_dtypes() {
  echo "== dtypes (variant 11, the reduction path) =="
  local dt line launches
  for dt in float64 float32 float16; do
    if ! line=$(RTENSOR_DTYPE=$dt "$BENCH" fused 11 1 25600 "$ITERS" 2>&1); then
      bad "$dt runs" "$(echo "$line" | tail -1)"; continue
    fi
    if echo "$line" | grep -q "Parse MLIR file failed"; then
      bad "$dt emits valid TTIR" "triton refused the generated kernel"; continue
    fi
    launches=$(echo "$line" | grep -E "^fused" | awk '{print $11}')
    if positive "${launches:-0}"; then ok "$dt" "launches/iter=$launches"
    else bad "$dt runs on the GPU" "launches/iter=${launches:-none}"; fi
  done
}

group_torch() {
  echo "== torch baselines =="
  local mode line
  for mode in compile eager; do
    line=$("$TORCH_PYTHON" "$HERE/../torch_bench.py" "$mode" 8 1 25600 "$ITERS" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then bad "torch-$mode runs"; continue; fi
    local steady acc
    steady=$(field "$line" 7); acc=$(field "$line" 9)
    if positive "$steady"; then ok "torch-$mode" "steady=${steady}us"
    else bad "torch-$mode steady_us" "got $steady"; fi
    if [ -n "${ACC_OURS:-}" ] && [ "$mode" = eager ]; then
      if close_enough "$acc" "$ACC_OURS"; then ok "ours agrees with torch" "acc=$acc"
      else bad "ours agrees with torch" "acc=$acc vs $ACC_OURS"; fi
    fi
  done
}

group_baselines() {
  echo "== jax / iree / triton baselines =="
  local line steady acc
  line=$("$TORCH_PYTHON" "$HERE/../triton_bench.py" triton 12 1 25600 "$ITERS" 2>/dev/null | tail -1)
  if [ -z "$line" ]; then bad "triton runs"; else
    steady=$(field "$line" 7); acc=$(field "$line" 9)
    if positive "$steady"; then ok "triton" "steady=${steady}us"; else bad "triton steady_us" "got $steady"; fi
    ref=$("$TORCH_PYTHON" "$HERE/../torch_bench.py" eager 12 1 25600 "$ITERS" 2>/dev/null | tail -1)
    if close_enough "$acc" "$(field "$ref" 9)"; then ok "triton agrees with torch" "acc=$acc"
    else bad "triton agrees with torch" "acc=$acc vs $(field "$ref" 9)"; fi
  fi
  if [ -n "${TRT_PYTHON:-}" ]; then
    line=$("$TRT_PYTHON" "$HERE/../torch_bench.py" tensorrt 12 1 25600 "$ITERS" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then bad "tensorrt runs"; else
      steady=$(field "$line" 7); acc=$(field "$line" 9)
      if positive "$steady"; then ok "tensorrt" "steady=${steady}us"; else bad "tensorrt steady_us" "got $steady"; fi
      if [ -n "${ref:-}" ] && close_enough "$acc" "$(field "$ref" 9)"; then ok "tensorrt agrees with torch" "acc=$acc"
      else bad "tensorrt agrees with torch" "acc=$acc vs $(field "$ref" 9)"; fi
    fi
  else skip "tensorrt" "TRT_PYTHON unset (setup.sh with WITH_TRT=1)"; fi
  if [ -z "${JAX_PYTHON:-}" ]; then skip "jax" "JAX_PYTHON unset (setup.sh with WITH_JAX=1)"; return; fi
  local mode
  for mode in jax iree; do
    line=$("$JAX_PYTHON" "$HERE/../jax_bench.py" "$mode" 12 1 25600 "$ITERS" 2>/dev/null | tail -1)
    if [ -z "$line" ]; then bad "$mode runs"; continue; fi
    steady=$(field "$line" 7); acc=$(field "$line" 9)
    if positive "$steady"; then ok "$mode" "steady=${steady}us"; else bad "$mode steady_us" "got $steady"; fi
    if [ -n "${ref:-}" ]; then
      if close_enough "$acc" "$(field "$ref" 9)"; then ok "$mode agrees with torch" "acc=$acc"
      else bad "$mode agrees with torch" "acc=$acc vs $(field "$ref" 9)"; fi
    fi
  done
}

group_models() {
  echo "== model path (tiny-gpt2) =="
  if [ ! -d "$WEIGHTS/tiny-gpt2" ]; then
    skip "tiny-gpt2" "no checkpoint, run bench.sh export"; return
  fi
  if ! bash "$HERE/run_models.sh" tiny-gpt2 >/dev/null 2>&1; then
    bad "run_models.sh"; return
  fi
  local ours diff
  ours=$(awk -F'\t' '$2=="ours"{print $4}' "$OUT/models.tsv" | head -1)
  diff=$(awk -F'\t' '$2=="torch-eager"{print $5}' "$OUT/models.tsv" | head -1)
  if positive "${ours:-0}"; then ok "tiny-gpt2 runs" "steady=${ours}us"
  else bad "tiny-gpt2 runs" "no steady_us in models.tsv"; fi
  if [ -n "$diff" ] && awk -v d="$diff" 'BEGIN{exit !(d+0 < 1e-3)}'; then
    ok "tiny-gpt2 matches torch" "maxabsdiff=$diff"
  else
    bad "tiny-gpt2 matches torch" "maxabsdiff=${diff:-missing}"
  fi
}

group_ablation() {
  echo "== ablation and dynamic paths =="
  if bash "$HERE/run_ablation.sh" budget_mb >/dev/null 2>&1 &&
     [ "$(wc -l < "$OUT/ablation.tsv")" -gt 1 ]; then
    ok "run_ablation.sh" "$(( $(wc -l < "$OUT/ablation.tsv") - 1 )) rows"
  else
    bad "run_ablation.sh"
  fi
}

group_dynamic() {
  if [ ! -d "$WEIGHTS/distilgpt2" ]; then
    skip "run_dynamic.sh" "no distilgpt2 checkpoint"; return
  fi
  # The app cycles through five sequence lengths, so a full cycle is the
  # smallest run that gives every length a sample.
  if ITERS=${CHECK_DYNAMIC_ITERS:-5} bash "$HERE/run_dynamic.sh" >/dev/null 2>&1 &&
     [ -s "$OUT/dynamic_summary.tsv" ]; then
    ok "run_dynamic.sh" "$(( $(wc -l < "$OUT/dynamic_summary.tsv") - 1 )) rows"
  else
    bad "run_dynamic.sh"
  fi
}

group_report() {
  echo "== reporting =="
  if [ -s "$OUT/results.jsonl" ] &&
     "$RTENSOR_PYTHON" -c "
import json, sys
for line in open('$OUT/results.jsonl'):
    json.loads(line)
" 2>/dev/null; then
    ok "results.jsonl" "$(grep -c . "$OUT/results.jsonl") valid records"
  else
    bad "results.jsonl" "missing or not valid JSON"
  fi
  if python3 "$HERE/summarize.py" "$OUT" >/dev/null 2>&1 && [ -s "$OUT/summary.md" ]; then
    ok "summarize.py"
  else
    bad "summarize.py"
  fi
  if "$RTENSOR_PYTHON" "$HERE/plot.py" "$OUT" --format pdf >/dev/null 2>&1 &&
     [ -n "$(ls "$OUT/figures"/*.pdf 2>/dev/null)" ]; then
    ok "plot.py" "$(ls "$OUT/figures"/*.pdf | wc -l) figures"
  else
    bad "plot.py"
  fi
}

CHECK_GROUPS=${*:-prereq micro dtypes torch baselines models ablation dynamic report}
started=$SECONDS
for g in $CHECK_GROUPS; do
  case "$g" in
    prereq|micro|dtypes|torch|baselines|models|ablation|dynamic|report) "group_$g" ;;
    *) echo "check.sh: unknown group '$g'" >&2; exit 2 ;;
  esac
done

echo
printf '%d passed, %d failed, %d skipped in %ds\n' "$PASS" "$FAIL" "$SKIP" "$((SECONDS-started))"
if [ "$FAIL" -gt 0 ]; then
  printf 'failed: %s\n' "${FAILED[*]}"
  echo "scratch output kept at $CHECK_OUT"
  exit 1
fi
rm -rf "$CHECK_OUT"
