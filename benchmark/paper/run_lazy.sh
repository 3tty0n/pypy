#!/bin/bash
# Where the operation DAG lives, with everything else held fixed.
#
# Three arms, one binary, one kernel emitter, one kernel cache, one device
# allocator, one set of forcing boundaries:
#
#   virtual   the fusion pass keeps the DAG in the JIT's virtual objects, so
#             it is built once, when a trace is optimized
#   deferred  METATENSOR_LAZY=1 with the pass disabled: the same DAG is built
#             out of runtime objects on every iteration, the way a
#             conventional lazy tensor library builds it, and handed to the
#             same emitter and launcher
#   eager     the pass disabled and no deferral: one kernel per operation
#
# The claim the deferred arm is there to test is that the two fused arms run
# the SAME kernels, which the kernel and compile counters check directly, and
# that what virtual buys over deferred is host work, allocations and the
# absence of a per-iteration rebuild.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
APP="$HERE/../applevel"
paper_setup_pypy
trap paper_cleanup_pypy EXIT

TSV="$OUT/lazy.tsv"
tsv_init "$TSV" "model\tsystem\tround\tsteady_us\tlaunches_per_iter\tkernels\tcompiles\tgc_bytes_per_iter\tlazy_nodes\tlazy_forces\tlazy_barriers\tlazy_scans\tlazy_fallbacks\tfirst_run_ms\tchecksum\targmax_match\tbinary"
WTSV="$OUT/lazy_warmup.tsv"
tsv_init "$WTSV" "model\tsystem\tround\titer\tus\tcompiled\tlaunches\tbinary"

PYPY_SHA=$(sha256sum "$PYPY" 2>/dev/null | cut -c1-12)
NOFUSE="enable_opts=intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"
JIT_INNER="threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=60000"

LAZY_MODELS=${LAZY_MODELS:-"tiny-gpt2 distilgpt2 bert-tiny bert-mini vit-tiny mixer_b16 resnet18-b1 smollm2-135m"}

script_for() {
  case $1 in
    distilgpt2|tiny-gpt2) echo "gpt2.py";;
    smollm2-135m) echo "llama.py";;
    bert-tiny|bert-mini) echo "bert.py";;
    resnet18-b1|resnet18-b8) echo "resnet.py";;
    mixer_b16) echo "mixer.py";;
    vit-tiny) echo "vit.py";;
    *) echo "";;
  esac
}
weights_for() {
  case $1 in
    resnet18-b1|resnet18-b8) echo "$WEIGHTS/resnet18";;
    *) echo "$WEIGHTS/$1";;
  esac
}
extra_for() { case $1 in resnet18-b1) echo 1;; resnet18-b8) echo 8;; *) echo "";; esac; }

# One arm of one model.  Prints the driver's whole stdout; the caller pulls
# the fields out of it.
run_arm() {
  local system=$1 model=$2 script=$3 weights=$4 extra=$5
  case $system in
    virtual)  RTENSOR_LAZY_STATS=1 "$RUN_PYPY" $JIT_FLAGS "$APP/$script" "$weights" "$ITERS" "$WARMUP" $extra;;
    deferred) RTENSOR_LAZY_STATS=1 METATENSOR_LAZY=1 "$RUN_PYPY" -S --jit "$JIT_INNER,$NOFUSE" "$APP/$script" "$weights" "$ITERS" "$WARMUP" $extra;;
    eager)    RTENSOR_LAZY_STATS=1 "$RUN_PYPY" -S --jit "$JIT_INNER,$NOFUSE" "$APP/$script" "$weights" "$ITERS" "$WARMUP" $extra;;
  esac
}

argmax_of() { echo "$1" | grep '^argmax ' | head -1; }

row() {
  local model=$1 system=$2 round=$3 out=$4 match=$5
  local steady launches kern comp gcb nodes forces barriers scans fallbacks first cks
  steady=$(steady_of "$out")
  launches=$(field_of "$out" launches_per_iter)
  kern=$(field_of "$out" kernels)
  comp=$(field_of "$out" compiles)
  gcb=$(field_of "$out" gc_bytes_per_iter)
  nodes=$(field_of "$out" lazy_nodes)
  forces=$(field_of "$out" lazy_forces)
  barriers=$(field_of "$out" lazy_barriers)
  scans=$(field_of "$out" lazy_scans)
  fallbacks=$(field_of "$out" lazy_fallbacks)
  first=$(field_of "$out" first_run_ms)
  cks=$(field_of "$out" checksum)
  echo -e "$model\t$system\t$round\t$steady\t$launches\t$kern\t$comp\t$gcb\t$nodes\t$forces\t$barriers\t$scans\t$fallbacks\t$first\t$cks\t$match\t${PYPY_SHA:-unknown}" >> "$TSV"
  bench_record lazy model="$model" system="$system" round="$round" \
    steady_us="$steady" launches_per_iter="$launches" kernels="$kern" \
    compiles="$comp" gc_bytes_per_iter="$gcb" lazy_nodes="$nodes" \
    lazy_forces="$forces" lazy_barriers="$barriers" lazy_scans="$scans" \
    lazy_fallbacks="$fallbacks" first_run_ms="$first" checksum="$cks" \
    argmax_match="$match" binary="${PYPY_SHA:-unknown}"
}

do_model() {
  local model=$1
  local script=$(script_for "$model") weights=$(weights_for "$model") extra=$(extra_for "$model")
  if [ -z "$script" ]; then echo "run_lazy.sh: unknown model $model" >&2; return 0; fi
  for round in $(seq "$ROUNDS"); do
    progress_step "$model round $round/$ROUNDS"
    local ref="" refmax=""
    for system in virtual deferred eager; do
      local out match=""
      if ! out=$(run_arm "$system" "$model" "$script" "$weights" "$extra" 2>&1); then
        echo "run_lazy.sh: $model/$system failed" >&2
        echo "$out" | tail -3 >&2
        continue
      fi
      if [ "$system" = virtual ]; then
        refmax=$(argmax_of "$out"); match=1
      else
        [ "$(argmax_of "$out")" = "$refmax" ] && match=1 || match=0
      fi
      row "$model" "$system" "$round" "$out" "$match"
    done
  done
}

# Start-up: every forward of a fresh process, cold kernel cache, so the cost
# of building the DAG the first time is visible on both arms.
do_warmup() {
  local n=${WARMUP_N:-60}
  for model in tiny-gpt2 distilgpt2; do
    local script=$(script_for "$model") weights=$(weights_for "$model")
    for round in $(seq "$ROUNDS"); do
      for system in virtual deferred eager; do
        progress_step "warmup $model/$system round $round/$ROUNDS"
        local tmp; tmp=$(mktemp -d)
        local out
        if out=$(TMPDIR="$tmp" WARMUP_TRACE=$n run_arm "$system" "$model" "$script" "$weights" "" 2>&1); then
          echo "$out" | grep '^iter=' | while read -r line; do
            local i us c l
            i=$(echo "$line" | grep -o 'iter=[0-9]*' | cut -d= -f2)
            us=$(echo "$line" | grep -o 'us=[0-9.]*' | cut -d= -f2)
            c=$(echo "$line" | grep -o 'compiled=[0-9]*' | cut -d= -f2)
            l=$(echo "$line" | grep -o 'launches=[0-9]*' | cut -d= -f2)
            echo -e "$model\t$system\t$round\t$i\t$us\t$c\t$l\t${PYPY_SHA:-unknown}" >> "$WTSV"
          done
        else
          echo "run_lazy.sh: warmup $model/$system failed" >&2
        fi
        rm -rf "$tmp"
      done
    done
  done
}

# The micro grid, where the tensors are small enough that host work is the
# whole story, and large enough at the other end that it is not.
MTSV="$OUT/lazy_micro.tsv"
do_micro() {
  tsv_init "$MTSV" "variant\tk\tn\tsystem\tround\tsteady_us\tlaunches_per_iter\tkernels\tacc\tbinary"
  local points="0:1:10000 0:4:1000000 8:1:65536 11:1:256000 13:1:65536"
  for point in $points; do
    local v=${point%%:*} rest=${point#*:}
    local k=${rest%%:*} n=${rest#*:}
    for round in $(seq "$ROUNDS"); do
      progress_step "micro v$v k$k n$n round $round/$ROUNDS"
      for system in virtual deferred eager; do
        local line
        case $system in
          virtual)  line=$("$RUN_PYPY" $JIT_FLAGS "$APP/micro.py" app "$v" "$k" "$n" "$ITERS" "$WARMUP" 2>/dev/null | tail -1);;
          deferred) line=$(METATENSOR_LAZY=1 "$RUN_PYPY" -S --jit "$JIT_INNER,$NOFUSE" "$APP/micro.py" app "$v" "$k" "$n" "$ITERS" "$WARMUP" 2>/dev/null | tail -1);;
          eager)    line=$("$RUN_PYPY" -S --jit "$JIT_INNER,$NOFUSE" "$APP/micro.py" app "$v" "$k" "$n" "$ITERS" "$WARMUP" 2>/dev/null | tail -1);;
        esac
        if [ -z "$line" ]; then
          echo "run_lazy.sh: micro v$v/$system failed" >&2
          continue
        fi
        # mode variant k n iters warm steady kernels acc kernels_delta launches dtype
        local steady launches kern acc
        steady=$(echo "$line" | awk '"'"'{print $7}'"'"')
        kern=$(echo "$line" | awk '"'"'{print $10}'"'"')
        acc=$(echo "$line" | awk '"'"'{print $9}'"'"')
        launches=$(echo "$line" | awk '"'"'{print $11}'"'"')
        echo -e "$v\t$k\t$n\t$system\t$round\t$steady\t$launches\t$kern\t$acc\t${PYPY_SHA:-unknown}" >> "$MTSV"
        bench_record lazy_micro variant="$v" k="$k" n="$n" system="$system" \
          round="$round" steady_us="$steady" launches_per_iter="$launches" \
          kernels="$kern" acc="$acc" binary="${PYPY_SHA:-unknown}"
      done
    done
  done
}

# Recovery: the six probe cases have to agree with the interpreted reference
# on values and on the effect log under deferred execution too, or the
# deferred arm is not computing the same program.
do_recovery() {
  progress_step "recovery probe"
  local out
  if out=$(METATENSOR_LAZY=1 "$RUN_PYPY" -S --jit "$JIT_INNER,$NOFUSE" "$APP/recovery_probe.py" 2>&1); then
    echo "$out" > "$OUT/lazy_recovery.txt"
    echo "recovery probe (deferred): $(echo "$out" | grep -c -i 'ok\|match') ok lines, see lazy_recovery.txt"
  else
    echo "$out" > "$OUT/lazy_recovery.txt"
    echo "run_lazy.sh: recovery probe failed under deferred execution" >&2
  fi
}

PHASES=${PHASES:-"models micro warmup recovery"}
if [ "$#" -gt 0 ]; then
  case $1 in
    models|micro|warmup|recovery) PHASES="$*";;
    *) LAZY_MODELS="$*"; PHASES=models;;
  esac
fi

nsteps=0
for p in $PHASES; do
  case $p in
    models) nsteps=$((nsteps + $(echo $LAZY_MODELS | wc -w) * ROUNDS));;
    micro) nsteps=$((nsteps + 5 * ROUNDS));;
    warmup) nsteps=$((nsteps + 2 * 3 * ROUNDS));;
    recovery) nsteps=$((nsteps + 1));;
  esac
done
progress_init lazy "$nsteps"

for p in $PHASES; do
  case $p in
    models) for m in $LAZY_MODELS; do do_model "$m"; done;;
    micro) do_micro;;
    warmup) do_warmup;;
    recovery) do_recovery;;
    *) echo "run_lazy.sh: unknown phase '$p'" >&2; exit 1;;
  esac
done

progress_done
echo "wrote $TSV"
