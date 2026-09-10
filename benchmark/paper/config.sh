#!/bin/bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [ -f "$HERE/env.sh" ]; then
  source "$HERE/env.sh"
fi

PYPY=${PYPY:-$HERE/build/pypy-c}
BENCH=${BENCH:-$HERE/build/metatensor-bench}
TORCH_PYTHON=${TORCH_PYTHON:-${RTENSOR_PYTHON:-}}
RTENSOR_PYTHON=${RTENSOR_PYTHON:-$TORCH_PYTHON}
export RTENSOR_PYTHON TORCH_PYTHON
# Optional JAX/XLA + IREE venv (setup.sh writes it into env.sh). Unset or
# missing means those rows are skipped, nothing else changes.
JAX_PYTHON=${JAX_PYTHON:-}
if [ -n "$JAX_PYTHON" ] && ! [ -x "$JAX_PYTHON" ]; then
  echo "config.sh: JAX_PYTHON=$JAX_PYTHON is not executable; jax/iree rows skipped" >&2
  JAX_PYTHON=""
fi
export JAX_PYTHON
# Optional Torch-TensorRT venv (CPython 3.13: torch-tensorrt has no 3.14 wheel).
TRT_PYTHON=${TRT_PYTHON:-}
if [ -n "$TRT_PYTHON" ] && ! [ -x "$TRT_PYTHON" ]; then
  echo "config.sh: TRT_PYTHON=$TRT_PYTHON is not executable; tensorrt rows skipped" >&2
  TRT_PYTHON=""
fi
export TRT_PYTHON

if [ -z "$RTENSOR_CUBLAS" ] && [ -n "$RTENSOR_PYTHON" ]; then
  for f in "$(dirname "$RTENSOR_PYTHON")"/../lib/python3*/site-packages/nvidia/cu*/lib/libcublas.so.*; do
    [ -e "$f" ] && export RTENSOR_CUBLAS="$f" && break
  done
fi

WEIGHTS=${WEIGHTS:-$HERE/weights}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-30}
ROUNDS=${ROUNDS:-3}
JIT_FLAGS=${JIT_FLAGS:---jit threshold=3,function_threshold=3,trace_eagerness=2,trace_limit=60000}
# Results are filed under the machine and the accelerator that produced them,
# because a number here means nothing without both: results/<host>-<gpu>/paper-<date>.
gpu_slug() {
  local name
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  if [ -z "$name" ]; then
    echo "nogpu"
    return
  fi
  echo "$name" | tr 'A-Z' 'a-z' | sed -e 's/nvidia //' -e 's/geforce //' \
                                     -e 's/[^a-z0-9]//g'
}
RUN_HOST=${RUN_HOST:-$(hostname)}
RUN_GPU=${RUN_GPU:-$(gpu_slug)}
OUT=${OUT:-$HERE/../results/$RUN_HOST-$RUN_GPU/paper-$(date +%F)}
export RTENSOR_BUDGET_MB=${RTENSOR_BUDGET_MB:-8}

mkdir -p "$OUT" "$WEIGHTS"

# The directory name carries the host and the GPU; this carries everything else
# a reader needs to place the numbers - driver, toolkit, wheels, CPU.
write_machine_txt() {
  local f="$OUT/machine.txt"
  {
    echo "host          $RUN_HOST"
    echo "date          $(date -Iseconds)"
    echo "gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "gpu_memory    $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
    echo "compute_cap   ${RTENSOR_CC:-unknown}"
    echo "driver        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "cuda_home     ${CUDA_HOME:-unset}"
    echo "cpu           $(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
    echo "cpu_threads   $(nproc)"
    echo "kernel        $(uname -sr)"
    if [ -n "$RTENSOR_PYTHON" ] && [ -x "$RTENSOR_PYTHON" ]; then
      "$RTENSOR_PYTHON" - <<'PY' 2>/dev/null
import importlib
for mod in ("torch", "triton", "transformers"):
    try:
        print("%-13s %s" % (mod, importlib.import_module(mod).__version__))
    except Exception:
        print("%-13s -" % mod)
PY
    fi
    if [ -n "$JAX_PYTHON" ]; then
      "$JAX_PYTHON" - <<'PY' 2>/dev/null
import importlib
for mod, attr in (("jax", "__version__"), ("jaxlib", "__version__"),
                  ("iree.compiler", "version"), ("iree.runtime", "version")):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, attr, None)
        v = getattr(v, "VERSION", v) if v is not None else None
        if v is None:
            from importlib.metadata import version
            v = version(mod.replace("iree.", "iree-base-"))
        print("%-13s %s" % (mod, v))
    except Exception:
        print("%-13s -" % mod)
PY
    else
      echo "jax           - (JAX_PYTHON unset)"
    fi
    if [ -n "$TRT_PYTHON" ]; then
      "$TRT_PYTHON" - <<'PY' 2>/dev/null
import importlib
for mod in ("torch_tensorrt", "tensorrt"):
    try:
        print("%-13s %s" % (mod, importlib.import_module(mod).__version__))
    except Exception:
        print("%-13s -" % mod)
print("trt_torch     %s" % importlib.import_module("torch").__version__)
PY
    fi
    echo "iters         ${ITERS}"
    echo "rounds        ${ROUNDS}"
    echo "budget_mb     ${RTENSOR_BUDGET_MB}"
    echo "jit_flags     ${JIT_FLAGS}"
    echo "commit        $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  } > "$f"
}
write_machine_txt

if [ -z "$PYPY" ]; then
  echo "config.sh: PYPY not set (translated pypy-c with _metatensor)" >&2
fi
if [ -z "$BENCH" ]; then
  echo "config.sh: BENCH not set (translated metatensor-bench)" >&2
fi

# Every measurement also lands in $OUT/results.jsonl as a named-field record;
# the tsv files stay for the existing readers.
bench_record() {
  OUT="$OUT" "${RTENSOR_PYTHON:-python3}" "$HERE/record.py" "$@" || true
}

# Ours emits 12 fields and torch 13 - torch has no launch counter and adds
# graph/break counts - so the tail is read by field count, not by position.
# torch, jax, iree and triton now append compile_ms and first_run_ms (15).
record_micro_line() {
  local line=$1
  # shellcheck disable=SC2086
  set -- $line
  local mode=$1 variant=$2 k=$3 n=$4 iters=$5 warm=$6 steady=$7 kernels=$8
  local acc=$9
  shift 9
  local compiled=$1
  shift
  local extra=("$@")
  if [ ${#extra[@]} -ge 5 ]; then
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" graphs="${extra[0]}" \
      breaks="${extra[1]}" dtype="${extra[2]}" compile_ms="${extra[3]}" \
      first_run_ms="${extra[4]}"
  elif [ ${#extra[@]} -ge 3 ]; then
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" graphs="${extra[0]}" \
      breaks="${extra[1]}" dtype="${extra[2]}"
  else
    bench_record micro mode="$mode" variant="$variant" k="$k" n="$n" \
      iters="$iters" warm_s="$warm" steady_us="$steady" kernels="$kernels" \
      acc="$acc" compiled_in_timed="$compiled" launches_per_iter="${extra[0]}" \
      dtype="${extra[1]}"
  fi
}

tsv_init() {
  local path=$1 header=$2
  [ -f "$path" ] || echo -e "$header" > "$path"
}

steady_of() { echo "$1" | grep -o 'steady_us=[0-9.]*' | head -1 | cut -d= -f2; }
field_of() { echo "$1" | grep -o "$2=[0-9.eE+-]*" | head -1 | cut -d= -f2; }
diff_of() { echo "$1" | grep -o 'maxabsdiff=[0-9.eE+-]*' | head -1 | cut -d= -f2; }

PAPER_PYPY_LINK="$REPO/pypy-c-paper"
paper_setup_pypy() {
  if [ -n "$PYPY" ] && [ "$(dirname "$PYPY")" != "$REPO" ]; then
    ln -sf "$PYPY" "$PAPER_PYPY_LINK"
    RUN_PYPY="$PAPER_PYPY_LINK"
  else
    RUN_PYPY="$PYPY"
  fi
}
paper_cleanup_pypy() {
  rm -f "$PAPER_PYPY_LINK"
}

# Progress reporting. bench.sh all redirects stdout+stderr into tee for the
# log, and saves the original stderr on fd 3, so the bar keeps its terminal
# while the log gets one line per step.
PROG_WIDTH=24

_prog_fd() {
  if [ -t 3 ]; then echo 3; else echo 2; fi
}

_prog_hms() {
  printf '%d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60))
}

progress_init() {
  PROG_LABEL=$1
  PROG_TOTAL=$2
  PROG_N=0
  PROG_T0=$(date +%s)
}

progress_step() {
  local what=$1 finished=$PROG_N fd el eta bar='' i filled
  [ "${PROG_TOTAL:-0}" -gt 0 ] || return 0
  PROG_N=$((PROG_N + 1))
  fd=$(_prog_fd)
  el=$(( $(date +%s) - PROG_T0 ))
  if [ "$finished" -gt 0 ]; then
    eta=$(_prog_hms $(( el * (PROG_TOTAL - finished) / finished )))
  else
    eta='--:--:--'
  fi
  filled=$(( PROG_WIDTH * finished / PROG_TOTAL ))
  i=0
  while [ $i -lt $PROG_WIDTH ]; do
    if [ $i -lt $filled ]; then bar="$bar#"; else bar="$bar-"; fi
    i=$((i + 1))
  done
  if [ -t "$fd" ]; then
    printf '\r\033[K[%s] %s %d/%d  %s  eta %s' \
      "$bar" "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$what" "$eta" >&"$fd"
  else
    printf '[%s %d/%d] %s  elapsed %s  eta %s\n' \
      "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$what" "$(_prog_hms $el)" "$eta" >&"$fd"
  fi
}

progress_done() {
  local fd el
  [ "${PROG_TOTAL:-0}" -gt 0 ] || return 0
  fd=$(_prog_fd)
  el=$(( $(date +%s) - PROG_T0 ))
  [ -t "$fd" ] && printf '\r\033[K' >&"$fd"
  printf '%s: %d/%d done in %s\n' \
    "$PROG_LABEL" "$PROG_N" "$PROG_TOTAL" "$(_prog_hms $el)" >&"$fd"
}
