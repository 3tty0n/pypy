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

tsv_init() {
  local path=$1 header=$2
  [ -f "$path" ] || echo -e "$header" > "$path"
}

steady_of() { echo "$1" | grep -o 'steady_us=[0-9.]*' | head -1 | cut -d= -f2; }
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
