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
OUT=${OUT:-$HERE/../results/paper-$(date +%F)-$(hostname)}
export RTENSOR_BUDGET_MB=${RTENSOR_BUDGET_MB:-8}

mkdir -p "$OUT" "$WEIGHTS"

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
