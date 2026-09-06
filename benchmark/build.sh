#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -z "$RTENSOR_CUBLAS" ] && [ -n "$RTENSOR_PYTHON" ]; then
  for f in "$(dirname "$RTENSOR_PYTHON")"/../lib/python3*/site-packages/nvidia/cu*/lib/libcublas.so.*; do
    [ -e "$f" ] && export RTENSOR_CUBLAS="$f" && break
  done
fi
ROOT=$(dirname "$HERE")
OUT=${1:-$HERE/metatensor-bench}
PYTHONPATH=$ROOT ${PYTHON2:-python2} "$ROOT/rpython/bin/rpython" --batch --make-jobs=${MAKE_JOBS:-4} -Ojit --output="$OUT" "$HERE/metatensor_bench.py"
