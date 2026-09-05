#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
OUT=${1:-$HERE/rtensor-bench}
PYTHONPATH=$ROOT ${PYTHON2:-python2} "$ROOT/rpython/bin/rpython" --batch --make-jobs=${MAKE_JOBS:-4} -Ojit --output="$OUT" "$HERE/rtensor_bench.py"
