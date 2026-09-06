#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
LOG="$OUT/log.txt"
exec > >(tee -a "$LOG") 2>&1

step() {
  local name=$1 skipvar=$2
  shift 2
  if [ "${!skipvar}" = "1" ]; then
    echo "skipping $name ($skipvar=1)"
    return
  fi
  echo "== $name =="
  "$@" || echo "$name FAILED (see log)"
}

step export SKIP_EXPORT bash "$HERE/export_weights.sh"
step micro SKIP_MICRO bash "$HERE/run_micro.sh"
step models SKIP_MODELS bash "$HERE/run_models.sh"
step ablation SKIP_ABLATION bash "$HERE/run_ablation.sh"
step dynamic SKIP_DYNAMIC bash "$HERE/run_dynamic.sh"
step summarize SKIP_SUMMARIZE python3 "$HERE/summarize.py" "$OUT"
