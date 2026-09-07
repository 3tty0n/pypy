#!/bin/bash
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TOOLCHAIN="$HERE/.toolchain"
BUILD="${BUILD:-$HERE/build}"
WEIGHTS="${WEIGHTS:-$HERE/weights}"
VENV="${VENV:-$HOME/.venvs/metatensor}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"

echo "== checking prerequisites =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || echo "setup.sh: nvidia-smi present but failed (driver/NVML mismatch is usually harmless for CUDA itself)" >&2
else
  echo "setup.sh: nvidia-smi not found; is the NVIDIA driver installed?" >&2
  exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -f "$CUDA_HOME/include/cuda.h" ]; then
  echo "setup.sh: $CUDA_HOME/include/cuda.h not found; set CUDA_HOME" >&2
  exit 1
fi

for tool in gcc make; do
  command -v "$tool" >/dev/null 2>&1 || { echo "setup.sh: $tool not found" >&2; exit 1; }
done

echo "== python2 for the RPython toolchain =="
PYTHON2=""
for cand in pypy2 python2; do
  if command -v "$cand" >/dev/null 2>&1; then
    PYTHON2=$(command -v "$cand")
    break
  fi
done
if [ -z "$PYTHON2" ]; then
  mkdir -p "$TOOLCHAIN"
  PYPY2_TGZ="$TOOLCHAIN/pypy2.7-linux64.tar.bz2"
  PYPY2_DIR="$TOOLCHAIN/pypy2.7-linux64"
  if [ ! -x "$PYPY2_DIR/bin/pypy" ]; then
    echo "no pypy2/python2 on PATH, downloading PyPy2.7 into $TOOLCHAIN"
    curl -fsSL -o "$PYPY2_TGZ" \
      https://downloads.python.org/pypy/pypy2.7-v7.3.17-linux64.tar.bz2
    mkdir -p "$PYPY2_DIR"
    tar -xjf "$PYPY2_TGZ" -C "$PYPY2_DIR" --strip-components=1
  fi
  PYTHON2="$PYPY2_DIR/bin/pypy"
fi
echo "using $PYTHON2 ($("$PYTHON2" --version 2>&1))"

echo "== python3 venv at $VENV =="
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q triton torch --index-url "$TORCH_INDEX"
"$VENV/bin/pip" install -q transformers safetensors timm pillow numpy
RTENSOR_PYTHON="$VENV/bin/python"

echo "== detecting compute capability =="
CC=$("$RTENSOR_PYTHON" -c '
import torch
major, minor = torch.cuda.get_device_capability()
print(major * 10 + minor)
')
echo "RTENSOR_CC=$CC"

RTENSOR_CUBLAS=""
for f in "$VENV"/lib/python3*/site-packages/nvidia/cu*/lib/libcublas.so.*; do
  [ -e "$f" ] && RTENSOR_CUBLAS="$f" && break
done

cat > "$HERE/env.sh" <<EOF
export RTENSOR_PYTHON="$RTENSOR_PYTHON"
export RTENSOR_CC="$CC"
export RTENSOR_CUBLAS="$RTENSOR_CUBLAS"
export CUDA_HOME="$CUDA_HOME"
EOF
echo "wrote $HERE/env.sh"

echo "== translating metatensor-bench =="
mkdir -p "$BUILD"
if [ ! -e "$BUILD/metatensor-bench" ] || [ "$HERE/../metatensor_bench.py" -nt "$BUILD/metatensor-bench" ]; then
  ( cd "$BUILD" && RTENSOR_PYTHON="$RTENSOR_PYTHON" RTENSOR_CUBLAS="$RTENSOR_CUBLAS" \
    PYTHON2="$PYTHON2" MAKE_JOBS="$MAKE_JOBS" \
    "$REPO/benchmark/build.sh" "$BUILD/metatensor-bench" )
else
  echo "skip metatensor-bench (up to date)"
fi

echo "== translating pypy-c (this can take 15-60 minutes) =="
if [ ! -e "$BUILD/pypy-c" ]; then
  ( cd "$BUILD" && PYTHONPATH="$REPO" "$PYTHON2" "$REPO/rpython/bin/rpython" \
      --batch --make-jobs="$MAKE_JOBS" -Ojit --no-shared \
      --output="$BUILD/pypy-c" \
      "$REPO/pypy/goal/targetpypystandalone.py" --withmod-_metatensor )
else
  echo "skip pypy-c (already built)"
fi

if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  echo "== exporting checkpoints into $WEIGHTS =="
  WEIGHTS="$WEIGHTS" RTENSOR_PYTHON="$RTENSOR_PYTHON" bash "$HERE/export_weights.sh"
else
  echo "skip weights export (SKIP_WEIGHTS=1)"
fi

echo "== done =="
echo "PYPY=$BUILD/pypy-c BENCH=$BUILD/metatensor-bench WEIGHTS=$WEIGHTS"
echo "next: benchmark/paper/run_all.sh"
