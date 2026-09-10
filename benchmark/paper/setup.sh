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

# setup.sh is documented as idempotent, so a re-run has to pick up what the
# first run detected.  env.sh holds CUDA_HOME (and the derived paths); without
# this a second run falls back to the /usr/local/cuda default and stops at the
# prerequisite check on any machine whose CUDA lives elsewhere.
if [ -f "$HERE/env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HERE/env.sh"
fi

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

echo "== python3 for the venv =="
# Triton and the torch wheels are CPython-only, so a PyPy3 "python3" on PATH
# (a pyenv shim, say) has to be skipped in favour of a real CPython.
is_cpython() {
  "$1" -c 'import platform, sys; sys.exit(0 if platform.python_implementation() == "CPython" else 1)' \
    >/dev/null 2>&1
}
PYTHON3="${PYTHON3:-}"
if [ -n "$PYTHON3" ]; then
  is_cpython "$PYTHON3" || { echo "setup.sh: PYTHON3=$PYTHON3 is not CPython" >&2; exit 1; }
  "$PYTHON3" -c 'import ensurepip, venv' >/dev/null 2>&1 || \
    { echo "setup.sh: $PYTHON3 lacks the venv/ensurepip modules" >&2; exit 1; }
else
  for cand in python3 python3.13 python3.12 python3.11 python3.10 \
              /usr/bin/python3 /usr/bin/python3.13 /usr/bin/python3.12 \
              /usr/bin/python3.11 /usr/bin/python3.10; do
    path=$(command -v "$cand" 2>/dev/null) || continue
    is_cpython "$path" || continue
    "$path" -c 'import ensurepip, venv' >/dev/null 2>&1 || continue
    PYTHON3="$path"
    break
  done
fi
if [ -z "$PYTHON3" ]; then
  echo "setup.sh: no CPython 3 with the venv module found; install one (e.g." >&2
  echo "  apt install python3 python3-venv) or set PYTHON3=/path/to/python3" >&2
  exit 1
fi
echo "using $PYTHON3 ($("$PYTHON3" --version 2>&1))"

echo "== python3 venv at $VENV =="
if [ -x "$VENV/bin/python" ] && ! is_cpython "$VENV/bin/python"; then
  echo "$VENV is not a CPython venv, recreating it"
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON3" -m venv "$VENV"
fi

# requirements.lock pins the torch/triton the numbers were measured with; that
# pairing is what decides whether the kernels run on the GPU at all. uv is
# fetched into $TOOLCHAIN like PyPy2, with pip as the fallback.
UV=$(command -v uv 2>/dev/null || true)
if [ -z "$UV" ] && [ -x "$TOOLCHAIN/uv/uv" ]; then
  UV="$TOOLCHAIN/uv/uv"
fi
if [ -z "$UV" ] && [ "${NO_UV:-0}" != "1" ]; then
  echo "no uv on PATH, downloading it into $TOOLCHAIN"
  mkdir -p "$TOOLCHAIN/uv"
  if curl -fsSL https://astral.sh/uv/install.sh \
       | env UV_INSTALL_DIR="$TOOLCHAIN/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 \
     && [ -x "$TOOLCHAIN/uv/uv" ]; then
    UV="$TOOLCHAIN/uv/uv"
  else
    echo "setup.sh: could not install uv, falling back to pip" >&2
  fi
fi

LOCK="$HERE/requirements.lock"
# The lock pulls torch/triton from the CUDA index and everything else from
# PyPI; uv only looks at the first index that has a package, so a pin like
# certifi== that the CUDA index carries at another version is unresolvable
# without letting it consider every index.
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
if [ -n "$UV" ] && [ -f "$LOCK" ]; then
  echo "installing from $(basename "$LOCK") with $("$UV" --version)"
  VIRTUAL_ENV="$VENV" "$UV" pip sync -q "$LOCK"
elif [ -n "$UV" ]; then
  echo "no requirements.lock yet, resolving from pyproject.toml"
  VIRTUAL_ENV="$VENV" "$UV" pip install -q -r "$HERE/pyproject.toml"
else
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q triton torch --index-url "$TORCH_INDEX"
  "$VENV/bin/pip" install -q transformers safetensors timm pillow numpy matplotlib
fi
RTENSOR_PYTHON="$VENV/bin/python"

# Optional baselines (JAX/XLA, IREE) live in their own venv: their CUDA wheels
# would otherwise move the torch/triton pins.  WITH_JAX=0 skips it; a missing
# venv only makes the jax/iree columns absent.
JAX_VENV="${JAX_VENV:-$VENV-jax}"
JAX_PYTHON=""
if [ "${WITH_JAX:-1}" = "1" ]; then
  echo "== optional backends venv at $JAX_VENV =="
  if [ -n "$UV" ]; then
    [ -x "$JAX_VENV/bin/python" ] || "$UV" venv -q --python "$PYTHON3" "$JAX_VENV"
    if VIRTUAL_ENV="$JAX_VENV" "$UV" pip sync -q "$HERE/requirements-jax.lock"; then
      JAX_PYTHON="$JAX_VENV/bin/python"
    else
      echo "setup.sh: jax/iree install failed; those columns will be absent" >&2
    fi
  else
    echo "setup.sh: no uv, skipping the jax/iree venv (WITH_JAX=0 silences this)" >&2
  fi
fi

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
export JAX_PYTHON="$JAX_PYTHON"
EOF
echo "wrote $HERE/env.sh"

mkdir -p "$BUILD"
echo "== translating metatensor-bench and pypy-c (this can take 15-60 minutes) =="
# Makefile owns the staleness checks (only rebuilds what actually moved).
BUILD="$BUILD" REPO="$REPO" PYTHON2="$PYTHON2" RTENSOR_PYTHON="$RTENSOR_PYTHON" \
  RTENSOR_CUBLAS="$RTENSOR_CUBLAS" MAKE_JOBS="$MAKE_JOBS" \
  make -C "$HERE" all

if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  echo "== exporting checkpoints into $WEIGHTS =="
  WEIGHTS="$WEIGHTS" RTENSOR_PYTHON="$RTENSOR_PYTHON" bash "$HERE/export_weights.sh"
else
  echo "skip weights export (SKIP_WEIGHTS=1)"
fi

echo "== done =="
echo "PYPY=$BUILD/pypy-c BENCH=$BUILD/metatensor-bench WEIGHTS=$WEIGHTS"
echo "next: benchmark/paper/run_all.sh"
