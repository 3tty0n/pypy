#!/bin/bash
# The PyTorch 2 benchmark suites, pinned the way torch's own CI pins them at
# the release the baselines use (v2.14.0): benchmarks/dynamo from that tag,
# TorchBench at .ci/docker/ci_commit_pins/torchbench.txt, transformers from
# huggingface-requirements.txt, timm at timm.txt.  suite_census.py and
# suite_export.py run in the venv this makes.
#
#   SUITES_VENV   (~/.venvs/torchbench)
#   DYNAMO_BENCH  (~/src/github.com/pytorch/dynamo-bench-v2.14.0)
#   TORCHBENCH    (~/src/github.com/pytorch/benchmark)
set -e
TAG=v2.14.0
TORCH="torch==2.14.0+cu130 torchvision==0.29.0+cu130"
VENV=${SUITES_VENV:-$HOME/.venvs/torchbench}
DYN=${DYNAMO_BENCH:-$HOME/src/github.com/pytorch/dynamo-bench-$TAG}
TB=${TORCHBENCH:-$HOME/src/github.com/pytorch/benchmark}
IDX="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130"

pin() { gh api "repos/pytorch/pytorch/contents/.ci/docker/ci_commit_pins/$1?ref=$TAG" --jq .content | base64 -d; }

mkdir -p "$DYN"
for f in common.py torchbench.py huggingface.py timm_models.py \
         torchbench.yaml huggingface.yaml timm_models.yaml \
         torchbench_models_list.txt huggingface_models_list.txt \
         timm_models_list.txt huggingface_llm_models.py; do
  gh api "repos/pytorch/pytorch/contents/benchmarks/dynamo/$f?ref=$TAG" \
    --jq .content | base64 -d > "$DYN/$f"
done

[ -d "$TB" ] || git clone -q --filter=blob:none https://github.com/pytorch/benchmark.git "$TB"
git -C "$TB" checkout -q "$(pin torchbench.txt)"

[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
export VIRTUAL_ENV=$VENV
uv pip install -q $IDX $TORCH torchaudio numpy pip wheel
(cd "$TB" && "$VENV/bin/python" install.py --continue_on_fail) || true
# install.py moves transformers and timm; put back the pins torch's CI uses.
# setuptools<81 keeps pkg_resources, which librosa (via transformers) imports.
uv pip install -q $IDX $TORCH $(pin huggingface-requirements.txt) \
  "timm @ git+https://github.com/huggingface/pytorch-image-models@$(pin timm.txt)" \
  pandas psutil pyyaml scipy tqdm "setuptools<81"
"$VENV/bin/python" -c "import torch, transformers, timm; print(torch.__version__, transformers.__version__, timm.__version__)"
