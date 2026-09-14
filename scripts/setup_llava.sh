#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$PROJECT_DIR/third_party/LLaVA-NeXT"
SOURCE_COMMIT=bce12e479bc4dfee2b9c50c88137b01ff51bd483
CONDA="${VIDEOQA_CONDA:-/App/conda/bin/conda}"
ENV_PREFIX="${VIDEOQA_ENV_PREFIX:-/App/conda/envs/explore_video_qa}"
if [[ ! -d "$SOURCE_DIR" ]]; then
    mkdir -p "$PROJECT_DIR/third_party"
    git clone --no-checkout --filter=blob:none https://github.com/LLaVA-VL/LLaVA-NeXT.git "$SOURCE_DIR"
    git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"
fi
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { echo 'Unexpected source commit; inspect the checkout before proceeding.' >&2; exit 1; }
[[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]] || { echo 'Upstream source has local changes.' >&2; exit 1; }
if [[ ! -e "$ENV_PREFIX/conda-meta/history" ]]; then
    if [[ -f "$PROJECT_DIR/conda-linux-64.lock.txt" ]]; then
        "$CONDA" create -y -p "$ENV_PREFIX" --file "$PROJECT_DIR/conda-linux-64.lock.txt"
    else
        "$CONDA" create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.10 pip ffmpeg
    fi
fi
PYTHON="$ENV_PREFIX/bin/python"
"$PYTHON" -c 'import sys; from pathlib import Path; assert sys.version_info[:2] == (3, 10); assert (Path(sys.prefix) / "conda-meta/history").exists()'
REQUIREMENTS="$PROJECT_DIR/requirements-runtime.txt"
if [[ -f "$PROJECT_DIR/requirements-runtime.lock.txt" ]]; then
    REQUIREMENTS="$PROJECT_DIR/requirements-runtime.lock.txt"
fi
"$PYTHON" -m pip install --disable-pip-version-check -r "$REQUIREMENTS"
"$PYTHON" -m pip install --no-deps -e "$SOURCE_DIR"
"$PYTHON" -m pip check
"$PYTHON" "$PROJECT_DIR/scripts/prepare_runtime.py"
