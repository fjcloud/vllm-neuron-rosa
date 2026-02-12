#!/bin/bash
set -euo pipefail

# ── Pre-download model + Neuron compiled artifacts from HuggingFace ───
# vLLM's default downloader only fetches *.safetensors and *.bin at the
# repo root, ignoring the neuron-compiled-artifacts/ subdirectory.
# This entrypoint does a full snapshot_download first, then points
# NEURON_COMPILED_ARTIFACTS at the right path so NxDI skips hashing.

MODEL_ID="${HF_MODEL_ID:?HF_MODEL_ID must be set}"
MODEL_DIR="${HF_HOME:-/cache}/model"

echo "[entrypoint] Downloading ${MODEL_ID} → ${MODEL_DIR} ..."
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    '${MODEL_ID}',
    local_dir='${MODEL_DIR}',
    token=os.environ.get('HF_TOKEN'),
)
print('[entrypoint] Download complete.')
"

# ── Point NxDI at the compiled artifacts (bypasses config hash) ───────
ARTIFACTS_DIR="${MODEL_DIR}/neuron-compiled-artifacts"
if [ -f "${ARTIFACTS_DIR}/neuron_config.json" ]; then
    export NEURON_COMPILED_ARTIFACTS="${ARTIFACTS_DIR}"
    echo "[entrypoint] NEURON_COMPILED_ARTIFACTS=${ARTIFACTS_DIR}"
else
    echo "[entrypoint] WARNING: ${ARTIFACTS_DIR}/neuron_config.json not found, NxDI will compile from scratch"
fi

# ── Launch vLLM with --model pointing to local dir ────────────────────
echo "[entrypoint] Starting vLLM with --model ${MODEL_DIR} ..."
exec python -m vllm.entrypoints.openai.api_server --model "${MODEL_DIR}" "$@"
