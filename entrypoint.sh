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
    echo "[entrypoint] No pre-compiled artifacts found, NxDI will compile from scratch"
fi

# ── Patch config.json for MistralConfig compatibility ─────────────────
# Models with model_type "ministral3" use rope_parameters instead of
# rope_scaling, and may have wrong top-level rope_theta.  Fix these
# so that NxDI/MistralConfig reads the correct values.
python3 << 'PATCH_EOF'
import json, os

config_path = os.path.join(os.environ.get("HF_HOME", "/cache"), "model", "config.json")
if not os.path.exists(config_path):
    exit(0)

with open(config_path) as f:
    cfg = json.load(f)

changed = False
rp = cfg.get("rope_parameters", {})
if rp:
    # Fix top-level rope_theta from rope_parameters
    rp_theta = rp.get("rope_theta")
    if rp_theta and cfg.get("rope_theta", 10000.0) != rp_theta:
        cfg["rope_theta"] = rp_theta
        changed = True
        print(f"[patch] rope_theta -> {rp_theta}")

    # Add rope_scaling from rope_parameters (MistralConfig compat)
    if "rope_scaling" not in cfg:
        cfg["rope_scaling"] = rp
        changed = True
        print("[patch] Added rope_scaling from rope_parameters")

if changed:
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[patch] Updated {config_path}")
PATCH_EOF

# ── Write launcher that registers ministral3 before starting vLLM ─────
cat > /tmp/vllm_launcher.py << 'LAUNCHER_EOF'
"""vLLM launcher with extended model type registration."""
import sys

# Register ministral3 -> MistralConfig so transformers recognises it
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.mistral.configuration_mistral import MistralConfig
    if "ministral3" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("ministral3", MistralConfig)
        print("[launcher] Registered ministral3 -> MistralConfig")
except Exception as e:
    print(f"[launcher] Warning: could not register ministral3: {e}")

# Launch vLLM
import runpy
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
LAUNCHER_EOF

# ── Launch vLLM with --model pointing to local dir ────────────────────
echo "[entrypoint] Starting vLLM with --model ${MODEL_DIR} ..."
exec python3 /tmp/vllm_launcher.py --model "${MODEL_DIR}" "$@"
