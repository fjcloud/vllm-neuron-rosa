# vLLM on AWS Inferentia2 — OpenShift / ROSA

Deploy LLM inference on AWS Inferentia2 using vLLM + Neuron SDK on OpenShift, with pre-compiled model artifacts from HuggingFace.

**Tested:** Mistral-7B-Instruct-v0.3 on `inf2.xlarge` — pre-compiled, no recompilation needed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  OpenShift / ROSA Cluster                                │
│                                                          │
│  ┌──────────────────┐  ┌─────────────────────────────┐   │
│  │ NFD Operator     │  │ KMM Operator                │   │
│  │ (node labeling)  │  │ (Neuron kernel module)      │   │
│  └──────────────────┘  └─────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ AWS Neuron Operator                                 │ │
│  │ ├─ neuron-device-plugin  (exposes Neuron devices)   │ │
│  │ ├─ neuron-scheduler      (device-aware scheduling)  │ │
│  │ └─ neuron-node-metrics   (Prometheus metrics)       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────── neuron-inference ns ────────────────────┐ │
│  │                                                     │ │
│  │  BuildConfig (Git) ──► ImageStream (vllm-neuron)    │ │
│  │                              │                      │ │
│  │  Deployment ─────────────────┘                      │ │
│  │  ├─ entrypoint.sh:                                  │ │
│  │  │   1. snapshot_download (HF repo → PVC)           │ │
│  │  │   2. set NEURON_COMPILED_ARTIFACTS               │ │
│  │  │   3. exec vLLM                                   │ │
│  │  ├─ PVC: vllm-neuron-cache (50Gi)                   │ │
│  │  └─ scheduler: neuron-scheduler                     │ │
│  │                                                     │ │
│  │  Service ──► Route (HTTPS edge, 300s timeout)       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  inf2.xlarge node                                        │
│  └─ 1 Neuron device (2 NeuronCores, 32 GB HBM)          │
└──────────────────────────────────────────────────────────┘
```

---

## How it works

The `entrypoint.sh` solves two problems in the vLLM + Neuron pipeline:

1. **vLLM's downloader ignores subdirectories** — it only fetches `*.safetensors` and `*.bin` at the repo root, skipping `neuron-compiled-artifacts/`. The entrypoint does a full `snapshot_download` first.

2. **NxDI uses a config hash to locate artifacts** — the hash changes depending on the model path, making pre-compiled artifacts hard to reuse. The entrypoint sets `NEURON_COMPILED_ARTIFACTS` to bypass the hash lookup entirely.

A dummy `model.safetensors` (161 bytes) in the HuggingFace repo satisfies `transformers`' hard-coded validation that checks for standard weight files before the Neuron loader takes over.

---

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| OpenShift / ROSA | 4.17+ | Tested on ROSA HCP |
| Node instance | `inf2.xlarge` or `inf2.8xlarge` | 1 Neuron device, 2 NeuronCores, 32 GB HBM |
| HuggingFace token | — | Only for gated base models (not needed for pre-compiled repo) |

### Instance sizing

| Instance | Devices | Cores | HBM | RAM | Use case |
|----------|---------|-------|-----|-----|----------|
| `inf2.xlarge` | 1 | 2 | 32 GB | 16 GB | Inference with pre-compiled + sharded weights |
| `inf2.8xlarge` | 1 | 2 | 32 GB | 64 GB | Compilation + inference |
| `inf2.24xlarge` | 6 | 12 | 192 GB | 384 GB | Large models (30B+) |
| `inf2.48xlarge` | 12 | 24 | 384 GB | 768 GB | Very large models (70B+) |

> `inf2.xlarge` and `inf2.8xlarge` have the **same Neuron chip**. The only difference is system RAM. With pre-sharded weights (`save_sharded_checkpoint`), each rank loads only ~7 GB instead of the full 14 GB, making `inf2.xlarge` viable for 7B models.

---

## Quick Start

### 1. Install operators (one-time)

```bash
oc apply -k https://github.com/fjcloud/vllm-neuron-rosa/deploy/prereqs
```

This installs:
- **NFD Operator** + instance + Neuron PCI rule
- **KMM Operator** (Kernel Module Management)
- **AWS Neuron Operator** (device plugin, scheduler, metrics)

Wait for all operators to be ready:
```bash
oc get csv -n openshift-nfd
oc get csv -n openshift-kmm
oc get pods -n ai-operator-on-aws
```

### 2. Create namespace

```bash
oc new-project neuron-inference
```

> **Note:** If using a gated model (e.g. `mistralai/Mistral-7B-Instruct-v0.3`), create a HF token secret:
> `oc create secret generic hf-token --from-literal=token=hf_YOUR_TOKEN -n neuron-inference`
> and add `HF_TOKEN` env var to the deployment.

### 3. Deploy

```bash
oc apply -k https://github.com/fjcloud/vllm-neuron-rosa/deploy -n neuron-inference
```

This creates:
- **BuildConfig** — builds the vLLM + Neuron image from the Dockerfile (via Git)
- **ImageStream** — stores the built image
- **PVC** (50 Gi) — caches the downloaded model and compiled artifacts
- **Deployment** — runs vLLM with `entrypoint.sh`
- **Service + Route** — exposes the OpenAI-compatible API

### 4. Start the build (first time only)

```bash
oc start-build vllm-neuron -n neuron-inference --follow
# ~10 minutes (large pip downloads)
```

The Deployment will automatically pick up the image once the build completes (via ImageStream trigger).

### 5. Watch startup

```bash
oc logs -f deployment/vllm-neuron -n neuron-inference
```

You should see:
```
[entrypoint] Downloading fjcloud/Mistral-7B-Instruct-v0.3-neuron-inf2-tp2 → /cache/model ...
[entrypoint] Download complete.
[entrypoint] NEURON_COMPILED_ARTIFACTS=/cache/model/neuron-compiled-artifacts
[entrypoint] Starting vLLM with --model /cache/model ...
...
INFO: Application startup complete.
```

### 6. Test

```bash
ROUTE=$(oc get route vllm-neuron -n neuron-inference -o jsonpath='{.spec.host}')

curl -sk "https://$ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/cache/model",
    "messages": [{"role": "user", "content": "Explain Kubernetes in 2 sentences."}],
    "max_tokens": 100
  }' | python3 -m json.tool
```

---

## Pre-compiled Model Repository

The HuggingFace repo [`fjcloud/Mistral-7B-Instruct-v0.3-neuron-inf2-tp2`](https://huggingface.co/fjcloud/Mistral-7B-Instruct-v0.3-neuron-inf2-tp2) contains:

```
├── config.json, tokenizer.json, ...   # Model config + tokenizer
├── model.safetensors                  # Dummy (161 bytes) — satisfies transformers validation
└── neuron-compiled-artifacts/
    ├── model.pt                       # Compiled NEFF (128 MB)
    ├── neuron_config.json             # NxDI configuration
    └── weights/
        ├── tp0_sharded_checkpoint.safetensors  (6.8 GB, rank 0)
        └── tp1_sharded_checkpoint.safetensors  (6.8 GB, rank 1)
```

Compiled with: `tp=2, max_model_len=4096, max_num_seqs=4, block_size=32, save_sharded_checkpoint=true`

---

## vLLM + Neuron Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--tensor-parallel-size 2` | 2 | Split across 2 NeuronCores (1 device = 2 cores) |
| `--max-model-len 4096` | 4096 | Maximum sequence length |
| `--max-num-seqs 4` | 4 | Max concurrent sequences |
| `--block-size 32` | 32 | KV cache block size for Neuron |
| `--num-gpu-blocks-override 4` | 4 | KV cache blocks (Neuron can't auto-detect) |
| `--no-enable-prefix-caching` | — | Not supported on Neuron |
| `save_sharded_checkpoint` | true | Pre-shard weights per rank (reduces RAM at load time) |
| `NEURON_CC_FLAGS` | `-O1` | Lower optimization = less RAM during compilation |

---

## Repo Structure

```
.
├── Dockerfile                         # UBI 10 + vLLM 0.13 + Neuron SDK
├── entrypoint.sh                      # Downloads HF repo, sets NEURON_COMPILED_ARTIFACTS, runs vLLM
└── deploy/
    ├── kustomization.yaml             # Main deployment
    ├── buildconfig.yaml               # Builds image from this Git repo
    ├── imagestream.yaml
    ├── deployment.yaml                # Neuron-scheduled, Recreate strategy
    ├── pvc.yaml                       # 50 Gi cache
    ├── service.yaml
    ├── route.yaml                     # HTTPS edge, 300s timeout
    └── prereqs/
        ├── kustomization.yaml         # Operator prerequisites
        ├── nfd-*.yaml                 # NFD Operator + instance + rule
        ├── kmm-*.yaml                 # KMM Operator
        └── neuron-*.yaml              # AWS Neuron Operator
```

---

## References

- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- [vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron)
- [AWS Neuron Operator for OpenShift](https://github.com/awslabs/operator-for-ai-chips-on-aws)
- [Pre-compiled model repo](https://huggingface.co/fjcloud/Mistral-7B-Instruct-v0.3-neuron-inf2-tp2)
