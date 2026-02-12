# vLLM on AWS Neuron (Inferentia2) — OpenShift / ROSA

Deploy LLM inference on AWS Inferentia2 instances using vLLM, built from scratch on Red Hat UBI 10 with the Neuron SDK.

**Tested with:** Mistral-7B-Instruct-v0.3 on `inf2.8xlarge` (1 Neuron device, 2 NeuronCores, 32 GB HBM) — ~30 tok/s.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  OpenShift / ROSA Cluster                               │
│                                                         │
│  ┌──────────────────┐   ┌────────────────────────────┐  │
│  │ NFD Operator     │   │ AWS Neuron Operator         │  │
│  │ (node labeling)  │   │ ├─ neuron-device-plugin     │  │
│  │                  │   │ ├─ neuron-scheduler         │  │
│  └──────────────────┘   │ └─ neuron-node-metrics      │  │
│                         └────────────────────────────┘  │
│                                                         │
│  ┌─────────────── neuron-inference ns ───────────────┐  │
│  │  BuildConfig ──► ImageStream (vllm-neuron:latest) │  │
│  │                         │                         │  │
│  │  Deployment (neuron-vllm-test)                    │  │
│  │  ├─ init: fetch-model (downloads from HF)         │  │
│  │  ├─ main: granite (vLLM + Neuron)                 │  │
│  │  ├─ PVC: model-cache (50Gi)                       │  │
│  │  └─ scheduler: neuron-scheduler                   │  │
│  │                                                   │  │
│  │  Service ──► Route (HTTPS edge)                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  inf2.8xlarge node                                      │
│  └─ 1 Neuron device (2 NeuronCores, 32 GB HBM)         │
│  └─ 128 GB system RAM                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| OpenShift / ROSA | 4.17+ | Tested on ROSA HCP |
| Node instance | `inf2.8xlarge` | 1 Neuron device, 2 cores, 32 GB HBM, 128 GB RAM |
| AWS Neuron Driver | 2.25+ | Installed on the node via KMM |
| HuggingFace token | — | For gated models (Mistral, Llama) |

### Instance sizing guide

| Instance | Devices | Cores | HBM | RAM | Max model |
|----------|---------|-------|-----|-----|-----------|
| `inf2.xlarge` | 1 | 2 | 32 GB | 16 GB | ~3B params |
| `inf2.8xlarge` | 1 | 2 | 32 GB | 128 GB | ~7-8B params |
| `inf2.24xlarge` | 6 | 12 | 192 GB | 384 GB | ~30B params |
| `inf2.48xlarge` | 12 | 24 | 384 GB | 768 GB | ~70B params |

> **Important:** `inf2.8xlarge` has the same HBM (32 GB) as `inf2.xlarge` but 8x the system RAM (128 GB). The extra RAM is needed for Neuron compilation which is very memory-intensive (20-40 GB for 7B models).

---

## Step 1: Install Operators

### 1.1 Node Feature Discovery (NFD) Operator

NFD labels nodes with hardware capabilities so the Neuron operator knows which nodes have Inferentia chips.

```bash
# Install the operator (from Red Hat CoP GitOps Catalog)
oc apply -k https://github.com/redhat-cop/gitops-catalog/nfd/operator/overlays/stable

# Wait for the operator to be ready
oc get csv -n openshift-nfd -w

# Create the NFD instance
oc apply -k https://github.com/redhat-cop/gitops-catalog/nfd/instance/overlays/default
```

Verify:
```bash
oc get pods -n openshift-nfd
# Should see: nfd-controller-manager-xxx and nfd-worker-xxx (one per node)

# Check that the inf2 node has feature labels:
oc get node <inf2-node> -o json | jq '.metadata.labels | with_entries(select(.key | contains("feature")))'
```

### 1.2 AWS Neuron Operator

The Neuron operator deploys:
- **neuron-device-plugin**: exposes `aws.amazon.com/neuron` resources to Kubernetes
- **neuron-scheduler**: custom scheduler that manages Neuron device allocation
- **neuron-node-metrics**: Prometheus metrics for Neuron devices
- **KMM (Kernel Module Management)**: loads the Neuron kernel driver on nodes

```bash
# Install via OperatorHub:
# Operators → OperatorHub → search "AWS Neuron"
# Install in namespace: ai-operator-on-aws (created automatically)
# Approval: Automatic
```

Verify:
```bash
# Check operator pods
oc get pods -n ai-operator-on-aws
# Should see:
#   neuron-device-plugin-xxx    (DaemonSet on inf2 nodes)
#   neuron-node-metrics-xxx     (DaemonSet on inf2 nodes)
#   neuron-custom-scheduler-xxx (Deployment)
#   neuron-custom-scheduler-extension-xxx (Deployment)

# Check Neuron devices are discovered
oc get node <inf2-node> -o jsonpath='{.status.allocatable}' | jq .
# Should show: "aws.amazon.com/neuron": "1", "aws.amazon.com/neuroncore": "2"
```

### 1.3 (If needed) Add an Inf2 MachinePool

For ROSA HCP:
```bash
rosa create machinepool \
  --cluster=<cluster-name> \
  --name=neuron-inf2 \
  --instance-type=inf2.8xlarge \
  --replicas=1 \
  --availability-zone=<az>
```

Wait for the node to become `Ready`:
```bash
oc get nodes -l node.kubernetes.io/instance-type=inf2.8xlarge
```

---

## Step 2: Create the Project and Secrets

```bash
# Create namespace
oc new-project neuron-inference

# Create HuggingFace token secret (required for gated models like Mistral)
oc create secret generic hf-token \
  --from-literal=token=hf_YOUR_TOKEN_HERE \
  -n neuron-inference
```

---

## Step 3: Build the Custom vLLM + Neuron Image

### 3.1 The Dockerfile

The image is based on **Red Hat UBI 10 minimal** and installs:
- AWS Neuron runtime (`libnrt.so`) and collectives (`libnccom.so`) from the Neuron YUM repo (RPMs)
- Python Neuron SDK (`torch-neuronx`, `neuronx-cc`, NxDI) from pip
- vLLM 0.13.0 + vllm-neuron plugin

The key trick: **the Neuron RPM packages for Amazon Linux are compatible with UBI 10/RHEL** since both are RPM-based. You must run `ldconfig` after installing to register the libs in the dynamic linker cache, otherwise subprocesses won't find `libnrt.so.1`.

See [`Dockerfile`](./Dockerfile).

### 3.2 Build on OpenShift

```bash
# Create BuildConfig + ImageStream
oc new-build --binary --name=vllm-neuron --strategy=docker -n neuron-inference

# Start the build (from the directory containing the Dockerfile)
oc start-build vllm-neuron --from-dir=. -n neuron-inference --follow

# Build takes ~10 minutes (large pip downloads: torch, neuronx-cc)
```

Verify:
```bash
oc get builds -n neuron-inference
# vllm-neuron-1   Docker   Binary   Complete   ...

oc get is vllm-neuron -n neuron-inference
# vllm-neuron   image-registry.../neuron-inference/vllm-neuron   latest   ...
```

---

## Step 4: Deploy vLLM with Neuron

### 4.1 Create PVC for model cache

```yaml
# model-cache-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: neuron-inference
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi
  storageClassName: gp3-csi
```

```bash
oc apply -f model-cache-pvc.yaml
```

### 4.2 Create the Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuron-vllm-test
  namespace: neuron-inference
  labels:
    app: neuron-vllm-test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neuron-vllm-test
  template:
    metadata:
      labels:
        app: neuron-vllm-test
    spec:
      schedulerName: neuron-scheduler    # REQUIRED: uses the Neuron-aware scheduler
      initContainers:
      - name: fetch-model
        image: python:3.11-slim
        command: ["/bin/sh", "-c"]
        args:
        - |
          set -ex
          export PYTHONUSERBASE=/tmp/pip
          export PATH=$PYTHONUSERBASE/bin:$PATH
          pip install --no-cache-dir --user 'huggingface_hub>=1.0'
          if [ -f /model/.mistral_v3_done ]; then
            echo 'Mistral-7B already downloaded'
          else
            echo 'Downloading Mistral-7B-Instruct-v0.3...'
            rm -rf /model/*
            rm -rf /model/.*_done
            HF_HUB_DISABLE_XET=1 hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /model
            touch /model/.mistral_v3_done
            echo 'Download complete'
          fi
          ls -la /model/
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        - name: HF_HOME
          value: /model
        - name: HF_HUB_DISABLE_XET
          value: "1"
        volumeMounts:
        - name: model-volume
          mountPath: /model
        resources:
          requests:
            memory: 2Gi
          limits:
            memory: 4Gi
      containers:
      - name: granite
        image: image-registry.openshift-image-registry.svc:5000/neuron-inference/vllm-neuron:latest
        args:
        - "--model"
        - "/.cache"
        - "--max-model-len"
        - "4096"
        - "--no-enable-prefix-caching"
        - "--max-num-seqs"
        - "4"
        - "--block-size"
        - "32"
        - "--num-gpu-blocks-override"
        - "4"
        - "--tensor-parallel-size"
        - "2"
        env:
        - name: VLLM_SERVER_DEV_MODE
          value: "1"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        - name: HF_HOME
          value: "/.cache"
        - name: NEURON_CC_FLAGS
          value: "--auto-cast=matmul --auto-cast-type=bf16 -O1"
        - name: NEURON_COMPILED_ARTIFACTS
          value: "/.cache/neuron-compiled-artifacts"
        ports:
        - containerPort: 8000
        resources:
          requests:
            aws.amazon.com/neuron: "1"    # 1 Neuron device = 2 NeuronCores
            memory: 32Gi
          limits:
            aws.amazon.com/neuron: "1"
            memory: 64Gi
        volumeMounts:
        - name: model-volume
          mountPath: /.cache
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-cache
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 1Gi
```

```bash
oc apply -f deployment.yaml
```

### 4.3 Expose the service

```bash
# ClusterIP service
oc create service clusterip neuron-vllm-test --tcp=8000:8000 -n neuron-inference

# HTTPS route with extended timeout (Neuron compilation can take 10+ minutes)
oc create route edge neuron-vllm-test \
  --service=neuron-vllm-test \
  --port=8000 \
  -n neuron-inference

# Increase route timeout for long inference requests
oc annotate route neuron-vllm-test \
  haproxy.router.openshift.io/timeout=300s \
  -n neuron-inference --overwrite
```

---

## Step 5: Wait for Compilation

On first startup, vLLM + Neuron compiles the model for the NeuronCores. This takes **5-15 minutes** depending on model size and instance RAM.

```bash
# Watch the pod
oc logs -f deployment/neuron-vllm-test -c granite -n neuron-inference

# You'll see:
# 1. "Neuron plugin activated" — plugin loaded
# 2. "Resolved architecture: MistralForCausalLM" — model detected
# 3. "Starting compilation..." then dots (....) — Neuron compiler working
# 4. "Application startup complete." — READY!
```

> **Note:** Compilation artifacts are cached on the PVC (`/.cache/neuron-compiled-artifacts`). Subsequent restarts skip compilation and start in ~30 seconds.

---

## Step 6: Test

```bash
ROUTE=$(oc get route neuron-vllm-test -n neuron-inference -o jsonpath='{.spec.host}')

# Check models
curl -sk "https://$ROUTE/v1/models" | python3 -m json.tool

# Chat completion
curl -sk "https://$ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/.cache",
    "messages": [{"role": "user", "content": "Explain Kubernetes in 2 sentences."}],
    "max_tokens": 100,
    "temperature": 0.3
  }' | python3 -m json.tool
```

---

## Key vLLM + Neuron Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--tensor-parallel-size 2` | 2 | Split model across 2 NeuronCores (1 device = 2 cores) |
| `--max-model-len 4096` | 4096 | Maximum sequence length (reduce to save HBM) |
| `--no-enable-prefix-caching` | — | Prefix caching not supported on Neuron |
| `--block-size 32` | 32 | KV cache block size for Neuron |
| `--num-gpu-blocks-override 4` | 4 | Number of KV cache blocks (Neuron can't auto-detect) |
| `--max-num-seqs 4` | 4 | Max concurrent sequences |
| `NEURON_CC_FLAGS` | `--auto-cast=matmul --auto-cast-type=bf16 -O1` | Compiler flags: bf16 precision, low optimization (saves RAM during compilation) |

---

## Troubleshooting

### Pod stuck in `Pending`
```bash
oc describe pod <pod> -n neuron-inference
# Check for: "Insufficient aws.amazon.com/neuron" or scheduler errors
# Fix: Check that the Neuron device plugin is running and the node has available devices
oc get node <inf2-node> -o jsonpath='{.status.allocatable}' | grep neuron
```

### `libnrt.so.1: cannot open shared object file`
The Neuron runtime library is not in the linker cache. Ensure the Dockerfile has:
```dockerfile
RUN echo "/opt/aws/neuron/lib" > /etc/ld.so.conf.d/neuron.conf && ldconfig
```

### `libnccom.so: cannot open shared object file`
The `aws-neuronx-collectives` RPM is missing. Required for `tensor-parallel-size >= 2`:
```dockerfile
RUN microdnf install -y aws-neuronx-runtime-lib aws-neuronx-collectives
```

### OOMKilled during compilation
Neuron compilation is RAM-hungry (20-40 GB for 7B models). Solutions:
- Use `inf2.8xlarge` (128 GB RAM) instead of `inf2.xlarge` (16 GB)
- Set `NEURON_CC_FLAGS="-O1"` (lower optimization = less RAM)
- Increase container memory limits

### `NCC_EVRF009: Size of tensors exceeds HBM limit`
Model doesn't fit in NeuronCore HBM. Solutions:
- Increase `--tensor-parallel-size` (splits across more cores)
- Reduce `--max-model-len`
- Use a smaller model

### Neuron scheduler stuck (device shows in-use after pod deletion)
```bash
# Check node annotations
oc get node <inf2-node> -o jsonpath='{.metadata.annotations.NEURON_DEV_USAGE_MAP}'
# If "true" but no pod is using it:
oc annotate node <inf2-node> NEURON_DEV_USAGE_MAP=false --overwrite
oc annotate node <inf2-node> NEURON_CORE_USAGE_MAP=false,false --overwrite
# Restart scheduler
oc rollout restart deploy neuron-custom-scheduler -n ai-operator-on-aws
oc rollout restart deploy neuron-custom-scheduler-extension -n ai-operator-on-aws
```

### Image not updated after rebuild
OpenShift may cache the `:latest` tag. Force the digest:
```bash
DIGEST=$(oc get istag vllm-neuron:latest -n neuron-inference -o jsonpath='{.image.dockerImageReference}')
oc patch deployment neuron-vllm-test -n neuron-inference --type=json \
  -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/image\",\"value\":\"$DIGEST\"}]"
```

---

## Compilation Cache & Cost Optimization

### How compilation caching works

On first start, vLLM + NxDI compiles the model into Neuron-optimized `.neff` format. This is saved as `model.pt` at the path defined by `NEURON_COMPILED_ARTIFACTS` (on the PVC). On subsequent restarts, NxDI detects the cached `model.pt` and **skips compilation entirely**:

```
# First start (compilation):
[INFO] Starting compilation for the priority HLO
..........................  (8-15 min)
[INFO] Application startup complete.

# Subsequent starts (cached):
[INFO] Successfully loaded pre-compiled model artifacts from /.cache/neuron-compiled-artifacts
[INFO] Application startup complete.   (~2 min)
```

### What's persisted on the PVC

```
/.cache/                              # PVC mount (model-cache, 50Gi)
├── config.json                       # Model config (from HuggingFace)
├── model-00001-of-00005.safetensors  # Model weights
├── ...
├── neuron-compiled-artifacts/        # Compiled Neuron model (SURVIVES restart)
│   ├── model.pt                      # Serialized compiled model (~128 MB)
│   └── neuron_config.json            # Compilation config
└── neuron-cache/                     # Neuron compiler cache (SURVIVES restart)
    └── ...                           # Intermediate compiler results
```

### Environment variables for caching

```yaml
env:
- name: NEURON_COMPILED_ARTIFACTS    # Where NxDI saves/loads compiled model.pt
  value: "/.cache/neuron-compiled-artifacts"
- name: NEURON_CACHE_URL             # Where neuronx-cc caches compiler outputs
  value: "/.cache/neuron-cache"      # MUST be on PVC, NOT /tmp
```

> **Warning:** If you change `--max-model-len`, `--max-num-seqs`, `--tensor-parallel-size`, or `NEURON_CC_FLAGS`, the cached `model.pt` won't match and **NxDI will recompile**. Keep your parameters stable in production.

### Cost optimization: compile on inf2.8xlarge, run on inf2.xlarge

Since `inf2.xlarge` and `inf2.8xlarge` have the **same Neuron chip** (1 device, 2 cores, 32 GB HBM), the only difference is system RAM (16 GB vs 128 GB). Compilation needs ~30 GB RAM but inference only needs ~10-12 GB.

**Strategy:**
1. **Compile once** on `inf2.8xlarge` (128 GB RAM) — let the pod start, wait for `Application startup complete`
2. The compiled `model.pt` is saved to the PVC
3. **Scale down** the `inf2.8xlarge` machinepool
4. **Move to `inf2.xlarge`** (16 GB RAM, ~2.5x cheaper) — the pod loads `model.pt` directly, no recompilation
5. Reduce memory requests to fit `inf2.xlarge`:

```bash
# After compilation is cached, switch to inf2.xlarge-friendly resources:
oc patch deployment neuron-vllm-test -n neuron-inference --type=json -p '[
  {"op":"replace","path":"/spec/template/spec/containers/0/resources","value":{
    "requests":{"aws.amazon.com/neuron":"1","memory":"8Gi"},
    "limits":{"aws.amazon.com/neuron":"1","memory":"14Gi"}
  }}
]'
```

| Phase | Instance | RAM | Duration | Cost/h |
|-------|----------|-----|----------|--------|
| Compile (one-time) | inf2.8xlarge | 128 GB | ~10 min | $1.97 |
| Inference (ongoing) | inf2.xlarge | 16 GB | permanent | $0.76 |

> **Savings: ~60%** on ongoing inference costs.

---

## Performance Results (Mistral-7B on inf2.8xlarge)

| Metric | Value |
|--------|-------|
| Throughput | ~30 tok/s |
| Latency (short answer) | ~1-3s |
| Latency (500 tokens) | ~16s |
| Latency (1000 tokens) | ~30s |
| Max tested output | 2000 tokens (67s) |
| Compilation time (first start) | ~8 min |
| Restart time (cached) | ~30s |
| Memory usage (inference) | ~32 GB system + 32 GB HBM |

---

## Supported Models

Tested or expected to work with this setup:

| Model | Params | HBM needed (TP=2) | Status |
|-------|--------|--------------------|--------|
| Mistral-7B-Instruct-v0.3 | 7B | ~16 GB | **Validated** |
| Llama-3.1-8B-Instruct | 8B | ~18 GB | Expected to work |
| Qwen2.5-7B-Instruct | 7B | ~16 GB | Expected to work |
| Qwen3-8B | 8B | ~17 GB | Compiles but GQA→MHA conversion causes garbage output |

> **Note on Qwen3:** The Qwen3 "thinking" model (`<think>` tokens) produces incoherent output with TP=2 on Neuron due to the forced GQA→MHA attention conversion. Use non-thinking models (Mistral, Llama, Qwen2.5) for now.

---

## Files

```
.
├── Dockerfile          # UBI 10 + vLLM 0.13.0 + Neuron SDK
└── README.md           # This file
```

---

## References

- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)
- [vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron)
- [Neuron YUM Repository](https://yum.repos.neuron.amazonaws.com)
- [OpenShift NFD Operator](https://docs.openshift.com/container-platform/latest/hardware_enablement/psap-node-feature-discovery-operator.html)
