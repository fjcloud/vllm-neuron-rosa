# vLLM on AWS Inferentia2 — OpenShift / ROSA

Run pre-compiled LLMs on Inferentia2 with vLLM + Neuron SDK. No compilation needed.

## Prerequisites

- OpenShift / ROSA 4.17+ — deploy a cluster with [`fjcloud/rosa-quickstart`](https://github.com/fjcloud/rosa-quickstart)
- An `inf2.xlarge` (or larger) node:
  ```bash
  rosa create machine-pool -c <cluster-name> \
    --name neuron --replicas=1 \
    --instance-type inf2.xlarge \
    --availability-zone <az>
  ```

## Deploy

```bash
# 1. Install operators (NFD, KMM, Neuron)
oc apply -k https://github.com/fjcloud/vllm-neuron-rosa/deploy/prereqs/operators

# Wait for operators to be ready
oc get csv -n openshift-nfd -w
oc get csv -n openshift-kmm -w

# 2. Create operator instances (NFD discovery + Neuron device config)
oc apply -k https://github.com/fjcloud/vllm-neuron-rosa/deploy/prereqs/instances

# 3. Deploy vLLM
oc new-project neuron-inference
oc apply -k https://github.com/fjcloud/vllm-neuron-rosa/deploy -n neuron-inference

# 4. Build the image (first time only, ~10 min)
oc start-build vllm-neuron -n neuron-inference --follow

# 5. Test
ROUTE=$(oc get route vllm-neuron -n neuron-inference -o jsonpath='{.spec.host}')
curl -sk "https://$ROUTE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/cache/model","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## What it does

The `entrypoint.sh` handles everything at startup:
1. Downloads the full HuggingFace repo to the PVC (including compiled artifacts)
2. Sets `NEURON_COMPILED_ARTIFACTS` to bypass NxDI's config hash
3. Launches vLLM pointing to the local model

## Repo structure

```
├── Dockerfile           # UBI 10 + Neuron SDK + vLLM 0.13
├── entrypoint.sh        # Download model, set artifacts path, run vLLM
└── deploy/
    ├── *.yaml           # BuildConfig, Deployment, PVC, Service, Route
    └── prereqs/
        ├── operators/   # NFD, KMM, Neuron operator subscriptions
        └── instances/   # NFD instance, Neuron DeviceConfig
```
