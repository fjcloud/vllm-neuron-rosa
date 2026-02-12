# vLLM + AWS Neuron on Red Hat UBI 10
# Supports: Llama 2/3.x/4, Qwen 2.5, Qwen 3, Mistral
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest

ARG PYTHON_VERSION=3.12
ARG VLLM_VERSION=0.13.0
ARG VLLM_NEURON_VERSION=release-0.3.0

USER root
WORKDIR /opt

# ── Neuron YUM repo (RPM packages for runtime) ──────────────
RUN rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB && \
    printf '[neuron]\nname=Neuron YUM Repository\nbaseurl=https://yum.repos.neuron.amazonaws.com\nenabled=1\nmetadata_expire=0\ngpgcheck=0\n' \
    > /etc/yum.repos.d/neuron.repo

# ── System dependencies + Neuron runtime + collectives ───────
RUN microdnf install -y --nodocs \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-pip \
        python${PYTHON_VERSION}-devel \
        gcc gcc-c++ make cmake git \
        openssl-devel \
        libatomic \
        hwloc-devel \
    && microdnf install -y --nodocs \
        aws-neuronx-runtime-lib \
        aws-neuronx-collectives \
    && microdnf clean all

# ── Register Neuron libs with the dynamic linker ────────────
RUN echo "/opt/aws/neuron/lib" > /etc/ld.so.conf.d/neuron.conf && \
    ldconfig && \
    ldconfig -p | grep nrt

# ── Add Neuron tools to PATH ────────────────────────────────
ENV PATH="/opt/aws/neuron/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/aws/neuron/lib:${LD_LIBRARY_PATH}"

# ── Python venv ──────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/vllm
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Neuron SDK (torch-neuronx + compiler + NxDI) ────────────
RUN pip install --no-cache-dir \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        torch-neuronx \
        neuronx-cc \
        neuronx-distributed-inference>=0.7 \
        libneuronxla

# ── vLLM (matches vllm-neuron plugin requirement) ───────────
RUN pip install --no-cache-dir \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        "vllm==${VLLM_VERSION}"

# ── vllm-neuron plugin (Qwen3, Llama4, etc.) ────────────────
RUN pip install --no-cache-dir \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        "git+https://github.com/vllm-project/vllm-neuron.git@${VLLM_NEURON_VERSION}"

# ── OpenShift compatibility (arbitrary UID in group 0) ───────
RUN mkdir -p /home/vllm/.cache /tmp/neuron_cache && \
    useradd --uid 2000 --gid 0 --home-dir /home/vllm vllm && \
    chmod -R g+rwx /home/vllm /tmp/neuron_cache && \
    chgrp -R 0 /home/vllm /tmp/neuron_cache

ENV HOME=/home/vllm \
    HF_HOME=/home/vllm/.cache \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_USE_V1=1

COPY entrypoint.sh /opt/entrypoint.sh

WORKDIR /home/vllm
USER 2000
EXPOSE 8000

ENTRYPOINT ["/opt/entrypoint.sh"]
