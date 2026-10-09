# Vendored from the author's spark-projects: pi05-spark-inference/docker/Dockerfile (build context: a checkout
# with openpi under third_party/openpi).
# π0.5 inference on DGX Spark (GB10 / Blackwell / aarch64)
# Base: NVIDIA NGC PyTorch for ARM with CUDA 13 — matches the Jetson Thor recipe.
# The openpi PyTorch env is baked in (verified working on GB10):
#   - keep the base image's Blackwell torch (do NOT install openpi's torch==2.7.1)
#   - CPU-only JAX (PyTorch inference path uses jax only for tree utils + orbax load)
#   - overlay openpi's patched transformers files
#
# Build context is the REPO ROOT (see build.sh) so we can COPY third_party/openpi.
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive \
    OPENPI_DATA_HOME=/cache/openpi \
    HF_HOME=/cache/hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# gcloud/gsutil for gs:// checkpoint download (openpi assets live on GCS).
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt && \
    ln -sf /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil

# openpi runtime deps — NOTE: no torch (keep base image's Blackwell build), CPU jax only.
# This is the full closure needed for the conversion + inference paths (NOT training:
# lerobot/wandb/gym-aloha/polars are deliberately omitted — they pin torch and aren't
# used for inference; the benchmark builds the Policy directly to avoid importing them).
RUN pip install \
        "jax==0.5.3" \
        "flax==0.10.2" \
        "orbax-checkpoint==0.11.13" \
        "jaxtyping==0.2.36" \
        "transformers==4.53.2" \
        "tyro>=0.9.5" \
        "numpy>=1.22.4,<2.0.0" \
        "numpydantic>=1.6.6" \
        "augmax" "einops" "sentencepiece" "pillow" "opencv-python-headless" \
        "beartype==0.19.0" "chex==0.1.90" "equinox>=0.11.8" "dm-tree>=0.1.8" \
        "ml_collections==1.0.0" "treescope>=0.1.7" "filelock>=3.16.1" \
        "tqdm-loggable>=0.2" "rich>=14.0.0" "imageio>=2.36.1" \
        "flatbuffers>=24.3.25" "fsspec[gcs]>=2024.6.0"

# Bake openpi into the image at /opt/openpi (source copied from build context).
COPY third_party/openpi /opt/openpi
RUN TF_DIR="$(python -c 'import os,transformers; print(os.path.dirname(transformers.__file__))')" && \
    cp -r /opt/openpi/src/openpi/models_pytorch/transformers_replace/* "$TF_DIR/" && \
    pip install --no-deps -e /opt/openpi/packages/openpi-client && \
    pip install --no-deps -e /opt/openpi

WORKDIR /workspace
CMD ["/bin/bash"]
