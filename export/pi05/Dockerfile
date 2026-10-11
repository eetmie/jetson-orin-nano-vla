# Immutable aarch64 NGC base includes the tested Blackwell PyTorch/CUDA build.
FROM nvcr.io/nvidia/pytorch@sha256:f3b4a33fd60d5e0358287bb44d57e91eda45e8f1681fef1bfd6c969371417130
ENV OPENPI_DATA_HOME=/cache/openpi HF_HOME=/cache/hf PIP_NO_CACHE_DIR=1
COPY export/pi05/constraints.txt export/pi05/requirements.txt /opt/pi05-locks/
RUN python -m pip install -c /opt/pi05-locks/constraints.txt pip==25.2 && \
    python -m pip install -r /opt/pi05-locks/requirements.txt
# Pin upstream modeling code; no implicit third_party checkout in the build context.
RUN mkdir -p /opt/openpi && \
    curl --fail --location https://codeload.github.com/Physical-Intelligence/openpi/tar.gz/c23745b5ad24e98f66967ea795a07b2588ed6c79 \
    | tar -xz -C /opt/openpi --strip-components=1
RUN TF_DIR="$(python -c 'import os,transformers; print(os.path.dirname(transformers.__file__))')" && \
    cp -r /opt/openpi/src/openpi/models_pytorch/transformers_replace/* "$TF_DIR/"
ENV PYTHONPATH=/opt/openpi/src:/opt/openpi/packages/openpi-client/src
WORKDIR /workspace
CMD ["/bin/bash"]
