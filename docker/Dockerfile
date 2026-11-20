# syntax=docker/dockerfile:1
#===========
# Project HEARTH — Isaac Sim 5.x training/eval container
#
# This image provides the heavy runtime (Isaac Sim 5.x + PhysX GPU) that the
# lightweight, importable code in hearth/ is designed to plug into. The
# `hearth` package itself never imports Isaac Sim at module load time; every
# runtime entry point raises NotImplementedError until this container (or any
# GPU + Isaac Sim 5.x host) is present.
#
# Build:
#   docker build -f docker/Dockerfile -t hearth:isaac5 .
# Run (GPU):
#   docker run --rm --gpus all -v "$PWD":/workspace/projects/hearth \
#       -e HEADLESS=1 hearth:isaac5 \
#       python -m hearth.scene.cli generate --seed 0 --out /tmp/scene0
#
# Requirements:
#   * NVIDIA driver >= 525 (for CUDA 12.x)
#   * NVIDIA Container Toolkit installed and `nvidia-smi` works
#   * Isaac Sim 5.x base image available locally or pullable
#===========
FROM nvcr.io/nvidia/isaac-sim:5.1.0 AS base

# Non-root user for the container.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} hearth && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash hearth

WORKDIR /workspace

# Copy the declaration AND the package source before installing: an editable
# install (`-e .`) builds against the package, so the source tree must already
# be present when pip runs. The old order ran `pip install -e .` before copying
# hearth/, so the install could not see the package it was installing.
COPY pyproject.toml requirements*.txt ./
COPY hearth ./hearth
COPY configs ./configs
COPY docker /workspace/docker
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

USER hearth

# Default: drop into a shell inside the project. Override command at runtime.
CMD ["/bin/bash"]
