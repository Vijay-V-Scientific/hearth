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
FROM isaacsim:5.0 AS base

# Non-root user for the container.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} hearth && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash hearth

WORKDIR /workspace

# Install Python deps first (better layer caching) so the source can change
# without rebuilding them.
COPY pyproject.toml requirements*.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source last so source edits don't bust the dependency layers.
COPY hearth ./hearth
COPY configs ./configs
COPY docker /workspace/docker

USER hearth

# Default: drop into a shell inside the project. Override command at runtime.
CMD ["/bin/bash"]
