#!/usr/bin/env bash
# Convenience wrapper for the Isaac Sim 5.x container.
#   ./docker/run.sh python -m hearth.scene.cli generate --seed 0 --out /tmp/scene0
#   ./docker/run.sh bash
set -euo pipefail

# Default to headless. Override with HEADLESS=0.
: "${HEADLESS:=1}"
export HEADLESS

# Build the image on first use.
if ! docker image inspect hearth:isaac5 >/dev/null 2>&1; then
  echo "Building hearth:isaac5 (first run)..."
  docker compose -f docker/docker-compose.yml build hearth
fi

exec docker run --rm --gpus all \
  -e HEADLESS="${HEADLESS}" \
  -e RUST_LOG=warn \
  -e XRES=1920 -e YRES=1080 \
  --ipc host \
  -e DISPLAY="${DISPLAY}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$PWD":/workspace/projects/hearth \
  hearth:isaac5 \
  "$@"
