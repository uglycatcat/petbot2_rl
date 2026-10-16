#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$SRC/../.." && pwd)"
CT=petbot-upright-night
IMAGE=petbot-upright-runtime:20261016
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
 docker commit petbot-upright-train "$IMAGE"
fi
if ! docker inspect "$CT" >/dev/null 2>&1; then
 docker create --name "$CT" --gpus all --user root --shm-size 16g \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \
  -e PETBOT_MODEL_DIR=/workspace/robot_description \
  -v "$SRC":/workspace/demo_project/upright_rl_demo \
  -v "$ROOT/robot_description":/workspace/robot_description:ro \
  -v "$ROOT/demo_project/runs":/workspace/runs \
  -v "$ROOT/.repo":/workspace/.repo:ro \
  -w /workspace/demo_project/upright_rl_demo --entrypoint bash \
  "$IMAGE" -c 'while :; do sleep 3600; done'
fi
docker start "$CT"
docker exec "$CT" git config --global --add safe.directory '*'
