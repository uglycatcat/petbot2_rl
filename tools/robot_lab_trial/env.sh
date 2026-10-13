#!/usr/bin/env bash
# robot_lab v2.3.2 专用容器环境（与 recovery / agile 的容器各自独立，不共用命名与挂载）。
# 宿主机不跑 Isaac Sim、不训练；所有 python 都在容器内。
#
#   ./env.sh fetch      # 确认基础镜像存在（robot_lab v2.3.2 对应 Isaac Lab 2.3.2，本机已有）
#   ./env.sh up         # 创建/启动容器 robotlab-train（已存在则复用）
#   ./env.sh install    # 容器内 pip install -e source/robot_lab
#   ./env.sh verify     # 容器内自检：导入 robot_lab、任务注册、Go2W 环境能否创建
#   ./env.sh status     # 镜像/容器状态
#   ./env.sh shell      # 进容器
#   ./env.sh train --task ... --num_envs 4096 --headless    # 容器内跑上游 scripts/rsl_rl/train.py
#   ./env.sh run  <任意命令>                                  # 在容器的工作目录里执行命令
#   ./env.sh down       # 停止容器（不删除）
#   ./env.sh proxy start|stop|status   # 127.0.0.1:7890 直连代理（本机 dockerd 强制走该端口，拉镜像前需要）
#
# 说明：
#   * 官方 isaac-lab 镜像里没有 `python`，容器内统一用 /isaac-sim/python.sh。
#   * 基础镜像已带 isaaclab / isaaclab_rl / rsl-rl-lib / torch，无需联网装依赖（PIP_INDEX_URL 用清华源兜底）。
#   * 环境变量 ROBOTLAB_GPU（默认 2,3 —— 0/1 号 4090 被他人任务占用）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
SRC="$PROJ/reference_project/robot_lab"
CACHE="$HERE/cache"
BASE_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.2"
CT="robotlab-train"
GPU="${ROBOTLAB_GPU:-2,3}"
PY=/isaac-sim/python.sh
UPSTREAM_TRAIN="scripts/rsl_rl/train.py"

log() { printf '\033[1m[robotlab]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[robotlab] %s\033[0m\n' "$*" >&2; exit 1; }

common_args=(
  --gpus all
  --shm-size 32g
  -e "CUDA_VISIBLE_DEVICES=$GPU"
  -e ACCEPT_EULA=Y
  -e OMNI_KIT_ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e PYTHONUNBUFFERED=1
  -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
  -v "$SRC":/workspace/robot_lab
  -v "$HERE":/workspace/robot_lab_env:ro
  -v "$CACHE/ov":/root/.cache/ov
  -v "$CACHE/nvidia":/root/.cache/nvidia
  -v "$CACHE/kit":/root/.cache/kit
  -v "$CACHE/ov-data":/root/.local/share/ov
  -w /workspace/robot_lab
  --entrypoint bash
)

cmd_fetch() {
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && log "基础镜像已存在：$BASE_IMAGE" && return
  die "缺少 $BASE_IMAGE（本机应有）；如需从镜像站拉取：./env.sh proxy start 后用 docker pull，或参考 recovery_env 的 parallel_pull.py"
}

cmd_up() {
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "先执行 ./env.sh fetch"
  mkdir -p "$CACHE"/{ov,nvidia,kit,ov-data}
  if docker container inspect "$CT" >/dev/null 2>&1; then
    docker start "$CT" >/dev/null && log "$CT 已存在，已启动"
  else
    docker create --name "$CT" "${common_args[@]}" "$BASE_IMAGE" -c 'while :; do sleep 3600; done' >/dev/null
    docker start "$CT" >/dev/null && log "$CT 已创建并启动"
  fi
  cmd_status
}

cmd_install() {
  docker start "$CT" >/dev/null 2>&1 || true
  log "$CT: pip install -e source/robot_lab"
  docker exec "$CT" bash -lc "cd /workspace/robot_lab/source/robot_lab && $PY -m pip install -e ." 2>&1 | tail -5
}

cmd_verify() {
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec -i "$CT" "$PY" /workspace/robot_lab_env/verify.py
}

cmd_run() {  # 在容器工作目录执行任意命令
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec -i "$CT" bash -lc "cd /workspace/robot_lab && exec \"\$@\"" bash "$@"
}

cmd_status() {
  echo "--- 镜像 ---"; docker images | grep -E "REPOSITORY|isaac-lab" || true
  echo "--- 容器 ---"; docker ps -a --filter "name=robotlab-" --format '{{.Names}}\t{{.Status}}' || true
}

cmd_proxy() {
  local pidfile="$CACHE/direct_proxy.pid"
  mkdir -p "$CACHE"
  case "${2:-status}" in
    start)
      if pgrep -f "direct_proxy_7890[.]py" >/dev/null; then log "代理已在运行"; return; fi
      nohup python3 "$HERE/direct_proxy_7890.py" >"$CACHE/direct_proxy.log" 2>&1 &
      echo $! > "$pidfile"; sleep 1
      pgrep -f "direct_proxy_7890[.]py" >/dev/null && log "已启动 → docker pull 可用" || die "启动失败，见 $CACHE/direct_proxy.log" ;;
    stop)  pkill -f "direct_proxy_7890[.]py" && log "已停止" || log "本来就没在跑"; rm -f "$pidfile" ;;
    status) pgrep -f "direct_proxy_7890[.]py" >/dev/null && echo "代理：运行中（127.0.0.1:7890）" || echo "代理：未运行" ;;
    *) die "proxy start|stop|status" ;;
  esac
}

case "${1:-}" in
  fetch)   cmd_fetch ;;
  up)      cmd_up ;;
  install) cmd_install ;;
  verify)  cmd_verify ;;
  status)  cmd_status ;;
  proxy)   cmd_proxy "$@" ;;
  run)     shift; cmd_run "$@" ;;
  train)   shift; [ "${1:-}" = "--" ] && shift; cmd_run "$PY" "$UPSTREAM_TRAIN" "$@" ;;
  shell)   docker start "$CT" >/dev/null 2>&1 || true; exec docker exec -it "$CT" bash -l ;;
  down)    docker stop "$CT" >/dev/null 2>&1 || true; log "已停止 $CT" ;;
  *)       sed -n '2,22p' "${BASH_SOURCE[0]}" ;;
esac
