#!/usr/bin/env bash
# WBC-AGILE（StandUp-T1-v0）专用容器环境（与 recovery / robot_lab 的容器各自独立，不共用命名与挂载）。
# 宿主机不跑 Isaac Sim、不训练；所有 python 都在容器内。
#
#   ./env.sh fetch        # 拉基础镜像 nvcr.io/nvidia/isaac-lab:3.0.0-beta2（约 16 GB 压缩，走镜像站多连接）
#   ./env.sh up           # 创建/启动容器 agile-train（已存在则复用）
#   ./env.sh install      # 容器内按上游方式装依赖：uv sync --frozen（锁在 uv.lock）
#   ./env.sh install-lite # 备选：pip install -e . --no-deps + 只补非 isaaclab 的小依赖（镜像已带 isaaclab/torch）
#   ./env.sh verify       # 容器内自检：版本、任务注册、StandUp-T1-v0 环境能否创建
#   ./env.sh status       # 镜像/容器状态
#   ./env.sh shell        # 进容器
#   ./env.sh train --task StandUp-T1-v0 --num_envs 4096 --headless
#   ./env.sh run <任意命令>          # 在 /workspace/agile 下执行
#   ./env.sh down         # 停止容器（不删除）
#   ./env.sh proxy start|stop|status # 127.0.0.1:7890 直连代理（本机 dockerd 强制走该端口）
#
# 说明：
#   * 上游要求 Isaac Lab 3.0.0-beta2 / Isaac Sim 6.0 / Python 3.12 / rsl-rl-lib 5.4.1（见 pyproject.toml 与 uv.lock）。
#   * 官方 isaac-lab 镜像里没有 `python`：容器内统一用 /isaac-sim/python.sh；uv 需在容器内装（pip install uv）。
#   * 环境变量 AGILE_GPU（默认 2,3 —— 0/1 号 4090 被他人任务占用）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
SRC="$PROJ/reference_project/WBC-AGILE"
CACHE="$HERE/cache"
IMAGES="$HERE/_images"
BASE_IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2"
# 本机直连 nvcr.io 实测 ~0.7 MB/s；daocloud 的 NGC 镜像多连接聚合明显更快（与 2.3.2 同样的做法）。
PULL_REF="${AGILE_PULL_REF:-nvcr.m.daocloud.io/nvidia/isaac-lab:3.0.0-beta2}"
TAR="$IMAGES/isaac-lab-3.0.0-beta2.tar"
CT="agile-train"
GPU="${AGILE_GPU:-2,3}"
PY=/isaac-sim/python.sh
TRAIN="scripts/train.py"
# uv 走清华源；项目环境放在挂载的 checkout 之外（$HERE/.venv 落在 /data）
UV_ENV=( -e "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple" -e "UV_HTTP_TIMEOUT=120" -e "UV_PROJECT_ENVIRONMENT=/workspace/agile_env/.venv" )

log() { printf '\033[1m[agile]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[agile] %s\033[0m\n' "$*" >&2; exit 1; }

common_args=(
  --gpus all
  --user root   # 3.0.0-beta2 镜像默认以 ubuntu(1000) 运行，挂载的 checkout 属主是宿主 anna(1028) → 以 root 运行（与上游 Dockerfile 的 USER root 一致）
  --shm-size 32g
  -e "CUDA_VISIBLE_DEVICES=$GPU"
  -e ACCEPT_EULA=Y
  -e OMNI_KIT_ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e PYTHONUNBUFFERED=1
  -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
  "${UV_ENV[@]}"
  -v "$SRC":/workspace/agile
  -v "$HERE":/workspace/agile_env
  -v "$CACHE/ov":/root/.cache/ov
  -v "$CACHE/nvidia":/root/.cache/nvidia
  -v "$CACHE/kit":/root/.cache/kit
  -v "$CACHE/ov-data":/root/.local/share/ov
  -w /workspace/agile
  --entrypoint bash
)

cmd_fetch() {
  if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then log "基础镜像已存在：$BASE_IMAGE"; return; fi
  mkdir -p "$IMAGES"
  log "多连接拉取 $PULL_REF（压缩 ~16 GB，视镜像站速率约 1~3 小时）"
  python3 "$HERE/parallel_pull.py" "$PULL_REF" "$TAR" -j 8
  log "docker load"
  docker load -i "$TAR" | tee "$IMAGES/load.log"
  docker tag "$PULL_REF" "$BASE_IMAGE"
  log "完成；可删除 $TAR 回收空间"
}

cmd_up() {
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "先执行 ./env.sh fetch"
  mkdir -p "$CACHE"/{ov,nvidia,kit,ov-data} "$HERE/.venv"
  if docker container inspect "$CT" >/dev/null 2>&1; then
    docker start "$CT" >/dev/null && log "$CT 已存在，已启动"
  else
    docker create --name "$CT" "${common_args[@]}" "$BASE_IMAGE" -c 'while :; do sleep 3600; done' >/dev/null
    docker start "$CT" >/dev/null && log "$CT 已创建并启动"
  fi
  cmd_status
}

cmd_ensure_uv() {
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec "$CT" bash -lc "$PY -m pip show uv >/dev/null 2>&1 || $PY -m pip install -q uv" 2>&1 | tail -2
}

cmd_install() {   # 上游方式：uv sync --frozen（按 uv.lock）
  cmd_ensure_uv
  log "$CT: uv sync --frozen --no-dev（按 uv.lock 装 isaaclab 3.0.0b2 / torch / rsl-rl-lib 5.4.1）"
  docker exec -i "$CT" bash -lc "cd /workspace/agile && uv sync --frozen --no-dev 2>&1 | tail -20"
  log "校验上游的 rsl-rl 兼容补丁"
  docker exec -i "$CT" bash -lc "cd /workspace/agile && uv run --frozen scripts/verify_rsl_rl.py 2>&1 | tail -10"
}

cmd_install_lite() {  # 备选：不重装 isaaclab/torch，只装 AGILE 包与其小依赖
  docker start "$CT" >/dev/null 2>&1 || true
  log "$CT: pip install -e . --no-deps + 补齐非 isaaclab 依赖"
  docker exec -i "$CT" bash -lc "
    set -e
    cd /workspace/agile
    $PY -m pip install -e . --no-deps
    $PY -m pip install 'rsl-rl-lib==5.4.1' tensordict wandb tqdm pyyaml matplotlib pandas mujoco 'leapp==0.5.2'
  " 2>&1 | tail -15
}

cmd_verify() {
  docker start "$CT" >/dev/null 2>&1 || true
  local py="$PY"
  [ "${1:-}" = "--uv" ] && py="uv run --frozen python"
  docker exec -i "$CT" bash -lc "cd /workspace/agile && exec $py /workspace/agile_env/verify.py"
}

cmd_run() {
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec -i "$CT" bash -lc "cd /workspace/agile && exec \"\$@\"" bash "$@"
}

cmd_status() {
  echo "--- 镜像 ---"; docker images | grep -E "REPOSITORY|isaac-lab" || true
  echo "--- 容器 ---"; docker ps -a --filter "name=agile-" --format '{{.Names}}\t{{.Status}}' || true
  [ -f "$TAR" ] && echo "--- 下载中 ---" && ls -lh "$TAR" && tail -1 "$IMAGES/pull.log" 2>/dev/null || true
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
  fetch)        cmd_fetch ;;
  up)           cmd_up ;;
  install)      cmd_install ;;
  install-lite) cmd_install_lite ;;
  verify)       shift; cmd_verify "$@" ;;
  status)       cmd_status ;;
  proxy)        cmd_proxy "$@" ;;
  run)          shift; cmd_run "$@" ;;
  train)        shift; [ "${1:-}" = "--" ] && shift; cmd_run $PY "$TRAIN" "$@" ;;
  shell)        docker start "$CT" >/dev/null 2>&1 || true; exec docker exec -it "$CT" bash -l ;;
  down)         docker stop "$CT" >/dev/null 2>&1 || true; log "已停止 $CT" ;;
  *)            sed -n '2,22p' "${BASH_SOURCE[0]}" ;;
esac
