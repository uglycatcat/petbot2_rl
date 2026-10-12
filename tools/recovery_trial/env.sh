#!/usr/bin/env bash
# Kitjesen/recovery 的专用容器环境（训练容器 + 测试观察容器）。
# 宿主机不跑 Isaac Sim、不训练；所有 python 都在容器内。
#
#   ./env.sh fetch      # 拉镜像（两种方式：先 proxy start 让 docker pull 可用，或用本目录多连接拉取器）
#   ./env.sh up         # 创建/启动两个容器（已存在则复用）
#   ./env.sh install    # 容器内按上游 README 装 thunder_recovery（pip install -e .）
#   ./env.sh proxy start|stop|status   # 在 127.0.0.1:7890 放直连代理，让 docker pull 可用（本机 dockerd 强制走该端口）
#   ./env.sh verify [train|vis] [--skip-app]   # 容器内自检（版本、任务注册、依赖缺口），不训练
#   ./env.sh status     # 镜像/容器状态
#   ./env.sh train --task ... --num_envs 4096 --headless          # 训练容器内跑上游 scripts/train.py
#   ./env.sh vis --task ... --num_envs 64 --headless --enable_cameras --video   # 观察容器内带录制跑
#   ./env.sh shell train|vis    # 进容器
#   ./env.sh down       # 停止两个容器（不删除）
#
# 说明：
#   * 官方 isaac-lab 镜像里没有 `python`，容器内统一用 /isaac-sim/python.sh（Isaac Sim 自带 py3.11）。
#   * 镜像已带 isaaclab 0.54.2 / isaaclab_rl 0.4.7 / rsl-rl-lib 3.1.2 / torch 2.7.0+cu128，
#     所以不需要派生镜像，也不需要联网装依赖；PIP_INDEX_URL 只作为兜底（本机 pypi.org 不可达）。
#   * 环境变量 RECOVERY_GPU（默认 2,3 —— 0/1 号 4090 已被其他任务占用）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
SRC="$PROJ/reference_project/recovery"        # 上游 checkout（被 .gitignore 覆盖）
CACHE="$HERE/cache"                            # Isaac Sim 运行时缓存，留在项目内，不写 ~
IMAGES="$HERE/_images"
BASE_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.2"
# 镜像源：本机直连 nvcr.io 实测 ~0.7 MB/s；daocloud 的 NGC 镜像多连接聚合 ~10 MB/s，
# 且 amd64 manifest digest 与 nvcr.io 完全一致（f07c37e3...）。拉完 retag 成 BASE_IMAGE。
PULL_REF="${RECOVERY_PULL_REF:-nvcr.m.daocloud.io/nvidia/isaac-lab:2.3.2}"
TRAIN_CT="recovery-train"
VIS_CT="recovery-vis"
GPU="${RECOVERY_GPU:-2,3}"
PY=/isaac-sim/python.sh

log() { printf '\033[1m[env]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[env] %s\033[0m\n' "$*" >&2; exit 1; }

# 两个容器共用：全部 4090 可见，具体用哪张由 CUDA_VISIBLE_DEVICES 运行时决定
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
  -v "$SRC":/workspace/recovery
  -v "$PROJ/reference_project/recovery_variants":/workspace/variants
  -v "$HERE":/workspace/recovery_env:ro
  -v "$CACHE/ov":/root/.cache/ov
  -v "$CACHE/nvidia":/root/.cache/nvidia
  -v "$CACHE/kit":/root/.cache/kit
  -v "$CACHE/ov-data":/root/.local/share/ov
  -w /workspace/recovery
  --entrypoint bash
)

cmd_fetch() {
  if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    log "基础镜像已存在：$BASE_IMAGE"; return
  fi
  mkdir -p "$IMAGES"
  log "多连接拉取 $PULL_REF（约 8.4 GB，实测聚合 ~5 MB/s，约 30 分钟）"
  python3 "$HERE/parallel_pull.py" "$PULL_REF" "$IMAGES/isaac-lab-2.3.2.tar" -j 8
  log "docker load"
  docker load -i "$IMAGES/isaac-lab-2.3.2.tar" | tee "$IMAGES/load.log"
  docker tag "$PULL_REF" "$BASE_IMAGE"
  log "完成，可删除 $IMAGES/isaac-lab-2.3.2.tar 回收 8.4 GB"
}

cmd_up() {
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "先执行 ./env.sh fetch"
  mkdir -p "$CACHE"/{ov,nvidia,kit,ov-data}
  for ct in "$TRAIN_CT" "$VIS_CT"; do
    if docker container inspect "$ct" >/dev/null 2>&1; then
      docker start "$ct" >/dev/null && log "$ct 已存在，已启动"
    else
      docker create --name "$ct" "${common_args[@]}" "$BASE_IMAGE" -c 'while :; do sleep 3600; done' >/dev/null
      docker start "$ct" >/dev/null && log "$ct 已创建并启动"
    fi
  done
  cmd_status
}

cmd_install() {
  # 上游步骤：容器内 pip install -e .（在挂载的上游 checkout 里）
  for ct in "$TRAIN_CT" "$VIS_CT"; do
    log "$ct: pip install -e ."
    docker exec "$ct" bash -lc "cd /workspace/recovery && $PY -m pip install -e ." 2>&1 | tail -3
  done
}

cmd_run() {  # cmd_run <容器> <上游参数...>
  local ct="$1"; shift
  docker start "$ct" >/dev/null 2>&1 || true
  docker exec -i "$ct" bash -lc "cd /workspace/recovery && exec $PY scripts/train.py \"\$@\"" env "$@"
}

cmd_status() {
  echo "--- 镜像 ---"; docker images | grep -E "REPOSITORY|isaac-lab" || true
  echo "--- 容器 ---"; docker ps -a --filter "name=recovery-" --format '{{.Names}}\t{{.Status}}' || true
}

cmd_proxy() {  # 本机 dockerd 强制走 127.0.0.1:7890；在该端口放直连隧道代理，docker pull 即可用
  local pidfile="$HERE/cache/direct_proxy.pid"
  mkdir -p "$CACHE"
  case "${2:-status}" in
    start)
      if pgrep -f "direct_proxy_7890.py" >/dev/null; then log "代理已在运行"; return; fi
      nohup python3 "$HERE/direct_proxy_7890.py" >"$CACHE/direct_proxy.log" 2>&1 &
      echo $! > "$pidfile"
      sleep 1
      pgrep -f "direct_proxy_7890.py" >/dev/null && log "已启动（日志 $CACHE/direct_proxy.log）→ docker pull 可用" \
        || die "启动失败，见 $CACHE/direct_proxy.log" ;;
    stop)
      pkill -f "direct_proxy_7890.py" && log "已停止（docker pull 会再次失败）" || log "本来就没在跑"
      rm -f "$pidfile" ;;
    status)
      pgrep -f "direct_proxy_7890.py" >/dev/null && echo "代理：运行中（127.0.0.1:7890）" || echo "代理：未运行（docker pull 会失败）"
      ss -ltn 2>/dev/null | grep -w 7890 || true ;;
    *) die "proxy start|stop|status" ;;
  esac
}

case "${1:-}" in
  fetch)   cmd_fetch ;;
  up)      cmd_up ;;
  install) cmd_install ;;
  status)  cmd_status ;;
  proxy)   cmd_proxy "$@" ;;
  verify)  shift; ct="$TRAIN_CT"
           if [ "${1:-}" = "train" ] || [ "${1:-}" = "vis" ]; then [ "$1" = "vis" ] && ct="$VIS_CT"; shift; fi
           docker start "$ct" >/dev/null 2>&1 || true
           docker exec -i "$ct" "$PY" /workspace/recovery_env/verify.py "$@" ;;
  down)    docker stop "$TRAIN_CT" "$VIS_CT" >/dev/null 2>&1 || true; log "已停止" ;;
  train)   shift; [ "${1:-}" = "--" ] && shift; cmd_run "$TRAIN_CT" "$@" ;;
  vis)     shift; [ "${1:-}" = "--" ] && shift; cmd_run "$VIS_CT" "$@" ;;
  shell)   case "${2:-train}" in train) ct="$TRAIN_CT";; vis) ct="$VIS_CT";; *) die "shell train|vis";; esac
           docker start "$ct" >/dev/null 2>&1 || true
           exec docker exec -it "$ct" bash -l ;;
  *)       sed -n '2,22p' "${BASH_SOURCE[0]}" ;;
esac
