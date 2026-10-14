#!/usr/bin/env bash
# PetBot demo（demo_project/recovery_agile_01）专用训练容器。与 recovery / agile / robot_lab 的容器彼此独立。
#
#   ./env.sh up        # 创建/启动容器 petbot-train（已存在则复用；含容器内 git 可用性检查）
#   ./env.sh install   # 容器内装当前 demo 的 agile 包（pip install --no-deps -e .）+ rsl-rl-lib 与上游补丁校验
#   ./env.sh verify    # 容器内跑 demo 自带的 CPU 测试（不启动 Isaac）
#   ./env.sh prepare   # 生成派生 MJCF + USD 资产（需要 GPU；写了 asset.json 才允许实例化任务）
#   ./env.sh status    # 镜像/容器状态
#   ./env.sh shell     # 进容器（工作目录 = demo 根）
#   ./env.sh train <args...>   # 容器内 python scripts/train.py <args>（任务 PetBot-Recovery-AGILE-v0）
#   ./env.sh run <cmd...>      # 在 demo 根目录执行任意命令
#   ./env.sh down      # 停止容器（不删除）
#   ./env.sh proxy start|stop|status   # 127.0.0.1:7890 直连代理（拉镜像用）
#
# 约定（来自 demo 的 PETBOT.md）：
#   * 容器必须导入**当前 demo**的 agile 包，而不是 reference_project/WBC-AGILE 那一份
#     —— 本脚本把 demo checkout 挂进容器并在其中做 editable 安装。
#   * 源模型从 robot_description/ **只读**加载：挂到 /workspace/robot_description 并设 PETBOT_MODEL_DIR。
#   * 基础镜像用 Isaac Lab 3.0.0-beta2（demo 派生自 WBC-AGILE 6830cf9，依赖版本一致）。
#
# 关键坑（2026-10-13 实测）：demo 的 .git 是 repo 的**相对**符号链接
#   `../../.repo/projects/demo_project/recovery_agile_01.git`，且 bare repo 没有 core.worktree,
#   而 demo 的 pre_learn 会调用 git（脏树检查 + 记录 commit）。因此容器内必须**复刻宿主相对层级**：
#   $PROJ/demo_project/recovery_agile_01 ↔ /workspace/demo_project/recovery_agile_01
#   $PROJ/.repo/projects/demo_project    ↔ /workspace/.repo/projects/demo_project
#   否则符号链接会解析到 /.repo/...，容器里 git 报 "not a git repository"，训练在 pre_learn 处直接失败。
#   另外 bare repo 的 objects/hooks/rr-cache 也是相对符号链接（→ ../../.repo/project-objects/petbot2_rl.git/...，
#   repo 工具的对象共享），必须一并挂载，否则 objects 悬空、git 同样判定 "not a git repository"。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根：向上找到含 default.xml 的目录。这样脚本无论放在 demo_project/demo_env/ 还是
# 版本化的 tools/petbot_trial/ 下都能算对（demo_project/runs/ 等路径都从这里派生）。
PROJ="$HERE"
while [ ! -f "$PROJ/default.xml" ] && [ "$PROJ" != "/" ]; do PROJ="$(dirname "$PROJ")"; done
[ -f "$PROJ/default.xml" ] || { echo "找不到仓库根（default.xml）" >&2; exit 1; }
SRC="$PROJ/demo_project/recovery_agile_01"
# 跑实验用固定 commit 的独立副本目录（每组一个 clone），见 demo_project/demo_env/launch_variants.sh
RUNS="$PROJ/demo_project/runs"
MODEL="$PROJ/robot_description"
GITDIR="$PROJ/.repo/projects/demo_project"
GITOBJECTS="$PROJ/.repo/project-objects"
CACHE="$HERE/cache"
BASE_IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2"
CT="petbot-train"
GPU="${PETBOT_GPU:-2,3}"
PY=/isaac-sim/python.sh
UV_LIB="rsl-rl-lib==5.4.1"
WORKDIR=/workspace/demo_project/recovery_agile_01

log() { printf '\033[1m[petbot]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[petbot] %s\033[0m\n' "$*" >&2; exit 1; }

common_args=(
  --gpus all
  --user root   # 3.0.0-beta2 镜像默认 ubuntu(1000)，挂载的 checkout 属主是宿主 anna(1028)
  --shm-size 32g
  -e "CUDA_VISIBLE_DEVICES=$GPU"
  -e ACCEPT_EULA=Y
  -e OMNI_KIT_ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e PYTHONUNBUFFERED=1
  -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  -e PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
  -e "PETBOT_MODEL_DIR=/workspace/robot_description"
  -v "$SRC":$WORKDIR
  -v "$RUNS":/workspace/runs
  -v "$GITDIR":/workspace/.repo/projects/demo_project
  -v "$GITOBJECTS":/workspace/.repo/project-objects
  -v "$MODEL":/workspace/robot_description:ro
  -v "$HERE":/workspace/demo_env
  -v "$CACHE/ov":/root/.cache/ov
  -v "$CACHE/nvidia":/root/.cache/nvidia
  -v "$CACHE/kit":/root/.cache/kit
  -v "$CACHE/ov-data":/root/.local/share/ov
  -w "$WORKDIR"
  --entrypoint bash
)

cmd_up() {
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "缺少 $BASE_IMAGE（用 recovery_env 的 parallel_pull.py 或 agile_env 的 fetch_missing.py 拉取）"
  [ -d "$SRC" ] || die "缺少 demo checkout：$SRC（先 repo sync demo_project/recovery_agile_01）"
  [ -d "$GITDIR" ] || die "缺少 bare repo：$GITDIR"
  [ -d "$GITOBJECTS" ] || die "缺少 repo 对象库：$GITOBJECTS"
  mkdir -p "$CACHE"/{ov,nvidia,kit,ov-data}
  if docker container inspect "$CT" >/dev/null 2>&1; then
    docker start "$CT" >/dev/null && log "$CT 已存在，已启动"
  else
    docker create --name "$CT" "${common_args[@]}" "$BASE_IMAGE" -c 'while :; do sleep 3600; done' >/dev/null
    docker start "$CT" >/dev/null && log "$CT 已创建并启动"
  fi
  # 容器以 root 运行、checkout 属主是宿主 anna(1028)：不加 safe.directory 时 git 会拒绝操作。
  docker exec "$CT" bash -c 'git config --global --add safe.directory "*"' >/dev/null 2>&1 || true
  if docker exec "$CT" bash -lc "cd $WORKDIR && git rev-parse --short HEAD" >/dev/null 2>&1; then
    log "$CT: 容器内 git 可用（pre_learn 的脏树检查能通过）"
  else
    log "警告：容器内 git 不可用，训练会在 pre_learn 处失败"
  fi
  cmd_status
}

cmd_install() {
  docker start "$CT" >/dev/null 2>&1 || true
  log "$CT: 装 demo 自己的 agile 包（--no-deps）+ $UV_LIB + 校验上游 rsl-rl 补丁"
  docker exec -i "$CT" bash -lc "
    set -e
    cd $WORKDIR
    $PY -m pip install --no-deps -e . 2>&1 | tail -3
    $PY -m pip install '$UV_LIB' 2>&1 | tail -2
    $PY scripts/verify_rsl_rl.py 2>&1 | tail -5
  "
}

cmd_verify() {
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec -i "$CT" bash -lc "cd $WORKDIR && $PY -m pytest -q --confcutdir=petbot_tests petbot_tests"
}

# 生成派生 MJCF + USD 资产（需要 GPU；失败时不写 asset.json，任务会拒绝实例化）。
cmd_prepare() {
  docker start "$CT" >/dev/null 2>&1 || true
  log "prepare --headless（GPU $GPU）：派生 MJCF → USD → 校验质量/接触/摩擦"
  docker exec -i "$CT" bash -lc "cd $WORKDIR && $PY -m agile.petbot.prepare --headless"
}

cmd_run() {
  docker start "$CT" >/dev/null 2>&1 || true
  docker exec -i "$CT" bash -lc "cd $WORKDIR && exec \"\$@\"" bash "$@"
}

cmd_status() {
  echo "--- 镜像 ---"; docker images | grep -E "REPOSITORY|3.0.0-beta2" || true
  echo "--- 容器 ---"; docker ps -a --filter "name=petbot-" --format '{{.Names}}\t{{.Image}}\t{{.Status}}' || true
}

cmd_proxy() {
  local pidfile="$CACHE/direct_proxy.pid"
  mkdir -p "$CACHE"
  case "${2:-status}" in
    start)
      pgrep -f "direct_proxy_7890[.]py" >/dev/null && { log "代理已在运行"; return; }
      nohup python3 "$HERE/direct_proxy_7890.py" >"$CACHE/direct_proxy.log" 2>&1 &
      echo $! > "$pidfile"; sleep 1
      pgrep -f "direct_proxy_7890[.]py" >/dev/null && log "已启动 → docker pull 可用" || die "启动失败" ;;
    stop)  pkill -f "direct_proxy_7890[.]py" && log "已停止" || log "本来就没在跑"; rm -f "$pidfile" ;;
    *) echo "usage: ./env.sh proxy start|stop" ;;
  esac
}

case "${1:-}" in
  up)      cmd_up ;;
  install) cmd_install ;;
  verify)  cmd_verify ;;
  prepare) cmd_prepare ;;
  status)  cmd_status ;;
  proxy)   cmd_proxy "$@" ;;
  run)     shift; cmd_run "$@" ;;
  train)   shift; [ "${1:-}" = "--" ] && shift; cmd_run "$PY" scripts/train.py "$@" ;;
  shell)   docker start "$CT" >/dev/null 2>&1 || true; exec docker exec -it "$CT" bash -l ;;
  down)    docker stop "$CT" >/dev/null 2>&1 || true; log "已停止 $CT" ;;
  *)       sed -n '2,25p' "${BASH_SOURCE[0]}" ;;
esac
