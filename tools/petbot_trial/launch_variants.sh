#!/usr/bin/env bash
# PetBot 首版并行多组训练：每组一个固定 commit 的独立副本（demo_project/runs/<变体>），
# 在容器 petbot-train 内以 PYTHONPATH 覆盖主 checkout 的 editable 安装。
#
#   变体按卡分配：GPU3 最多 5 组、GPU2 最多 3 组（GPU2 上还有他人的 AGILE 训练，必须留显存余量）。
#   日志一律写在 demo_project/runs/_logs/<变体>.log —— 写进副本会让 pre_learn 的脏树检查拒绝启动。
#
#   ./launch_variants.sh start     # 按卡位与并发上限依次起全部变体（后台：terminal background / setsid）
#   ./launch_variants.sh status    # 逐组：卡位、是否在跑、最新 iteration、iteration time
#   ./launch_variants.sh log <v>   # 尾随某组日志
#   ./launch_variants.sh stop      # 停掉全部变体（不动容器）
#
# 环境变量：PETBOT_MAX_ITERS（默认 3000）、PETBOT_CAP_GPU3（5）、PETBOT_CAP_GPU2（3）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根：向上找到含 default.xml 的目录（脚本放在 demo_project/demo_env/ 或 tools/petbot_trial/ 都能用）。
PROJ="$HERE"
while [ ! -f "$PROJ/default.xml" ] && [ "$PROJ" != "/" ]; do PROJ="$(dirname "$PROJ")"; done
RUNS="$PROJ/demo_project/runs"
LOGDIR="$RUNS/_logs"
CT="petbot-train"
PY=/isaac-sim/python.sh
MAX_ITERS="${PETBOT_MAX_ITERS:-3000}"
CAP3="${PETBOT_CAP_GPU3:-6}"
CAP2="${PETBOT_CAP_GPU2:-3}"
BASE_ARGS=(--task PetBot-Recovery-AGILE-v0 --headless --num_envs 2048 --max_iterations "$MAX_ITERS" --logger tensorboard)

# 顺序即优先级：前 5 组占 GPU3、随后 3 组占 GPU2，其余排队。
ORDER=(base_s42 base_s43 gains2x_s42 gains_down_s42 decimation5_s42 \
       reward_still_s42 reward_config_heavy_s42 episode15_s42 \
       spawn1m_s42 ppo64_s42 ppo_explore_s42)
declare -A GPU_OF=(
  [base_s42]=3 [base_s43]=3 [gains2x_s42]=3 [gains_down_s42]=3 [decimation5_s42]=3
  [reward_still_s42]=2 [reward_config_heavy_s42]=3 [episode15_s42]=2
  [spawn1m_s42]=3 [ppo64_s42]=2 [ppo_explore_s42]=3
)

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# 变体的额外参数。凡是 __post_init__ 里赋值的字段（decimation / episode_length_s / 增益）都用
# 代码副本，不用 CLI 覆盖——那些值可能在覆盖之后才被 post_init 重新赋值。
extra_args() {
  case "$1" in
    base_s42|base_s43|gains2x_s42|gains_down_s42|decimation5_s42|episode15_s42)
      echo "--seed ${1##*_s}" ;;
    reward_still_s42|reward_config_heavy_s42)
      echo "--seed 42" ;;
    spawn1m_s42)
      echo "--seed 42 agent.petbot_collection.spawn_height=1.0" ;;
    ppo64_s42)
      echo "--seed 42 agent.num_steps_per_env=64 agent.algorithm.learning_rate=2e-4" ;;
    ppo_explore_s42)
      echo "--seed 42 agent.algorithm.entropy_coef=0.02 agent.policy.init_noise_std=1.0" ;;
    *) echo "" ;;
  esac
}

# 输出 "变体 GPU"
running_pairs() { docker exec "$CT" bash /workspace/demo_env/runs_probe.sh 2>/dev/null; }

count_on_gpu() { running_pairs | awk -v g="$1" '$2==g' | wc -l; }
is_running() { running_pairs | awk -v v="$1" '$1==v' | grep -q . ; }

launch() {
  local v="$1" gpu="$2" args
  args="$(extra_args "$v")"
  mkdir -p "$LOGDIR"
  log "启动 $v（GPU$gpu, $MAX_ITERS iter）→ demo_project/runs/_logs/$v.log"
  docker exec -d "$CT" bash -lc "cd /workspace/runs/$v && PYTHONPATH=/workspace/runs/$v CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY /workspace/runs/$v/scripts/train.py ${BASE_ARGS[*]} $args > /workspace/runs/_logs/$v.log 2>&1 < /dev/null"
}

cmd_start() {
  # 原子锁：避免两个调度器同时铺同一批变体。
  mkdir -p "$RUNS"
  if ! mkdir "$RUNS/.launcher.lock" 2>/dev/null; then
    log "已有调度器在跑（$RUNS/.launcher.lock），退出"; exit 0
  fi
  trap 'rmdir "$RUNS/.launcher.lock" 2>/dev/null' EXIT
  local pending=("${ORDER[@]}")
  while [ "${#pending[@]}" -gt 0 ] || [ -n "$(running_pairs)" ]; do
    local next=()
    for v in "${pending[@]}"; do
      is_running "$v" && continue                     # 已在跑
      local gpu="${GPU_OF[$v]}" cap
      [ "$gpu" = 3 ] && cap="$CAP3" || cap="$CAP2"
      if [ "$(count_on_gpu "$gpu")" -lt "$cap" ]; then launch "$v" "$gpu"; else next+=("$v"); fi
    done
    pending=("${next[@]}")
    [ "${#pending[@]}" -gt 0 ] && { log "卡位已满，待起：${pending[*]}"; sleep 300; }
  done
  log "全部变体结束"
  cmd_status
}

cmd_status() {
  local pairs; pairs="$(running_pairs)"
  printf '%-24s %-5s %-8s %-24s %s\n' 变体 GPU 状态 最新iteration iteration_time
  for v in "${ORDER[@]}"; do
    local state=已结束
    printf '%s\n' "$pairs" | awk -v v="$v" '$1==v' | grep -q . && state=运行中
    local it t
    it="$(grep -oE 'Learning iteration [0-9]+/[0-9]+' "$LOGDIR/$v.log" 2>/dev/null | tail -1)"
    t="$(grep -oE 'Iteration time: [0-9.]+s' "$LOGDIR/$v.log" 2>/dev/null | tail -1)"
    printf '%-24s %-5s %-8s %-24s %s\n' "$v" "${GPU_OF[$v]}" "$state" "${it:-—}" "${t:-—}"
  done
  echo; echo "--- 容器内进程 ---"; docker exec "$CT" bash -lc 'pgrep -af "scripts/train[.]py" | sed "s|/workspace/runs/||" | cut -c1-110' 2>/dev/null || true
  echo "--- GPU ---"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
}

case "${1:-status}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  log)    tail -f "$LOGDIR/${2:?变体名}.log" ;;
  stop)   docker exec "$CT" bash -lc 'pkill -f "scripts/train[.]py"' && log "已停止全部变体" || log "没有在跑的变体" ;;
  *)      sed -n '2,20p' "${BASH_SOURCE[0]}" ;;
esac
