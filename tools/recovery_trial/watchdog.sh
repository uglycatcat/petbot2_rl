#!/usr/bin/env bash
# 六个实验的看门狗（在宿主机上跑，不依赖 Hermes 会话）：
#   每 15 分钟：检查每个实验的进程是否还在；若已死且未跑满 10000 迭代，
#              用该实验最新的检查点自动续训（避免整晚白等）。
#   每 2 小时：对 paper_v1 做一次论文口径评估，追成功率曲线。
#   每 tick  ：记录一行进度（迭代数 / GPU / 负载）。
#   ./watchdog.sh start | stop | status | log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGF=$HERE/watchdog.log
PIDF=$HERE/.watchdog.pid
MAX_ITER=10000
EVAL_EVERY=8          # 每 8 个 tick（=2 小时）评估一次 paper_v1
TICK=900              # 秒

# 名称:容器:容器内项目根:GPU:PYTHONPATH
RUNS=(
  "w3_cw_rise_5s:recovery-vis:/workspace/variants/w3_cw_rise_5s:3:/workspace/variants/w3_cw_rise_5s/src"
  "p1_paper_exact:recovery-vis:/workspace/variants/p1_paper_exact:2:/workspace/variants/p1_paper_exact/src"
  "p3_paper_exact_s123:recovery-vis:/workspace/variants/p3_paper_exact_s123:2:/workspace/variants/p3_paper_exact_s123/src"
  "p4_paper_baseline_noED:recovery-vis:/workspace/variants/p4_paper_baseline_noED:3:/workspace/variants/p4_paper_baseline_noED/src"
  "g1_go2w_paper:recovery-vis:/workspace/variants/g1_go2w_paper:3:/workspace/variants/g1_go2w_paper/src"
  "g2_go2w_paper_s7:recovery-vis:/workspace/variants/g2_go2w_paper_s7:2:/workspace/variants/g2_go2w_paper_s7/src"
)
EVAL_TARGETS="p1_paper_exact g1_go2w_paper"

declare -A SEED=( [p2_paper_exact_s7]=7 )   # 其余实验统一 seed 42
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOGF"; }

last_iter() {  # <容器> <日志路径>
  docker exec "$1" bash -lc "grep 'Learning iteration' $2 2>/dev/null | tail -1" 2>/dev/null \
    | tr -d ' ' | sed 's/.*iteration//; s#/.*##'
}

is_alive() {  # <容器> <run_name>
  docker exec "$1" bash -lc "pgrep -f 'train.py.*--run_name $2'" >/dev/null 2>&1
}

relaunch() {  # <名称> <容器> <根目录> <GPU> <PYTHONPATH>
  local name=$1 ct=$2 root=$3 gpu=$4 pp=$5
  local latest folder
  latest=$(docker exec "$ct" bash -lc "ls -t $root/logs/rsl_rl/thunder_recovery/*${name}*/model_*.pt 2>/dev/null | head -1")
  [ -n "$latest" ] || { log "!! $name 找不到检查点，无法续训"; return 1; }
  folder=$(basename "$(dirname "$latest")")
  docker exec -d "$ct" bash -lc "cd $root && mkdir -p logs && \
    CUDA_VISIBLE_DEVICES=$gpu ${pp:+PYTHONPATH=$pp} setsid nohup /isaac-sim/python.sh scripts/train.py \
      --task RobotLab-Isaac-Velocity-Recovery-Thunder-v0 --num_envs 4096 --headless \
      --max_iterations $MAX_ITER --seed ${SEED[$name]:-42} --resume --load_run $folder \
      --checkpoint $(basename "$latest") --run_name $name \
      >> logs/train_$name.out 2>&1 < /dev/null"
  log ">> $name 从 $folder/$(basename "$latest") 续训（GPU $gpu）"
}

tick_once() {
  local tick=$1
  local line="tick $tick  进度:"
  for entry in "${RUNS[@]}"; do
    IFS=':' read -r name ct root gpu pp <<<"$entry"
    local it; it=$(last_iter "$ct" "$root/logs/train_$name.out")
    it=${it:-0}
    if is_alive "$ct" "$name"; then
      line+=" $name=$it"
    elif [ "$it" -ge $((MAX_ITER - 1)) ]; then
      line+=" $name=完成($it)"
    else
      line+=" $name=死了($it)"
      log "!! $name 进程已退出（迭代 $it/$MAX_ITER），自动续训"
      relaunch "$name" "$ct" "$root" "$gpu" "$pp"
    fi
  done
  line+="  GPU=$(timeout 12 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d '\n')"
  line+="  load=$(cut -d' ' -f1 /proc/loadavg)"
  log "$line"

  if [ $((tick % EVAL_EVERY)) -eq 0 ]; then
    for tgt in $EVAL_TARGETS; do
      log ">> 定期评估 $tgt（论文口径）"
      EVAL_GPU=3 "$HERE/eval_variant.sh" "$tgt" 512 1 >>"$LOGF" 2>&1
      sleep 200      # 串行，避免同卡并发把 PhysX 挤崩
    done
  fi
}

case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "看门狗已在运行 (pid $(cat "$PIDF"))"; exit 0
    fi
    setsid nohup bash -c "
      HERE='$HERE'; LOGF='$LOGF'; PIDF='$PIDF'
      $(declare -f log last_iter is_alive relaunch tick_once)
      MAX_ITER=$MAX_ITER; EVAL_EVERY=$EVAL_EVERY; TICK=$TICK; EVAL_TARGETS='$EVAL_TARGETS'
      RUNS=($(printf '"%s" ' "${RUNS[@]}")); HERE='$HERE'
      tick=0
      echo \$\$ > \"\$PIDF\"
      log '看门狗启动（间隔 ${TICK}s，评估间隔 $((TICK*EVAL_EVERY/60)) 分钟）'
      while true; do tick=\$((tick+1)); tick_once \$tick; sleep $TICK; done
    " >>"$LOGF" 2>&1 &
    sleep 2; echo "看门狗已启动: pid $(cat "$PIDF" 2>/dev/null)  日志 $LOGF"
    ;;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat "$PIDF")" 2>/dev/null && echo "已停止看门狗"; rm -f "$PIDF";
    else echo "看门狗未运行"; fi ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "运行中 pid $(cat "$PIDF")"; else echo "未运行"; fi
    tail -3 "$LOGF" 2>/dev/null ;;
  log) tail -"${2:-20}" "$LOGF" ;;
  once) tick_once 1 ;;
  *) echo "用法: $0 start|stop|status|log [行数]|once" ;;
esac
