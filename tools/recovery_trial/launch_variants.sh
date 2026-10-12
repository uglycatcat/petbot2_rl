#!/usr/bin/env bash
# 并行启动论文复现的变体训练（在 recovery-vis 容器内，每个变体一个进程）。
# 各变体源码在同一 checkout 的副本里：/workspace/variants/<name>（PYTHONPATH 覆盖
# 容器里 v1 的 editable 安装），日志各自落在 <变体>/logs/train_<name>.out。
#
#   ./launch_variants.sh           # 启动全部变体
#   ./launch_variants.sh v2_exp_ori   # 只启动指定变体
set -euo pipefail

CT=recovery-vis
TASK=RobotLab-Isaac-Velocity-Recovery-Thunder-v0
NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-10000}

# 变体 → 使用的 GPU（宿主机 2/3 号空闲；0/1 号是别人的任务）
declare -A GPU=(
  # 论文标准口径（p 系列）：论文 9 项奖励 + support_state(§III-E) + Eq.1 DS + Eq.3 CW + Eq.4 组合 + §IV-A 超参/DR/动作噪声
  [p1_paper_exact]=2
  [p3_paper_exact_s123]=2
  [p2_paper_exact_s7]=3
  [p4_paper_baseline_noED]=3
  [p5_paper_gated_support]=3
  # 经验延续：第三方复现仓库口径 + 论文 CW 修正（5 s @ 50 Hz，无动作噪声）
  [w3_cw_rise_5s]=2
  # 论文平台：Go2-W（作者 MJCF → USD），论文标准口径
  [g1_go2w_paper]=3
  [g2_go2w_paper_s7]=2
)

declare -A SEED=( [p2_paper_exact_s7]=7 [p3_paper_exact_s123]=123 [g2_go2w_paper_s7]=7 )   # 其余统一 seed 42
# 每个变体的 env 数：Go2-W 的网格碰撞体在 4096 envs 下会让 PhysX 初始化失败（已实测）→ 用 2048
declare -A ENVS=( [g1_go2w_paper]=2048 [g2_go2w_paper_s7]=2048 )
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=("${!GPU[@]}")

for v in "${targets[@]}"; do
  g=${GPU[$v]:?未知变体 $v}
  n=${ENVS[$v]:-$NUM_ENVS}
  docker exec -d "$CT" bash -lc "cd /workspace/variants/$v && mkdir -p logs && \
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/variants/$v/src \
    setsid nohup /isaac-sim/python.sh scripts/train.py --task $TASK \
      --num_envs $n --headless --max_iterations $ITERS --seed ${SEED[$v]:-42} \
      --run_name $v > logs/train_$v.out 2>&1 < /dev/null"
  echo "已启动 $v（GPU $g，$n envs × $ITERS iter）"
done

sleep 20
echo "--- 容器内训练进程 ---"
docker exec "$CT" bash -lc 'pgrep -af "scripts/train.py" | sed "s/.*train.py/train.py/" | head -5'
