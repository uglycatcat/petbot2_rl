#!/usr/bin/env bash
# 对指定变体的最新检查点做一次论文口径评估（中途或最终都可用）。
#   ./eval_variant.sh <variant|paper_v1> [envs] [episodes]
# 关键点：评估必须用与训练时同一份代码（PYTHONPATH 指向该变体的 src），
# 否则 episode 长度/奖励/伺服增益等口径不一致，数字没有可比性。
set -euo pipefail

V=${1:?用法: eval_variant.sh <variant|paper_v1> [envs] [episodes]}
ENVS=${2:-1024}
EPS=${3:-1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$V" = "paper_v1" ]; then
  CT=recovery-train
  ROOT=/workspace/recovery
  PP=""
else
  CT=recovery-vis
  ROOT=/workspace/variants/$V
  PP=$ROOT/src
fi

# 取该变体最新的检查点
CK=$(docker exec "$CT" bash -lc "ls -t $ROOT/logs/rsl_rl/thunder_recovery/*${V}*/model_*.pt 2>/dev/null | head -1")
[ -n "$CK" ] || { echo "找不到 $V 的检查点"; exit 1; }
echo "变体=$V 容器=$CT 检查点=$(basename "$CK")"

docker exec -d "$CT" bash -lc "cd $ROOT && CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3} \
  PYTHONPATH=$PP /isaac-sim/python.sh /workspace/recovery_env/eval_recovery.py \
  --checkpoint $CK --num_envs $ENVS --episodes $EPS \
  --out $ROOT/logs/eval_$V.json > logs/eval_$V.out 2>&1"
echo "已启动评估（日志 $ROOT/logs/eval_$V.out）"
