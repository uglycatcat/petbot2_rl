#!/usr/bin/env bash
# 对 demo_project/runs/<变体> 逐个做独立评估：用固定 commit 副本自己的最后检查点，
# split=eval 初态库（seed 与训练不同）、512 episodes；3 路并行（GPU3）。
#
# 原因：petbot_eval.py 的入口块是 try/main() finally app.close()，异常会被吞掉且静默退出，
# 所以这里统一用 demo_env/run_eval_debug.py 包装，把 traceback 打出来。
# 用法：./eval_all.sh [变体...]   （不给参数就评估全部 11 组）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根：向上找到含 default.xml 的目录（脚本放在 demo_project/demo_env/ 或 tools/petbot_trial/ 都能用）。
PROJ="$HERE"
while [ ! -f "$PROJ/default.xml" ] && [ "$PROJ" != "/" ]; do PROJ="$(dirname "$PROJ")"; done
RUNS="$PROJ/demo_project/runs"
LOGDIR="$RUNS/_logs"
CT="petbot-train"
PY=/isaac-sim/python.sh
GPU="${PETBOT_EVAL_GPU:-3}"
PARALLEL="${PETBOT_EVAL_PARALLEL:-3}"
# 评估脚本用主线（已修 restore_cfg）的那份，agile 包仍取副本自己的（PYTHONPATH）
EVAL_SCRIPT=/workspace/demo_project/recovery_agile_01/scripts/petbot_eval.py
ALL=(base_s42 base_s43 gains2x_s42 gains_down_s42 decimation5_s42 reward_still_s42 reward_config_heavy_s42 episode15_s42 spawn1m_s42 ppo64_s42 ppo_explore_s42)
TARGETS=("$@"); [ "${#TARGETS[@]}" -eq 0 ] && TARGETS=("${ALL[@]}")

running_evals() { docker exec "$CT" bash -lc 'pgrep -cf "petbot_eval[.]py" || true' 2>/dev/null | head -1; }

launch() {
  local v="$1" run ckpt
  run="$(basename "$(ls -d "$RUNS/$v/logs/rsl_rl/recovery_agile_01/"*/ | tail -1)")"
  ckpt="/workspace/runs/$v/logs/rsl_rl/recovery_agile_01/$run/model_2999.pt"
  printf '[%s] 评估 %s（%s）\n' "$(date +%H:%M:%S)" "$v" "$run"
  docker exec -d "$CT" bash -lc "cd /workspace/runs/$v && PYTHONPATH=/workspace/runs/$v CUDA_VISIBLE_DEVICES=$GPU setsid nohup $PY /workspace/demo_env/run_eval_debug.py $EVAL_SCRIPT --headless --num_envs 512 --episodes 512 --checkpoint $ckpt --output artifacts/petbot_eval.json > /workspace/runs/_logs/eval_$v.log 2>&1 < /dev/null"
}

for v in "${TARGETS[@]}"; do
  while [ "$(running_evals)" -ge "$PARALLEL" ]; do sleep 20; done
  launch "$v"
  sleep 5
done
while [ "$(running_evals)" -gt 0 ]; do sleep 20; done

echo "[$(date +%H:%M:%S)] 全部评估结束，汇总："
python3 - "$RUNS" "${TARGETS[@]}" <<'PY'
import json, sys
runs, targets = sys.argv[1], sys.argv[2:]
hdr = f'{"变体":<24}{"all":>8}{"upright":>9}{"inverted":>10}{"side/tilt":>11}{"count":>7}'
print(hdr); print("-" * len(hdr))
rows = []
for v in targets:
    try:
        d = json.load(open(f"{runs}/{v}/artifacts/petbot_eval.json"))["summary"]
    except Exception as exc:
        print(f'{v:<24} 读取失败: {exc}'); continue
    s = d["all"]
    f = lambda k: ("—" if d[k]["success_rate"] is None else f'{d[k]["success_rate"]:.3f}')
    rows.append((s["success_rate"] or 0, v, f, s, d))
for _, v, f, s, d in sorted(rows, reverse=True):
    print(f'{v:<24}{f("all"):>8}{f("upright"):>9}{f("inverted"):>10}{f("side_or_tilted"):>11}{s["count"]:>7}')
PY
