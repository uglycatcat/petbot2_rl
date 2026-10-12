#!/usr/bin/env bash
# 当前实验的学习趋势一屏（从各自训练日志抽取关键指标）。
# 注意：
#   1) 实验清单需与 launch_variants.sh / watchdog.sh 的 RUNS 保持一致，换实验时同步更新。
#   2) 训练日志里带 ANSI 颜色转义（^[[1m / ^[[0m），必须先剥掉再按字段解析，
#      且迭代数要按 "数字/数字" 这一 token 匹配，不能靠字段位置。
set -uo pipefail

# 名称:容器:目标迭代数   （目标数不同是因为续训会累加，见文档「坑 15」）
RUNS=(
  "g1_go2w_paper:recovery-vis:10000"
  "g2_go2w_paper_s7:recovery-vis:10000"
  "p1_paper_exact:recovery-vis:10000"
  "p3_paper_exact_s123:recovery-vis:10000"
  "p4_paper_baseline_noED:recovery-vis:10000"
  "w3_cw_rise_5s:recovery-vis:13200"
)

probe() {  # probe <容器> <日志路径>
  docker exec "$1" bash -lc "sed -r 's/\x1b\[[0-9;]*[A-Za-z]//g' $2 | awk '
    { for (i=1;i<=NF;i++) if (\$i ~ /^[0-9]+\/[0-9]+\$/) { split(\$i,a,\"/\"); it=a[1] } }
    /Mean reward:/               { mr[n++]=\$3 }
    /recovery_base_height:/      { bh=\$2 }
    /recovery_base_orientation:/ { bo=\$2 }
    /recovery_support_state:/    { ss=\$2 }
    END {
      k = (n>20? n-20 : 0); s=0; c=0
      for (i=k; i<n; i++) { s+=mr[i]; c++ }
      printf \"%s|%s|%.1f|%s|%s|%s|%s\n\", it, mr[0], (c?s/c:0), mr[n-1], bh, bo, ss
    }'"
}

echo "== 实验进度（$(date '+%F %T')）=="
printf "%-22s %-13s %-11s %-11s %-11s %-10s %-11s %-9s\n" 实验 迭代 首个reward 最近20均 末个reward 高度 姿态 支撑态
for entry in "${RUNS[@]}"; do
  IFS=':' read -r v ct target <<<"$entry"
  log=/workspace/variants/$v/logs/train_$v.out
  IFS='|' read -r it first last20 last bh bo ss <<<"$(probe "$ct" "$log")"
  printf "%-22s %-13s %-11s %-11s %-11s %-10s %-11s %-9s\n" \
    "$v" "${it:-?}/${target}" "${first:-?}" "${last20:-?}" "${last:-?}" "${bh:-?}" "${bo:-?}" "${ss:-?}"
done
echo "（显示 ? 说明日志/容器不对；reward 绝对值跨实验不可比，看趋势与 by_initial_pose 分层请用 eval_variant.sh）"
