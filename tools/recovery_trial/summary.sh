#!/usr/bin/env bash
# 简要汇总六个实验的学习趋势（从各自训练日志里抽取关键指标）
set -uo pipefail
probe() {  # probe <容器> <日志路径>
  docker exec "$1" bash -lc "awk '
    /^ *Learning iteration/ { it=\$3 }
    /^ *Mean reward:/        { mr[n++]=\$3 }
    /Episode_Reward\\/recovery_base_height:/     { bh=\$2 }
    /Episode_Reward\\/recovery_base_orientation:/{ bo=\$2 }
    /Episode_Reward\\/recovery_stand_joint_pos:/ { sjp=\$2 }
    /Episode_Reward\\/recovery_support_state:/   { ss=\$2 }
    END {
      k = (n>20? n-20 : 0); s=0; c=0
      for (i=k; i<n; i++) { s+=mr[i]; c++ }
      printf \"%s|%s|%.1f|%s|%s|%s|%s|%s\n\", it, mr[0], (c?s/c:0), mr[n-1], bh, bo, sjp, ss
    }' $2"
}

printf "%-14s %-12s %-11s %-11s %-11s %-10s %-11s %-11s\n" 实验 迭代 首个reward 最近20均 末个reward 高度 姿态 支撑态
for v in paper_v1 v2_exp_ori v3_long_ep v4_scale_only v5_soft_servo v6_100hz; do
  case $v in
    paper_v1) ct=recovery-train; log=/workspace/recovery/logs/train_paper_v1.out ;;
    *)        ct=recovery-vis;   log=/workspace/variants/$v/logs/train_$v.out ;;
  esac
  IFS='|' read -r it first last20 last bh bo sjp ss <<<"$(probe "$ct" "$log")"
  printf "%-14s %-12s %-11s %-11s %-11s %-10s %-11s %-11s\n" \
    "$v" "${it:-?}/10000" "${first:-?}" "${last20:-?}" "${last:-?}" "${bh:-?}" "${bo:-?}" "${ss:-?}"
done
