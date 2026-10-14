#!/usr/bin/env bash
# 列出容器内正在训练的副本及其 GPU（供 launch_variants.sh 统计并发；容器内执行）。
# 只统计真正的 python 进程：/isaac-sim/python.sh 是 shell 包装脚本，其自身进程也会匹配
# "scripts/train.py" 且 cwd 相同，但环境里是容器默认的 CUDA_VISIBLE_DEVICES，会把计数搞乱。
for p in $(pgrep -f "scripts/train[.]py"); do
  exe=$(basename "$(readlink /proc/"$p"/exe 2>/dev/null)" 2>/dev/null)
  case "$exe" in python*) ;; *) continue ;; esac
  cwd=$(readlink /proc/"$p"/cwd 2>/dev/null) || continue
  gpu=$(tr '\0' '\n' < /proc/"$p"/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
  printf '%s %s\n' "${cwd#/workspace/runs/}" "${gpu:-?}"
done | sort -u
