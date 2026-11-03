# 历史源码与产物

重启前完整Git源码分支为`archive/upright_rl_demo_pre_rebuild_20261030`，固定b357dd720ae5cc42e0c630476a0df007fe7dbc3e；保留原提交历史。重建工作目录快照仍在`../runs/_upright_reorganization_20261030/checkout_before_rebuild/`，180份原跟踪文件SHA一致，含生成物和未跟踪文件。

2026-11-03已将旧系列文档和运行记录移到服务器仓库的`history/upright/`，不进Git；按原路径加该前缀查找。例如旧第26批取消证据为`history/upright/demo_project/runs/_upright_cycle_batch26/cancellation.json`。该批未完成1000更新，不恢复。完整映射和原文快照见[归档说明](../../docs/upright_history.md)。

2026-10-30后的新系列1—20及后续仍在原位置；原21—26无效、27取消，代码归档分支为`archive/upright_restart_through_round27_20261103`。当前执行以[交接](../../docs/current_handoff.md)为准。

历史源码可在核查依赖与有效性后参考；历史权重、冻结父网络和教师不得作为新PPO初始化。查看旧源码用Git对象或独立目录，不切换当前工作区恢复废弃训练：

```bash
git show archive/upright_rl_demo_pre_rebuild_20261030:README.md
```
