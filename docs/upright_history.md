# upright服务器本地历史

2026-11-04冻结归档；当前版本已包含随后单独完成的5000更新训练。**需要upright历史记录，请去服务器仓库的`history/`下面找；桌面与Git不再同步这些历史文件。** 当前有效训练代码是[1104工程](../rl_project/train_upright_project_1104/README.md)，当前结论见[冻结说明](upright_1104.md)。

本次1104长训是例外：日志和检查点保留在`rl_project/train_upright_project_1104/logs/`及`outputs/gpu3_4096x5000_s102_20261104/`；当前MuJoCo原始轨迹在`rl_project/std_s2s_upright_project/outputs/1104_5000_20261105/`。这些也不进Git，不移动回旧runs；默认ONNX与验证JSON随sim2sim发布。

| 本地路径（相对仓库根目录） | 内容 |
|---|---|
| `history/upright/freeze_1104/docs/` | 重启系列1—20、废弃21—26、取消27、21r1—r6、当前诊断与实验计划；保留原文件名 |
| `history/upright/freeze_1104/demo_project/runs/` | 上述运行副本、检查点、日志、状态、轨迹、固定诊断及已停用编排 |
| `history/upright/archival_20261104/` | 本次原路径→归档路径清单、迁移核验、入口原文、版本身份与包装检查 |
| `history/upright/docs/`、`history/upright/demo_project/runs/` | 11月3日已归档的更早系列；旧编号不与重启系列混用 |
| `history/upright/archival_20261103/` | 上次迁移清单、核验与入口原文 |
| `history/remote_archives_20261104/` | 全部4条原远程archive分支的完整Git bundle、分支头、SHA256及删除证据；2条upright、2条step |

本次迁移308个文件或目录，共39,077个目录项、92,417,329,222字节，采用同文件系统改名，逐树核对inode、大小和修改时间；不删除训练数据。少量旧容器创建的目录仅修正目录所有者以允许迁移，单独记录。混合目录只移upright条目，不移动其他任务产物。现用sim2sim的3个忽略参考链接已指向本地归档，其受版本管理代码与策略未改。

所有`archive/*`远程分支已备份后删除，历史分支只留服务器本地。`history/`由根`.gitignore`排除；这次撤下当前树中的历史文档和远程分支，不重写既有Git提交历史，不声称远端旧对象已物理清除。不得再次推送这些历史记录或创建远程archive分支。

检索示例：

```bash
rg --files --hidden --no-ignore history/upright/freeze_1104/docs
rg --hidden --no-ignore '落地' history/upright/freeze_1104/docs/upright_restart_round21r5_results.md
```

旧记录路径保持原文；先在`freeze_1104/`下加原相对路径查找，再查11月3日归档。完整映射见各次`move_plan.json`/`completed_moves.json`。历史内部链接或运行快照中的绝对路径可能失效，需要按清单定位，不能为兼容旧队列恢复runs入口。现用demo和sim2sim内的历史文档引用也按此定位，不为修复旧说明而改变其冻结版本。

如需恢复本地历史代码，先核对`backups.json`及`git bundle verify`，另建隔离目录读取，勿推回远程或启动旧任务。1104训练代码不依赖history、旧检查点或旧资产缓存。当前全部upright训练/自动迭代冻结；归档不会恢复任何旧授权。
