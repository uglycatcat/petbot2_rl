# upright旧系列归档

2026-11-03按用户要求整理。**需要重启前的历史记录，可以去训练服务器仓库的`history/upright/`下面找。** 该目录由根`.gitignore`排除，不随Git或repo同步到桌面；Git已有提交历史不重写。

| 位置（相对仓库根目录） | 内容 |
|---|---|
| `history/upright/docs/` | 重启前静止/旧移动/rebuild/完整循环/scratch系列文档、旧入口快照及实验计划；保留原文件名 |
| `history/upright/demo_project/runs/` | 上述旧系列固定副本、检查点、日志、状态和轨迹，保留原目录名 |
| `history/upright/archival_20261103/` | 原路径→归档路径清单、迁移校验、文件元数据、入口修改前全文及文档SHA256；`before/`保存未提交的原文版本 |

本次迁移154份旧文档/计划、546个旧运行目录及39份旧日志文件。目录采用同文件系统迁移，保留81,320份目录内普通文件及其inode/大小/修改时间，约133GB；文档另核对SHA256，没有删除训练数据。完整清单在归档内，不把几百个历史链接重新放回当前入口。

**保留边界：** 新系列`upright_restart_round01`至20、无效21—26、取消27、当前21r1及其工程/诊断目录全部保留原路径，文档见[新系列索引](upright_restart_history.md)。第20轮固定参考、现用sim2sim策略包及其证明材料不动；重建快照`_upright_reorganization_20261030`和运行依赖`_upright_runtime_sync_20261030`保留。混有recovery记录的`_night_20261015`只归档明确带upright名称的文件；其他任务目录不动。

需要追溯时在服务器执行，例如：

```bash
rg --files --hidden --no-ignore history/upright/docs
rg --hidden --no-ignore '起立超时' history/upright/docs/upright_cycle_batch24_results.md
```

旧记录里的路径、固定commit、状态和结论保持当时原文；从旧`docs/...`、`demo_project/runs/...`查找时加`history/upright/`前缀。迁移清单`completed_moves.json`记录个别文件和完整目录。旧快照中的绝对路径不代表可直接原地重新运行；如需复现，应另建隔离副本并审查依赖，不能恢复废弃队列、历史权重初始化或旧定时器。新训练不依赖history存在。

日常入口仅为AGENTS→当前交接→当前轮次；历史成功率、失败与旧授权不会覆盖现行[需求基线](current_scope.md)。
