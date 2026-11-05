# STEP 历史定位

2026-11-05整理。当前入口是[STEP交接](step_current.md)和[需求](step_scope.md)。历史只在服务器 `history/step/` 保存，不进Git、不默认加载；旧结论、授权、配置和控制文件均不能成为当前执行依据。

| 本地路径（相对仓库根目录） | 内容 |
|---|---|
| `history/step/pre_v2/docs/` | 重构前逐轮报告、辅助停车、回退及旧sim2sim说明 |
| `history/step/pre_v2/demo_project/runs/` | 旧round/restart系列、旧campaign、诊断及固定运行产物 |
| `history/step/pre_v2/demo_project/step_rl_demo/docs/` | demo内旧任务、算法、工程与自动迭代说明 |
| `history/step/v2_notes/` | 此次精简前的v2长篇叙述，按原相对路径保存；当前结果已提炼到交接，不默认读取流水 |
| `history/step/archival_20261105/` | 原入口备份、已有工作区改动、逐项迁移清单与核验记录 |
| `history/remote_archives_20261104/` | 先前保存的旧STEP源码Git bundle；不恢复远程archive分支 |

共归档202项、18,804个目录项，文件约84.10GB。196项同文件系统移动并核对inode、大小和修改时间；6个早期 `step_probe_dev01`—`06` 目录因原目录权限只复制归档，逐文件SHA256一致，原件留在忽略的runs中。未提权或更改目录权限。当前v2运行、工程证据与控制目录保留在原处。

旧资料内部路径保持原文，可能已失效；按 `move_plan.json` / `completed_moves.json` 找新位置，不为兼容旧队列创建返回原入口的链接。旧记录可能有错误，只能作为待复核的证据，不能未经代码和物理核验带回当前设计。

需要追查时只打开明确相关文件，例如：

```bash
rg --files --hidden --no-ignore history/step/pre_v2/docs
```

本次撤下当前树中的旧资料，不重写Git历史或删除原始实验数据。归档不恢复任何训练授权；源码、权重、轨迹均不得从旧系列迁入新训练。
