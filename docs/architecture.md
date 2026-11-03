# 当前模块职责

以下为当前模块职责。执行状态见主仓库[当前交接](../../../docs/current_handoff.md)；CPU检查不能证明仿真或策略有效。

| 模块 | 职责 |
|---|---|
| `model.py` / `prepare.py` / `asset_cfg.py` | 只读源模型、派生资源、源限幅伺服和被动前轮 |
| `geometry.py` / `controller.py` | 不依赖Isaac的运动/几何接口、单次调用状态与控制所有权 |
| `runtime.py` | 场景采样、局部观测、真实阶段评分、奖励 |
| `curriculum.py` | 同次运行的表现驱动晋级；开发检查与独立验收分开 |
| `task.py` | Isaac Lab适配和真实子步接触/力矩记录 |
| `distribution.py` / `learning.py` | 原生RSL PPO/单网络与完整状态保存 |
| `experiment.py` | 固定身份、重复运行防护、暂停与同次恢复边界 |
| `evaluation.py` / `export.py` | 首回合轨迹与结果、单网络ONNX及接口manifest |

公开入口只有一个训练`train.py`，以及专用`evaluate.py`、`export_policy.py`与资产准备模块。它们不扫描历史runs或自动加载旧模型。源模型与固定实验副本只读，新训练不依赖历史权重。

当前修订见[21r1设计](round21r1.md)；[第1轮实现](round01_implementation.md)仅作起点记录；[源码取用](source_provenance.md)说明实际保留和剥离的历史部分。
