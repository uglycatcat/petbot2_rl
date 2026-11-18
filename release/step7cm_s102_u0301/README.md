# step7cm_s102_u0301

7cm策略由GPU1、seed102从随机初始化训练301次PPO更新得到。主要交付`policy_raw.pt`，包含观测归一化；`model_300.pt`使用从0开始的文件索引。

原检查点与导出TorchScript在不同进程各64/64次成功，完成时间1.55–3.15s、平均1.9289s；实际录像1/1，1.85s完成。成功时四轮同时位于7cm台面范围内，轮底最大距台面1.751mm、最小竖直接触力0.188N。双耳固定默认构型，头/左右车架角度控制，左右后轮速度控制，前轮被动。

**策略周期50ms，物理/关节伺服2ms。** 当前回放必须传`--step-height 0.07 --control-dt 0.05`。

| 文件 | 用途 |
|---|---|
| `policy_raw.pt` | 40维观测→5维原始动作；PhysX闭环64/64 |
| `policy_targets.pt` | 同一观测→头/左右车架角度、左右后轮速度目标 |
| `policy_raw.onnx` | 原始动作ONNX；4032份真实输入最大绝对误差2.39e-6 |
| `selection.json`、`completion.json` | 训练身份、模型/资产哈希、运行结束状态 |
| `export.json`、`parity_actual.json` | 导出与真实输入数值校验 |
| `evaluation.json`、`trajectory_review.json` | 导出策略64次评估与代表轨迹复核 |
| `checkpoint_evaluation.json`、`video_playback.json` | 原检查点评估与实际录像身份 |
| `tensorboard_review.json` | 仅截至第301更新的训练曲线摘要 |

[完整交付与回放](../../DELIVERY.md)，[40维推理接口](../../docs/inference.md)。64次均为默认初态和固定平台执行，不代表64种随机场景；没有MuJoCo或实机验证结论。训练与自动迭代因目标达成结束，所选权重不得用作新批次初值。
