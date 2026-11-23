# step8cm_s1214_u1500：8cm登台

2026-11-23阶段成果，来源为本项目从零训练的`height_a/trial002`，seed1214，第1500更新。研究与自动迭代继续进行，原5cm/7cm交付保留。

原检查点与导出TorchScript独立评估各64/64，实际录像1/1。批量完成时间1.55–3.70s、平均2.110s；单例录像3.00s。平台8cm，默认平地初态；四轮同时支撑即成功，不附加保持或停车。

证明当前固定场景8cm登台可行；动作仍激烈，64个成功回合中5个出现身体明显翻转。它用于高度探索，没有作为动作平滑版本。

## 使用

`policy_raw.pt`是含归一化的TorchScript，输入40维、输出5维原始动作；`policy_targets.pt`直接输出头/左右车架角度及左右后轮速度目标。实际输入重载数值差为0，见`parity_actual.json`。本次未新增ONNX交付。

在本项目专用Isaac容器中运行：

```bash
/workspace/isaaclab/isaaclab.sh -p play.py \
  --policy release/step8cm_s1214_u1500/policy_raw.pt --scripted \
  --step-height 0.08 --control-dt 0.05 \
  --asset-dir assets --output runs/replay_step8cm_s1214_u1500 \
  --video --headless --device cuda:0
```

策略周期50ms、物理/关节伺服2ms，双耳默认锁定，前轮被动，头/车架角度与后轮速度接口不变。完整包为`deliverables/step8cm_s1214_u1500.zip`，包含发布源码、冻结资产、策略、64次原始评估及视频；解压后在`step_explore`目录运行同一命令。

## 证据

`selection.json`记录训练提交、来源、配置、模型和资产哈希；`evaluation.json`与`checkpoint_evaluation.json`为两次独立执行；`physical_audit.json`重新核对原始支撑几何、接触力与动作限幅；`motion_metrics.json`记录逐回合运动指标；`tensorboard_review.json`仅覆盖所选检查点；`video_playback.json`记录录像策略身份。完整包`evidence/`为原始轨迹与视频，`SHA256SUMS`与`SOURCE_COMMIT`用于校验。

[阶段比较](../../RESEARCH_RESULTS.md)，[40维推理接口](../../docs/inference.md)。上述为固定默认初态和对应台高的PhysX结果，不代表随机场景泛化或硬件验证。
