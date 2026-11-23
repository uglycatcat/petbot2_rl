# step7cm_quality_s1213_u1000：7cm动作质量改进

2026-11-23阶段成果，来源为本项目从零训练的`quality_b/trial002`，seed1213，第1000更新。研究与自动迭代继续进行，原5cm/7cm交付保留。

原检查点与导出TorchScript独立评估各64/64，实际录像1/1。批量完成时间1.45–1.55s、平均1.496s；单例录像1.50s。平台7cm，默认平地初态；四轮同时支撑即成功，不附加保持或停车。

原版对比：平均头部偏角0.483→0.050rad，远处车架偏角1.025→0.092rad，目标变化率7.73→3.09/s，实际角度关节加速度145.34→29.77rad/s²。远处抬轮比例82.5%→30%，有明显改善但并未完全消失。64/64轨迹没有身体倒置。

## 使用

`policy_raw.pt`是含归一化的TorchScript，输入40维、输出5维原始动作；`policy_targets.pt`直接输出头/左右车架角度及左右后轮速度目标。实际输入重载数值差为0，见`parity_actual.json`。本次未新增ONNX交付。

在本项目专用Isaac容器中运行：

```bash
/workspace/isaaclab/isaaclab.sh -p play.py \
  --policy release/step7cm_quality_s1213_u1000/policy_raw.pt --scripted \
  --step-height 0.07 --control-dt 0.05 \
  --asset-dir assets --output runs/replay_step7cm_quality_s1213_u1000 \
  --video --headless --device cuda:0
```

策略周期50ms、物理/关节伺服2ms，双耳默认锁定，前轮被动，头/车架角度与后轮速度接口不变。完整包为`deliverables/step7cm_quality_s1213_u1000.zip`，包含发布源码、冻结资产、策略、64次原始评估及视频；解压后在`step_explore`目录运行同一命令。

## 证据

`selection.json`记录训练提交、来源、配置、模型和资产哈希；`evaluation.json`与`checkpoint_evaluation.json`为两次独立执行；`physical_audit.json`重新核对原始支撑几何、接触力与动作限幅；`motion_metrics.json`记录逐回合运动指标；`tensorboard_review.json`仅覆盖所选检查点；`video_playback.json`记录录像策略身份。完整包`evidence/`为原始轨迹与视频，`SHA256SUMS`与`SOURCE_COMMIT`用于校验。

[阶段比较](../../RESEARCH_RESULTS.md)，[40维推理接口](../../docs/inference.md)。上述为固定默认初态和对应台高的PhysX结果，不代表随机场景泛化或硬件验证。
