# step5cm_s103_u0201

主要策略为 `policy_raw.pt`，从本项目GPU1 seed103第201次更新导出，包含训练得到的观测归一化。原始检查点文件 `model_200.pt` 使用从0开始的索引，不能误写为200次更新。

独立物理验证：原始检查点64/64，导出TorchScript64/64，单例实际录像1/1。导出批量完成时间1.70–3.58s、均值2.161875s。成功时所有轮子同时位于台面范围内，网格最低点最大距台面1.749mm，四轮均有正竖直接触力。

| 文件 | 用途 |
|---|---|
| `policy_raw.pt` | 40维观测 → 5维原始动作；本项目PhysX闭环已验证 |
| `policy_targets.pt` | 同一观测 → 头/左右车架角度、左右后轮速度目标 |
| `policy_raw.onnx` | 原始动作ONNX，动态batch，opset17；真实输入数值校验 |
| `selection.json` | 训练身份、源模型/资产哈希、证据路径与验收范围 |
| `export.json`、`parity_actual.json` | 文件哈希与实际输入推理对照 |
| `evaluation.json`、`trajectory_review.json` | 64次导出策略执行与动作复核 |
| `checkpoint_evaluation.json`、`video_playback.json` | 原网络评估与单例录像身份 |
| `tensorboard_review.json` | 截至所选更新的训练曲线 |

运行方式见[完整交付说明](../../docs/delivery5cm.md)，接口见[推理说明](../../docs/inference.md)。这里的64次均为当前默认初态和固定平台的执行，不代表64种随机场景，也不代表实机验证。训练结束后不得将所选权重当作下一批训练初值。
