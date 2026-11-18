# 策略加载与输入输出

此页定义本项目策略文件的使用接口；是否完成任务以对应检查点的独立物理评估和运动轨迹为准，导出通过只表示数值推理一致。

## 文件与控制

- `policy_raw.pt`：TorchScript，输入 `[N,40]` float32观测，输出 `[N,5]` 原始动作均值。包含训练得到的观测归一化，可用 `torch.jit.load()` 加载；本项目 `ExploreEnv` 接收此输出。
- `policy_targets.pt`：TorchScript，同样输入40维，直接输出头、左车架、右车架角度（rad），再输出左右后轮速度（rad/s）。包含相同归一化与动作映射。
- `policy_raw.onnx`：可选ONNX原始动作版本，输入名 `observations`，输出名 `raw_actions`，opset17，动态batch维。
- `export.json`：源检查点与导出文件哈希、更新数、数值对照样本和误差；不把导出校验写成登台验收。

控制周期20ms；角度/速度跟随由任务控制层每2ms执行。双耳固定默认构型，前轮被动。输出角度范围为头±1.3rad、双车架−1至+1.2rad；后轮速度目标±25rad/s。输入“上次动作”是上一周期原始策略输出的tanh值，不是实际关节反馈；回合开始清零。

```python
import torch
policy = torch.jit.load("policy_targets.pt", map_location="cpu").eval()
with torch.inference_mode():
    targets = policy(observation)  # observation: [N, 40]
```

## 40维观测顺序

全部字段最终裁剪到[-20,20]，随后使用策略文件内的训练归一化统计。本体坐标系为浮动基座/下身，世界前方是平台法线方向+x；四元数采用wxyz。

| 索引 | 内容 | 单位/缩放 |
|---|---|---|
| 0–2 | 本体坐标线速度 | m/s |
| 3–5 | 本体坐标角速度 | rad/s ×0.2 |
| 6–8 | 世界向下单位重力在本体中的投影 | 无量纲 |
| 9–11 | head、left_frame、right_frame角度 | rad |
| 12–18 | head、left_frame、right_frame、left_front_wheel、left_rear_wheel、right_front_wheel、right_rear_wheel速度 | rad/s ×0.04 |
| 19–21 | 台阶前沿中心点相对基座的位置，转到本体坐标 | m；目标点为(edge,0,height) |
| 22–24 | 世界+x方向转到本体坐标 | 单位向量 |
| 25 | 台阶高度 | 当前0.07m；原5cm策略验证为0.05m |
| 26–33 | 左前、右前、左后、右后轮各自的水平距台沿、最低点距台面 | 每轮2值：(edge−wheel_x, height−wheel_bottom)，m |
| 34–38 | 上一次原始动作的tanh | 每回合初始为0 |
| 39 | 回合控制步计数/最大控制步数 | 当前最大500步 |

车轮最低点由源网格支撑点和实际轮子姿态计算。当前观测允许读取仿真状态和台阶信息；本接口没有宣称已经具备实机传感估计或sim2real能力。

## 导出与实际执行

均在本项目专用容器内，从项目目录运行：

```bash
/isaac-sim/python.sh tools/export_policy.py --checkpoint runs/<批次>/gpu1/final.pt --output runs/<批次>/export --onnx
/workspace/isaaclab/isaaclab.sh -p play.py --policy runs/<批次>/export/policy_raw.pt --scripted --asset-dir runs/<批次>/assets --output runs/<批次>/playback --headless --device cuda:0 --video
```

录像直接采集PhysX执行中的图像，包含终止前的实际状态；`evaluation_traces.npz`保存物理轨迹，`policy_inputs.npz`保存真实策略输入。可再传 `--observations <policy_inputs.npz>` 给导出工具，验证实际输入上的推理一致性。单例录像与多环境独立评估共同用于交付复核。
