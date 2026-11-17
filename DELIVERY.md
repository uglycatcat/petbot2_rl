# 5cm台阶策略交付

交付版本为 `step5cm_s103_u0201`，在GPU1从随机初始化训练得到，完成201次PPO更新。双耳固定默认构型，头及双车架跟随角度，左右后轮跟随速度，前轮被动。

## 策略与验证

策略文件在 [release/step5cm_s103_u0201](release/step5cm_s103_u0201/README.md)：

- `policy_raw.pt`：主要交付，TorchScript，含观测归一化，输出原始5维动作，供本项目环境执行。
- `policy_targets.pt`：TorchScript，直接输出3个角度目标和2个后轮速度目标。
- `policy_raw.onnx`：相同原始动作接口的ONNX版本，已通过真实仿真输入数值对照。

原始检查点和导出TorchScript在不同进程中各完成64/64次四轮登台，导出策略完成时间1.70–3.58s，平均2.16s；单例实际执行录像2.52s完成。均从当前默认平地初态出发，平台高5cm、前沿x=0.65m、顶面2m×2m。没有改变耳朵/关节接口，没有指定动作序列。策略采用明显的车架转动、身体倾斜和转向；四轮同时到台面即成功，没有附加保持、归姿态或停车条件。

此结果覆盖当前固定初态和场景的PhysX任务。ONNX在11456份真实输入上与原网络最大绝对误差2.29e-5；闭环物理验证使用TorchScript。没有随机初态、MuJoCo或实机通过结论。

## 在已有项目容器回放

从本项目目录运行（容器内GPU编号为0）：

```bash
/workspace/isaaclab/isaaclab.sh -p play.py \
  --policy release/step5cm_s103_u0201/policy_raw.pt --scripted \
  --asset-dir assets --output runs/delivery_playback \
  --video --headless --device cuda:0
```

批量复核去掉 `--video`，增加 `--num-envs 64`，并使用新的输出目录。输出包含原始轨迹、真实策略输入、成功统计；录像为实际仿真帧，包含成功前最后物理状态。工具会从轨迹重新核对四轮台面支撑。

## 使用完整交付包

服务器本地完整包为 `deliverables/step5cm_s103_u0201.zip`；其中包含发布源码、上述策略、冻结仿真资产、64次评估原始轨迹、真实策略输入、演示录像和逐文件SHA256清单。解压后以该目录作为工作目录；不需要历史训练权重，也无需重新生成资产。

依赖基座为 `nvcr.io/nvidia/isaac-lab:2.3.2`，提供Isaac Sim5.1、PyTorch2.7、RSL-RL3.1.2。可把解压目录挂载到已有本项目容器；新建容器时仅选择本项目授权的物理GPU0或1，示例：

```bash
docker run --rm --gpus device=1 --shm-size=4g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMP_NUM_THREADS=4 \
  -v "$PWD:/workspace/step-explore:rw" -w /workspace/step-explore \
  --entrypoint /bin/bash nvcr.io/nvidia/isaac-lab:2.3.2 -lc \
  '/workspace/isaaclab/isaaclab.sh -p play.py --policy release/step5cm_s103_u0201/policy_raw.pt --scripted --asset-dir assets --output runs/playback --video --headless --device cuda:0'
```

只运行TorchScript推理时不需要仿真器，使用 `torch.jit.load` 即可。输入顺序、单位、坐标系、动作映射与控制周期必须遵守[推理接口](docs/inference.md)。

## 证据入口

- `release/step5cm_s103_u0201/selection.json`：训练提交、检查点、模型和资产哈希。
- 同目录 `evaluation.json`、`trajectory_review.json`：导出策略64次物理评估及代表动作。
- 同目录 `checkpoint_evaluation.json`：原始检查点独立评估。
- 同目录 `tensorboard_review.json`：仅截至所选第201更新的训练曲线摘要。
- 完整包 `evidence/exported64/`：原始轨迹与实际输入；`evidence/video/execution.mp4`：单例录像。
- 完整包 `SHA256SUMS`：文件校验；`SOURCE_COMMIT`：交付源码版本。
