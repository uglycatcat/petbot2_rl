# 7cm台阶策略交付

交付版本为 **[step7cm_s102_u0301](release/step7cm_s102_u0301/README.md)**，在GPU1从随机初始化训练得到，选用第301次PPO更新。双耳固定默认构型，头及左右车架跟随角度，左右后轮跟随速度，前轮被动。原[5cm策略与说明](docs/delivery5cm.md)保留。

## 策略与验证

- `release/step7cm_s102_u0301/policy_raw.pt`：主要交付，TorchScript，含观测归一化，输出5维原始动作。
- 同目录`policy_targets.pt`：直接输出头/左右车架3个角度目标和左右后轮2个速度目标。
- 同目录`policy_raw.onnx`：相同原始动作接口的ONNX，真实输入数值校验通过。

原始检查点与导出TorchScript在不同进程中各完成 **64/64** 次四轮登台，完成时间 **1.55–3.15s，平均1.93s**。单例实际执行录像 **1/1，1.85s** 完成。默认平地初态，平台高7cm、前沿x=0.65m、顶面2m×2m。四轮同时有台面支撑即结束，不附加保持、恢复默认姿态或停车条件。

机器人通过车架转动、身体倾斜和转向登台；没有人为指定动作序列。成功帧四轮最低点距台面最大1.751mm，最小竖直接触力0.188N。轨迹抽查包含最快、最慢、首末样本，目标角度及轮速符合接口限制；实测轮速在接触冲击下可超过目标值，不将目标限幅误写为实际速度限幅。

结果覆盖当前固定初态和平台的PhysX任务。ONNX在4032份实际输入上与原网络最大绝对误差2.39e-6；闭环物理验证使用TorchScript。尚无随机初态、MuJoCo或实机通过结论。

## 在已有项目容器回放

**7cm交付策略每50ms下发一次目标；底层物理和关节伺服仍为2ms。** 从本项目目录在容器中运行：

```bash
/workspace/isaaclab/isaaclab.sh -p play.py \
  --policy release/step7cm_s102_u0301/policy_raw.pt --scripted \
  --step-height 0.07 --control-dt 0.05 \
  --asset-dir assets --output runs/delivery7cm_playback \
  --video --headless --device cuda:0
```

批量复核去掉`--video`、增加`--num-envs 64`并选用新输出目录。输出包含原始物理轨迹、实际策略输入、统计和回放身份；录像直接采集仿真帧，包含成功前最后物理状态。

## 完整交付包

[step7cm_s102_u0301.zip](deliverables/step7cm_s102_u0301.zip)包含发布源码、策略、冻结仿真资产、原检查点与导出策略的64次原始轨迹、实际输入、录像和逐文件SHA256清单。解压后使用其中的`step_explore`为工作目录，无需历史训练权重或重新生成资产。

依赖基座`nvcr.io/nvidia/isaac-lab:2.3.2`提供Isaac Sim5.1、PyTorch2.7、RSL-RL3.1.2。在解压后的项目目录运行以下容器示例（物理GPU仅用本项目授权的0或1）：

```bash
docker run --rm --gpus device=1 --shm-size=4g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMP_NUM_THREADS=4 \
  -v "$PWD:/workspace/step-explore:rw" -w /workspace/step-explore \
  --entrypoint /bin/bash nvcr.io/nvidia/isaac-lab:2.3.2 -lc \
  '/workspace/isaaclab/isaaclab.sh -p play.py --policy release/step7cm_s102_u0301/policy_raw.pt --scripted --step-height 0.07 --control-dt 0.05 --asset-dir assets --output runs/playback --video --headless --device cuda:0'
```

只执行TorchScript推理可用`torch.jit.load`。输入顺序、单位、动作映射和周期见[推理接口](docs/inference.md)。

## 证据与运行结束

- [选择记录](release/step7cm_s102_u0301/selection.json)：冻结训练提交、配置、检查点及源模型/资产哈希。
- [物理评估](release/step7cm_s102_u0301/evaluation.json)与[轨迹复核](release/step7cm_s102_u0301/trajectory_review.json)：导出64次执行。
- [实际执行视频](runs/height7cm_hold50_20261118_163741/video301/execution.mp4)；完整包内对应`evidence/video/execution.mp4`。
- 完整包`evidence/exported64/`、`evidence/checkpoint64/`保存原始证据；`SOURCE_COMMIT`与`SHA256SUMS`用于定位与校验。

GPU1训练在第337更新保存退出，交付仍选择独立验证的第301更新。训练与自动迭代已关闭、实际进程已退出；所有已结束批次禁止恢复。完整经历见[7cm探索记录](docs/step7cm_trials.md)。
