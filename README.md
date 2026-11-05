# PetBot upright 1104：完整从零训练工程

本临时工程以21r5 seed102源码`8cac727`为基线，保留训练、评估、课程、奖励、资产转换、同次中断恢复和ONNX导出。当前`configs/train_1104.json`为**seed102、4096环境、5000更新**，单次GPU3训练已完成，固定训练提交`7ce7bc50`。新实验从随机初始化开始，不依赖demo、history、旧权重或教师；分支为`rlproj/train_upright_project_1104`。

默认桌面包位于[sim2sim工程](../std_s2s_upright_project/README.md)，现为`upright1104_5000_s102`，来自本次`update_005000.pt`，复用21r5控制器。原21r5的1000更新包保留对照；std_train继续为空，demo仅作辅助存档。现有基本闭环可用，落地质量、双向连续±0.75m/s和宽初态尚未全面验证；结果与需求见[当前交接说明](../../docs/upright_1104.md)及[需求基线](../../docs/current_scope.md)。

**没有运行中的upright任务或后续自动迭代授权。** `project.json`中的`training_enabled=true`保留已完成的单次执行配置，不是新的发批授权；工程、评估和自动迭代开关为false。本次文档更新不修改这些开关、训练参数或算法。

## 获取与只读核验

在主仓库根目录执行`repo sync rl_project/train_upright_project_1104`。须保留工作区只读`robot_description/`和相对布局；不需要历史资产缓存。

```bash
cd rl_project/train_upright_project_1104
python3 project.py check
python3 train.py --dry-run
```

配置保留30ms控制、2ms物理、64步rollout、同次课程及低范围回归；与原21r5有效训练参数相比，仅环境数2048→4096、更新数1000→5000。`release.json`区分基线来源、当前配置哈希、最新运行和ONNX身份；逐文件清单不包含训练产物。`tools/check_release.py`是封存时的空产物目录验包工具，包含outputs内不得存在PT的断言；只用于全新检出，不能把已完成训练目录的该项断言失败当作源码损坏。[原验包记录](docs/validation.json)仅描述2026-11-04首次封存。

## 本次运行与日志

本次运行名为`gpu3_4096x5000_s102_20261104`。5000更新完整结束，约19小时21分钟，未记录显存中断、恢复或运行/物理错误；运行完成不等于策略能力通过。

| 项目内路径 | 内容 |
|---|---|
| `logs/gpu3_4096x5000_s102_20261104/` | train.log、监督/启动/退出记录；结束状态以train.exit.json为准 |
| `outputs/gpu3_4096x5000_s102_20261104/run.json`、`status.json` | 随机初始化、配置/软件/源码身份及完成状态 |
| `outputs/gpu3_4096x5000_s102_20261104/checkpoints/update_005000.pt` | 已发布ONNX对应的最终完整检查点 |
| `outputs/gpu3_4096x5000_s102_20261104/tensorboard/` | 200个事件文件，合计5000更新 |
| `outputs/gpu3_4096x5000_s102_20261104/probe_*/` | 训练内课程验证结果和请求/执行/实测轨迹，不是独立验收 |

从本项目目录查看曲线：

```bash
tensorboard --logdir outputs/gpu3_4096x5000_s102_20261104/tensorboard --host 0.0.0.0 --port 6006
```

logs/、outputs/、artifacts/全部被Git忽略，只在服务器本地保存；克隆或repo sync不会下载PT和训练日志。旧轮次在服务器history，本次数据仍在本项目目录。Git中交付的是完整训练源码/配置；ONNX与发布验证随sim2sim工程交付。

## 重新授权后的复现入口

以下是未来操作示例，不表示现在允许启动。需重新确认运行范围，按需开启project.json中的训练/工程/评估项并提交固定版本，自动迭代保持false；训练只在单卡隔离的专用Isaac容器内执行，物理GPU2或3，桌面不训练。运行环境和挂载要求见[环境说明](environment/README.md)。

从宿主调用监督工具，工作区布局与容器名按实际部署设置；本次专用容器为`petbot-upright-1104-gpu3`。先确认没有同名任务，使用全新的run目录。资产缺失时才执行prepare，并需工程授权：

```bash
python3 tools/run_guarded.py --gpu 3 --container petbot-upright-1104-gpu3 \
  --workdir /workspace/petbot/rl_project/train_upright_project_1104 \
  --log logs/new_run_prepare.log -- -m upright.prepare --headless

python3 tools/run_guarded.py --gpu 3 --container petbot-upright-1104-gpu3 \
  --workdir /workspace/petbot/rl_project/train_upright_project_1104 \
  --log logs/new_seed102_train.log -- train.py \
  --config configs/train_1104.json --run-dir outputs/new_seed102 --device cuda:0
```

新运行保存随机初始网络作为证据。仅同次受迫中断可通过`--resume-interrupted outputs/new_seed102/checkpoints/<最新检查点>.pt`恢复完整优化器、课程及随机状态；配置/源码/模型/软件/run_id必须一致，未完成物理回合复位。已完成运行不能续训成新实验，也不能加载历史父模型。

重新获准评估后，在容器项目目录执行`evaluate.py --run-dir outputs/new_seed102 --checkpoint outputs/new_seed102/checkpoints/update_005000.pt --scenario static_cycle --output outputs/new_eval_static --count 32`。另有rise、low_cycle、capability_cycle，后两者可加`--duration 60`；输出目录不能覆盖已有结果。

ONNX标准导出入口为`export_policy.py --run-dir outputs/new_seed102 --checkpoint outputs/new_seed102/checkpoints/update_005000.pt --output outputs/new_export`，它受工程执行开关约束；导出后还需补齐sim2sim模型/执行器清单并检查对应控制器，不能只复制PT。当前5000更新包的来源、数值对照与基本MuJoCo证据见[发布记录](../std_s2s_upright_project/validation/20261105_1104_5000/review.json)。

## 结构与复现边界

- `upright/controller.py`：单次调用、髋轮目标、普通控制交接及限幅。
- `upright/runtime.py`、`curriculum.py`、`task.py`：场景、奖励、课程和Isaac物理轨迹。
- `upright/learning.py`、`distribution.py`、`experiment.py`：单actor/critic PPO、Beta动作及完整状态恢复。
- `upright/model.py`、`prepare.py`、`asset_cfg.py`：只读模型和可重建派生资产。
- `evaluate.py`、`export_policy.py`：显式固定策略验证和ONNX导出。

完整代码/配置已实际完成一次从零5000更新训练；不同驱动或求解器下不承诺逐位复现相同权重，环境镜像仍需准备。源码基线、冻结验包和本次长训证据应分开理解。源模型/物理只读，轮目标min(50,25)=±25rad/s；实际超调单列，无周期定时器或goal。
