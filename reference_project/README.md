# 本地参考项目

此处保留 9 个上游仓库快照（robot_lab、rl_sar、recovery、WBC-AGILE、getup_gym、HumanUP、tron2_rl_lab、b2w-rl、Wheel-Legged-Lab），此外，论文作者的部署栈 Recovery_go2w 也在 `default.xml` 清单里，目前只在一台设备（训练服务器）上物化过。

除 rl_sar、recovery 外只保存调研源码，不安装各仓库依赖，也不代表这些项目能直接解决 PetBot 的完整翻倒恢复。rl_sar 已构建并运行，用于验证「训练产物离开 Isaac Sim 后还能看到效果」这条链路（在桌面机上做，因为 MuJoCo 渲染是桌面机的职责）；recovery 正在训练服务器上试训，汇总见 [recovery 试训汇总](../docs/reference_recovery_validation.md)。

## 选用关系

训练框架优先参考 **robot_lab / Go2W**，完整恢复任务重点参考 **WBC-AGILE / stand_up**，方法上补充 **getup_gym、HumanUP**，策略验证与部署走 **rl_sar**，轮足全姿态起身的唯一可运行实现是 **recovery**；作者自己的部署栈是 **Recovery_go2w**。其余三项用于局部实现对照。需求见 [当前需求基线](../docs/current_scope.md)，仓库边界见 [项目 README](../README.md)。

| 项目 | 用途 | 复用边界 |
|---|---|---|
| [robot_lab](robot_lab/README.md) | 首选训练框架 | 复用 Isaac Lab 扩展、Go2W 混合 action、RSL-RL 训练 / 导出；恢复任务需要新增。 |
| [rl_sar](rl_sar/README.md) | 策略验证与部署入口 | 复用 TorchScript / ONNX 推理、MuJoCo 与 Gazebo 仿真、真机部署；本身不做训练，策略须来自 robot_lab `play.py` 导出，且关节顺序要与 robot_lab cfg 的 `joint_names` 一致。 |
| [recovery](recovery/README.md) | 当前试训对象 | 唯一同时满足轮足 / Isaac Lab / SO(3) 全姿态随机 / 零参考动作；自带 Thunder 的 URDF 与 STL、ED/CW 动态塑形奖励、非对称 actor-critic。需比 2.3.2 更新的 Isaac Lab（`dump_pickle`），只提供 Isaac Lab 通路、无 MuJoCo。试训结论见 [recovery 试训汇总](../docs/reference_recovery_validation.md)。 |
| Recovery_go2w | 作者部署栈 | 论文作者的公开实现，含 `model_7999.pt` 权重与 `simulate_python/` 运行入口；**在清单里，目前只在一台设备上物化过**。可用于对照「作者自己的检查点能否站起」，但权重可用性、观测 / 动作维度一致性与许可证均未核实。 |
| [WBC-AGILE](WBC-AGILE/README.md) | 恢复任务参考 | 倒地状态数据集、stand_up 奖励与课程；当前 Lab 3 beta API 需适配。 |
| [getup_gym](getup_gym/README.md) | 轮足恢复方法 | FTSR 力辅助探索、教师学生与分阶段奖励；基于 Isaac Gym。 |
| [HumanUP](HumanUP/README.md) | 恢复与平滑方法 | 先发现恢复动作再优化平滑性；G1 / Isaac Gym 代码需按 PetBot 适配。 |
| [tron2_rl_lab](tron2_rl_lab/README.md) | 轮足工程对照 | 轮足配置、动作 / 观测组织及训练与部署工程；不是 PetBot 恢复成品。 |
| [b2w-rl](b2w-rl/README.md) | 混合 action 示例 | B2-W 关节位置 + 轮速示例，结构较简洁；上游基线为较旧的 Lab 1.2 / Sim 4.2。 |
| [Wheel-Legged-Lab](Wheel-Legged-Lab/README.md) | 阶段训练对照 | 恢复与移动切换、评估记录；使用 VMC，Recovery 明确不覆盖完全侧躺起立。 |

## 建议优先阅读的源码

- [robot_lab：Go2W 配置](robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py)：位置 / 速度 action 分组与关节名映射。
- [rl_sar：Go2W 策略配置](rl_sar/policy/go2w/robot_lab/config.yaml)：12 个关节位置 + 4 个轮速的混合动作定义（`num_of_dofs: 16`、`wheel_indices: [12, 13, 14, 15]`）、观测顺序与 PD 增益，与本项目的五位置 + 两速度结构最接近。
- [rl_sar：Go2 状态机](rl_sar/src/rl_sar/fsm_robot/fsm_go2.hpp)：被动 / 起身 / 行走的状态切换与键位，以及每个状态硬编码的策略配置名。
- [rl_sar：MuJoCo 仿真主循环](rl_sar/src/rl_sar/src/rl_sim_mujoco.cpp)：观测组装、速度指令来源与重置逻辑。
- [recovery：自由落体重置](recovery/src/thunder_recovery/mdp/method1_deng/events.py)：SO(3) 均匀随机四元数 + 1.1 m 自由落体，前置 2 s 零刚度 / 零阻尼（`zero_action_freefall`），关节角 ±0.3 rad 噪声。
- [recovery：ED / CW 塑形](recovery/src/thunder_recovery/mdp/_utils.py)：`ED(t) = (t/T)^3` 按 episode 内步数计算、`CW(i) = 0.3 × 0.968^i` 随 iteration 衰减，两者与 `--max_iterations` 解耦。
- [recovery：任务配置](recovery/src/thunder_recovery/config/method1_deng/env_cfg.py)：13 项奖励、非对称 actor / critic 观测（78 / 285，实测）。
- [WBC-AGILE：T1 stand_up](WBC-AGILE/agile/rl_env/tasks/stand_up/t1/stand_up_env_cfg.py) 与 [倒地数据预采集](WBC-AGILE/agile/rl_env/tasks/stand_up/t1/pre_learn.py)：重置、奖励、课程及倒地库。
- [getup_gym：轮足配置](getup_gym/getup_gym/envs/bipedal_wheeled/config.py) 与 [环境](getup_gym/getup_gym/envs/bipedal_wheeled/env.py)：轮足恢复实现。
- [HumanUP：训练说明](HumanUP/simulation/README.md) 与 [环境目录](HumanUP/simulation/legged_gym/legged_gym/envs/)：动作发现及后续 refinement。
- [TRON2：轮足配置](tron2_rl_lab/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/WF_TRON2A/limx_base_env_cfg.py)：轮足环境组织。
- [b2w-rl：混合动作环境](b2w-rl/b2w_hybrid_env.py)：位置与轮速协同的简化示例。
- [Wheel-Legged-Lab：阶段训练](Wheel-Legged-Lab/STAGED_TRAINING.md)：课程与评估组织。其 VMC action 与本项目的五位置 + 两速度并不等价。

上游 README 中的效果和训练指标属于对应机器人与任务，不作为 PetBot 的性能预期。完整恢复不能直接复用普通 locomotion 的触地终止或速度跟踪奖励。

## 克隆快照

日期：2026-09-29；rl_sar 与 recovery 于 2026-09-30 追加；Recovery_go2w 于 2026-10-10 在训练服务器上追加。完整来源、分支和 commit 保存在 [manifest.json](manifest.json)。上游分支会变化，以这里的 commit 为本次调研依据。

| 项目 | 上游 | 分支 | commit |
|---|---|---|---|
| robot_lab | [robot_lab](https://github.com/fan-ziqi/robot_lab) | `main` | `500399ed75f510aeaff28705a8ce736c514dbec3` |
| rl_sar | [rl_sar](https://github.com/fan-ziqi/rl_sar) | `main` | `376d42c9b128f963ab08579762d5a216a976ce39` |
| recovery | [recovery](https://github.com/Kitjesen/recovery) | `main` | `5b57477598963695be27474c0ba941a1ba4afd38` |
| Recovery_go2w（仅服务器） | [Recovery_go2w](https://github.com/boyuandeng/Recovery_go2w) | `main` | `e6424625011fbddebe74a2ea0b496fae324b6778` |
| WBC-AGILE | [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) | `main` | `6830cf995714e81c91ce63247e8e016d36e28f14` |
| getup_gym | [getup_gym](https://github.com/2350575870/getup_gym) | `master` | `238d2962a0fee8e3da4da914cadcea2be00ce402` |
| HumanUP | [HumanUP](https://github.com/RunpeiDong/HumanUP) | `master` | `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb` |
| tron2_rl_lab | [tron2_rl_lab](https://github.com/limxdynamics/tron2_rl_lab) | `main` | `db021b4016648a6ddc97fb0acbe5df882cd53fc5` |
| b2w-rl | [b2w-rl](https://github.com/LauraMQuiros/b2w-rl) | `master` | `ad82b971bf69a84170f027035c1e2e3bf97e4ef9` |
| Wheel-Legged-Lab | [Wheel-Legged-Lab](https://github.com/zyicome/Wheel-Legged-Lab) | `main` | `e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7` |

## 本地使用与版本管理

- 使用浅克隆（depth 1、single branch）；除 rl_sar 外未初始化 submodule，并跳过 Git LFS 对象下载。源码可阅读，部分大型资产或权重可能仍是 LFS 指针，不保证每个上游项目都能直接启动。
- rl_sar 已初始化其 6 个 submodule（各厂商 robot SDK 与 joystick），并下载了运行库与机器人描述（`library/`、`cmake_build/`、`src/rl_sar_zoo/`），在本目录中既可读也可跑。
- recovery 无 submodule 与 LFS 依赖，其 `thunder_recovery` 包已在容器内以 `pip install -e .` 安装。
- 子目录保留各自 `.git`，由本目录 [.gitignore](.gitignore) 排除，避免作为主仓库的嵌套仓库提交；主仓库只管理本索引、manifest 与忽略规则。
- 参考项目 checkout **不随主仓库同步**。可按 manifest 记录的上游 URL、分支与 commit 重新克隆；要严格恢复旧快照，应 fetch 对应 commit 后 checkout，并用 `git rev-parse HEAD` 对照。
- 以后要运行某个上游项目，先按其文档检查 LFS / submodule / license 与独立环境要求，并在专用容器内运行、记录实际使用的 commit 与依赖。调研快照不是通用兼容基线。
- **不要把 PetBot 自有的唯一副本放在这些被忽略的目录里**：需要保留的代码与修改说明放进主仓库受版本管理的位置，并注明来源与许可证。

Isaac Lab 与 RSL-RL 属于待配置的训练依赖，本仓库不为它们提供统一安装方案；具体运行配置在对应参考项目的容器任务中确定。

## repo 清单与工作流

`reference_project/` 下的项目由仓库根的 [default.xml](../default.xml) 这份 **repo 清单**统一管理：

```bash
repo init -u https://github.com/uglycatcat/petbot2_rl.git -m default.xml
repo sync                 # 物化/更新全部；只取某类：repo sync -g recovery
```

- 清单是**唯一来源**：新增、删除或变更某个项目的 pin 时，必须同步更新 `default.xml` 与 [manifest.json](manifest.json)。
- 每个项目浅克隆并锁在确定的 commit 上；`clone-depth` 只能写在 `<project>` 上（写在 `<default>` 上会被 repo 忽略，已实测）。
- **被我们改过的参考项目**：把改后的内容推到本仓库的 `refproj/<项目名>` 分支（如 `refproj/recovery`），并把 `default.xml` 中该项目的 `remote` 指向 `ours`、`revision` 指向该分支。这样 `repo sync` 直接得到修改后的版本，改动不会丢在忽略目录里。
- `groups` 划分：`framework`（robot_lab）、`deploy`（rl_sar）、`recovery`（recovery / Recovery_go2w / WBC-AGILE / getup_gym / HumanUP）、`reference`（tron2_rl_lab / b2w-rl / Wheel-Legged-Lab）。

## rl_sar 构建与验证记录

用于验证「训练产物离开 Isaac Sim 后还能看到效果」这条链路 —— 这正是桌面机的职责，所以在桌面机上做，与训练无关。

- 环境：桌面机上的容器（镜像基于 `nvcr.io/nvidia/isaac-lab:2.3.2`；Isaac Lab 2.3.2 / Isaac Sim 5.1）。
- 构建：容器内 `./build.sh -mj`。该命令自动获取 libtorch 2.3.0、ONNX Runtime 1.22.0、MuJoCo 3.2.7 与机器人描述仓库 rl_sar_zoo 1.0.2，产物为 `cmake_build/bin/rl_sim_mujoco`。构建前需初始化 submodule，并补齐 cmake / yaml-cpp / eigen / boost / TBB / spdlog / fmt / lcm / GLFW / python3-dev 等系统依赖。
- 运行：`./cmake_build/bin/rl_sim_mujoco <机器人> <场景>`，窗口内按 `0` 起身、按 `1` 切入 RL 策略。MuJoCo 走 OpenGL 渲染，不依赖 RTX。
- 已验证：`b2 scene`（Unitree B2，纯四足）加载 `policy/b2/robot_lab/policy.pt`（TorchScript，来自 robot_lab / Isaac Sim），起身后稳定站立，策略持续闭环。
- 已知问题：`go2 scene` 的状态机把策略配置名硬编码为 `himloco`（上游基于 Isaac Gym 训练的策略），会起身后翻倒；其余四足（b2、b2w、go2w、d1、tita）默认指向 `robot_lab` 配置。
- 仿真内速度指令只来自手柄 `/dev/input/js0`，无手柄时策略保持原地站立。

## recovery 试训记录

在训练服务器上试训，是当前唯一同时满足轮足 / Isaac Lab / SO(3) 全姿态随机 / 零参考动作的候选实现。**完整记录**（结论、判据 A/B/C、实验矩阵、产物与查询、坑、参数速查）见 [recovery 试训记录](../docs/reference_recovery_validation.md)。
