# 参考项目选择建议

> 历史选型记录：以下“当前决定”只对应2026-10-13的 `recovery_agile_01` 原型。AGILE 参考训练已于2026-10-16停止；目前只迭代 [upright_rl_demo](../demo_project/upright_rl_demo/README.md)，新 recovery 将重新设计，不能按本页直接重启训练。

原记录日期：2026-10-13。当时决定以 AGILE 为早期恢复原型底座，并借鉴 robot_lab 混合动作；后续试训结果和关闭决定见[参考项目终局](reference_recovery_validation.md)及[项目里程碑](../AGENTS.md#进度报告文档主线)。

## 当前决定（2026-10-13）

用户确认首个 demo 为 `recovery_agile_01`：基于当前 repo 清单版本的 WBC-AGILE 修改，借鉴 robot_lab 混合动作配置。AGILE 的 `StandUp-T1-v0` 标准训练已于 2026-10-13 启动（4096 envs、100k iteration），robot_lab 尚未开训；PetBot 实现可并行推进，不等待参考试训结束。目录与分支管理、随机落地沉降初始化及首版交付见[工程实施约定](engineering_handoff.md#petbot-demo-管理与首版交付2026-10-13-用户确认)。下文源码比较为选型依据与历史核查，不能当作实测效果。

## 结论与选择依据

选型优先级已明确为：**以接入 PetBot 恢复任务的代码修改成本为主，Docker 环境更换不作为主要负担**。按此标准，在替代 recovery 的已核查候选中，首推 **WBC-AGILE / StandUp-T1-v0**；robot_lab / Go2W 降为混合动作实现参考。AGILE 已有恢复任务与倒地状态采集，预计以资产和任务配置适配为主；robot_lab 则需在运动任务之外补建恢复逻辑。这是源码层面的相对成本判断；现已据此完成首个 demo 的代码实现，并在服务器上跑通训练与评估链路（首轮扫描结论见[首版扫描记录](../demo_project/recovery_agile_01/PETBOT_SWEEP_01.md)）；"改动成本最小"与学习效果均仍需后续验证。

AGILE 接入时需替换资产、五位置两速度 action、观测关节分组、站立目标与接触奖励；关闭或重写 T1 专用镜像增强（`lr_mirror_T1`）。倒地采集按全部关节位置限位随机采样，须为连续轮关节设置有限且合理的初始角度采样方式，并核对缓存与模型版本。默认提升辅助及其课程应按需关闭或适配。资产转换和 MuJoCo 混合伺服适配是各候选共有的工作，不能计作 AGILE 已解决。

2026-10-12 合并服务器记录后补充：CW 递增语义等修正及 Go2-W 第三代实验已开展；作者权重在 MuJoCo 的 32 个随机倒地初态上，平台自适应成功率为 37.5%，已有可恢复站稳的实测基准。详见[最新试训记录](reference_recovery_validation.md)。这些结果来自服务器提交，本机未重新运行验证。AGILE 的代码接入建议不构成已证明其训练效果优于修正后 recovery 的结论，也不意味着停止现有实验；最终选型仍需比较。

2026-10-13 收官：**recovery 线已关闭**（用户决定）。三代 15+ 实验（含论文平台 Go2-W、论文标准口径）在三种判据下**全部 0.00%**，卡点定位为"**站定**"（关节速度 < 0.1 rad/s 保持 1 秒）——机器人普遍能起身、但停不下来；副产物还包括**种子方差达 8 倍**（单种子结论不可信）与"**站得高**"与"**站回默认构型**"冲突（指向奖励目标高度设定）。失败不能归因于判据或训练量。**对选型的影响**：recovery 作为"轮足 + Isaac Lab + 倒地起身"参考实现的价值已经榨干（可运行的参考实现、判据与评估工具链、以及"其奖励/判据不能直接照搬 PetBot"的实证都在[试训记录](reference_recovery_validation.md)里），因此不再投入；恢复任务接入的**实测**候选回到 **WBC-AGILE / `StandUp-T1-v0`**（尚未实测，接入判断见上文）。

| 候选 | 核查事实 | 对 PetBot 的判断（工程推断） |
|---|---|---|
| robot_lab / Go2W | Go2W 配置明确分离关节位置与轮速 action；v2.3.2 对应 Isaac Lab v2.3.2；现成任务为速度跟踪 | 混合动作实现参考；需另建恢复任务，按完整恢复任务的改动成本不再优先 |
| WBC-AGILE / T1 stand_up | 完整恢复任务；自动采集倒地初态；渐退提升辅助；任务配置集中；当前依赖 Lab 3.0.0-beta2、Sim 6.0、Python 3.12 | 按恢复任务接入代码成本首推；改为五位置两速度并调整人形专用项，环境迁移不作为主要排序因素 |
| getup_gym | Isaac Gym Preview 4；轮足配置已有 P/V 混合控制，代码中以外部控制律转换为力矩；力辅助与教师学生框架 | 与轮足恢复相关，但不符合当前内部伺服实现方向，接入需改框架与控制实现，暂作方法参考 |
| HumanUP | Stage I 发现起身与翻滚动作，Stage II 记录并跟踪前一阶段生成的轨迹 | 不因没有预先示范而排除；可自行生成数据，但增加轨迹处理与分阶段训练，暂不作为最省接入工作的首选 |

## PetBot 接入边界

- 默认构型可静稳定站立，无需先训练单独站立策略；恢复后的稳定保持仍需验证。
- PPO 可输出统一连续动作向量，由环境将 5 个分量映射到位置、2 个分量映射到速度；两类动作不要求两套策略。
- 无预先参考动作数据不妨碍任务奖励驱动的 RL；倒地初态库不属于示范动作流。HumanUP 的自生成轨迹与外部预先示范应区分。
- 新增独立 PetBot 恢复任务：资产、7 维动作和内部伺服、倒地初始化、观测、恢复奖励、终止条件与成功判据。不能沿用非足端接触惩罚、腿轮索引或轮绝对角度目标。
- 动作自然性需要视频与平滑性评估；框架选择本身不能保证自然动作或全姿态成功。

## 原建议验证顺序（2026-10-12；现已改为参考试训与 PetBot 实现并行）

1. 在服务器独立容器中复现 AGILE 的 StandUp-T1-v0，先核查倒地采集与短训，再进行关闭辅助的完整评估；上游演示不代替本项目实测。
2. 复现通过后新增 PetBot 资产、独立任务配置及注册，复用恢复流程、PPO 和训练入口；从 robot_lab 借鉴位置/速度分组，不把两个完整框架拼接。
3. 按 PetBot 全身接触和静稳定站姿重新配置目标，评估恢复成功率、保持与自然性。无需先复现 Go2W 行走作为接入前提。

## AGILE 的成熟度与容器运行边界

2026-10-12 核查：AGILE 是面向人形全身控制的训练框架，公开示例主要为 Booster T1 与 Unitree G1；推荐的 StandUp-T1-v0 是人形全身起身任务，并非现成轮足任务。恢复流程可复用，但混合动作与 PetBot 形态适配尚未验证。

工程完整性有文档、任务源码、训练/评估/导出入口、测试目录和官方 `workflows/Dockerfile` 支持；这不能等同于本服务器已复现成功。v1.3 使用 Isaac Lab 3.0.0-beta2，官方发布说明明确其为预发布依赖。GitHub issue #62 对官方 T1 起身视频的站后摆动提出疑问，属于观察与提问，不是已证实的通用训练失败。

官方 Dockerfile 以 `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` 为基底，通过锁定的 uv 环境安装依赖并校验 RSL-RL 补丁。运行原版任务原则上无需先改训练算法，但建容器之外仍须获取 Git LFS 资产、核对宿主驱动与 GPU 透传、完成依赖和任务启动检查。Docker 不隔离宿主 GPU 驱动；本次未检查服务器兼容性、未构建镜像、未启动训练。

当前官方预训练策略清单列出 G1/T1 速度及高度任务，未列出 StandUp-T1-v0 权重。因此不能承诺安装后即可播放已训练好的起身策略；应按需要自行训练或另行核实起身权重来源。启动成功、训练正常推进、最终无辅助恢复成功是三个独立验收层次。

- [官方安装要求](https://nvidia-isaac.github.io/WBC-AGILE/getting-started.html)
- [v1.3.0 发布与限制](https://nvidia-isaac.github.io/WBC-AGILE/releases/v1.3.0.html)
- [预训练策略清单](https://nvidia-isaac.github.io/WBC-AGILE/pretrained-policies.html)
- [T1 站后摆动提问 #62](https://github.com/nvidia-isaac/WBC-AGILE/issues/62)

## 证据与版本

本地源码沿用 [manifest.json](../reference_project/manifest.json) 的快照；另于本次核查上游文档及 robot_lab v2.3.2 源码。以下均为代码/文档证据，本次未实测候选训练表现。

- [robot_lab 版本对应与训练入口](https://github.com/fan-ziqi/robot_lab)
- [robot_lab v2.3.2 Go2W 混合动作配置](https://github.com/fan-ziqi/robot_lab/blob/v2.3.2/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py)
- [AGILE 任务与倒地初态采集](https://nvidia-isaac.github.io/WBC-AGILE/tasks.html#stand-up)
- [AGILE 安装与版本限制](https://nvidia-isaac.github.io/WBC-AGILE/getting-started.html)
- [getup_gym](https://github.com/2350575870/getup_gym)：本地核查 `getup_gym/envs/bipedal_wheeled/config.py` 与 `env.py` 的 `_compute_torques`。
- [HumanUP 两阶段训练说明](https://github.com/RunpeiDong/HumanUP/blob/master/simulation/README.md)
