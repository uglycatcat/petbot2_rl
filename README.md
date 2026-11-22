# step_explore：7cm台阶探索

更新：2026-11-22。

当前在已完成7cm任务基础上，开展动作多样性、动作质量与更高台阶三路探索，见[研究需求](docs/research_scope.md)。本项目与 `step_rl_demo` 的任务设计不同，需求、实现、实验和执行授权独立管理。

**7cm策略已交付：[step7cm_s102_u0301](release/step7cm_s102_u0301/README.md)。** 从零训练，原检查点与导出TorchScript独立评估各64/64，实际录像1/1；默认平地初态出发，四轮同时有台面支撑即成功。策略周期50ms，物理/关节伺服2ms。见[交付与回放](DELIVERY.md)、[40维推理接口](docs/inference.md)。原[5cm交付](docs/delivery5cm.md)保留。

## 已确认需求

- 机器人出现在平地，前方放置平台，目标是登上平台。当前动作多样性/质量实验以7cm为基线，高度路线独立提高台高。用户2026-11-18将原5cm目标提高到7cm，其余要求不变。
- **成功判据：四个轮子都到达台面即可。** 不额外要求稳定保持一段时间、恢复默认姿态或停车。
- **地形观测：允许策略读取台阶的相对位置和距离。** 可直接使用仿真中的对应信息。
- 两个耳朵锁住，保持默认位置，不参与策略动作；其余可控关节可以自由活动。
- 两个主动轮跟随速度目标，其他受控关节跟随角度目标。

遵守仓库模型只读约束，“自由活动”不取消源模型的关节范围或执行器力矩限制。

## 最终交付与执行授权

用户2026-11-17明确：最终交付是一份实际能完成本任务的策略。启动训练、奖励提高或导出文件均不等于完成；须交付可加载的策略文件、运行说明，以及独立评估和原始运动轨迹支持的四轮登台证据。持续迭代以这个结果为目标。

2026-11-22用户授权在GPU0/1/2/3本项目专用容器充分利用可用显存训练并自动迭代；所有实验仍从零训练。原5/7cm首轮工程选择为每卡一路、1024环境、64步采样、1000更新；是否扩容取决于本项目实测资源与吞吐，不继承其他任务预算。每批从随机初始化训练；仅同次受迫中断可恢复。未修改源机器人文件。

## 源模型与控制接口

已只读核对 [机器人MJCF](../../robot_description/mjcf/robot_description.xml)：两个后轮有速度执行器，两个前轮被动；头、双耳及左右车架为有界关节。

采用5维策略动作，左右独立：

| 关节 | 接口 | 策略控制 |
|---|---|---|
| `head` | 角度目标 | 是 |
| `left_frame`、`right_frame` | 各自角度目标 | 是 |
| `left_rear_wheel`、`right_rear_wheel` | 各自速度目标 | 是 |
| `left_ear`、`right_ear` | 默认构型的固定约束（派生仿真资产） | 否 |
| `left_front_wheel`、`right_front_wheel` | 被动关节 | 否 |

源文件中有界关节使用motor执行器，角度跟随在本项目的仿真适配层每2ms执行，保留原电机力矩上限。按双耳锁住要求，仅派生资产将双耳约束在默认构型，保留其质量、惯量与碰撞；源文件不改。容器内真实物理接口检查已通过，证据见[当前交接](docs/current.md)。

## 场景与训练安排

- **训练安排：** 当前四卡六路配置见`configs/research_20261122/plan.json`；交付7cm配置见 `configs/step7cm_hold50.json`（50ms）；20ms探索配置为 `configs/step7cm.json`，原5cm配置保留在 `configs/round01.json`，工程验证与结果见 `runs/` 和本项目后续交接。
- **当前场景：** 默认站姿正对台阶，台沿在初始基座前0.65m（约为整机最前端前0.5m），平台2m×2m。这些尺寸和初态是本项目工程选择。

## 当前状态与文档维护

独立分支为 `demoproj/step_explore`。7cm任务已交付GPU1 seed102第301更新策略，旧交付的训练已结束；新三路搜索获授权，当前运行先读[当前交接](docs/current.md)。没有宣称随机初态泛化、MuJoCo迁移或实机验证通过。

详细需求、设计、实验记录与交接在本项目内维护。外层文档只保留入口、核心信息及重要里程碑；项目内信息更新时，同步修正受影响的外层摘要，不在外层重复展开。

## 使用入口

- [Agent入口](AGENTS.md)、[实现说明](docs/design.md)、[当前交接](docs/current.md)。
- `python3 tools/deploy.py --gpu 0`（可选0–3，第二槽传`--slot b`）准备专用容器；依赖固定版本见 `requirements.txt`，安装包缓存在忽略的 `assets/wheels/`。缓存缺失时先在容器内按requirements安装。
- 在容器的本项目目录运行 `python tools/prepare_asset.py --headless --device cuda:0`，生成任务自有资产；不得覆盖在用的冻结资产。
- 单次训练入口：`python train.py --config configs/step7cm.json --output runs/<运行名> --headless --device cuda:0`。此为工程入口；当前新搜索使用`tools/research_campaign.py`与独立控制，不恢复旧运行。
- 独立加载验证：`python train.py --mode evaluate --checkpoint runs/<运行名>/final.pt --num-envs 64 --output runs/<运行名>/eval --headless --device cuda:0`。

以上 `python` 指容器中 `/workspace/isaaclab/isaaclab.sh -p` 提供的环境。训练与仿真不在宿主运行；宿主只运行部署和campaign脚本。模型与资产由相对项目位置或CLI参数解析，不依赖宿主绝对路径。
