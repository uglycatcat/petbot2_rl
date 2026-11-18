# step_explore 当前交接

更新：2026-11-18。**7cm任务已交付`step7cm_s102_u0301`，训练和自动迭代因目标达成结束，当前无运行队列。** [交付与回放](../DELIVERY.md)，[策略文件](../release/step7cm_s102_u0301/README.md)。

## 当前需求与结果

用户将台阶由5cm提高至7cm，其余要求不变：默认平地出发，四轮同时有台面支撑即成功；双耳锁默认位置；头/双车架独立角度控制、左右后轮独立速度控制，前轮被动；允许读取台阶相对位置和距离。源模型只读，与其他STEP任务独立。

GPU1 seed102从随机初始化训练，选用第301更新`model_300.pt`。原检查点与导出TorchScript在不同进程各64/64成功，1.55–3.15s、平均1.92890625s；录像1/1，1.85s。最快、最慢、首末样本均复核原始关节/接触轨迹，四轮同时支撑；成功帧轮底最大误差1.7504mm、最小支撑力0.1879N。机器人允许转向、倾斜，未增加保持、归姿态或停车要求。

导出TorchScript在4032份实际输入上误差0，ONNX最大绝对误差2.3842e-6；闭环使用TorchScript。结论仅限当前默认初态、固定7cm平台的PhysX场景，不代表随机场景、MuJoCo或实机验收。

## 复现身份与证据

- 冻结训练提交`0bdd718533aad2d8e3026ce930646dbbd1307817`，批次`runs/height7cm_hold50_20261118_163741`。
- 配置`configs/step7cm_hold50.json`，1024环境×32步，seed102、std0.65、熵0.01；初始预算1000更新，实际337更新后因目标达成退出，选用301更新。
- **7cm策略周期50ms，物理/关节伺服2ms。回放必须传`--step-height 0.07 --control-dt 0.05`。** 当前play的通用周期默认仍20ms，原5cm策略使用20ms。
- 批次内`checkpoint301_gpu1/`为原网络64次评估，`export301/`为导出，`export301_eval64/`为导出64次独立执行，`video301/`为实际视频。评估先读TensorBoard至索引300，再审查轨迹。
- `release/step7cm_s102_u0301/selection.json`保存哈希和完整配置。12个源模型文件与44个冻结源码文件均复核未变；原5cm交付zip哈希保持`e50ddebaf053510783fd9ae7f0b6620a8dfd7d8c70e5f57c7719c58e87216807`。
- 交付包`deliverables/step7cm_s102_u0301.zip`包含源码、资产、策略、原始评估轨迹、实际输入及视频。包内源码为`82a9a990`；80项SHA256全部通过，解压后另起进程回放1/1成功、1.85s完成。包哈希和验包结果见[验包记录](../release/step7cm_s102_u0301/bundle_verification.json)。50ms批次从冻结启动至训练退出约34.18分钟，不含此前失败对照。

## 执行状态与后续边界

`control/control.json`为enabled=false、automatic_iteration=false、stop_current=true；`control/active.json`无活动trial。宿主watchdog及GPU1训练进程已退出，专用容器仅空闲基础进程。底层status因统一停止机制写为interrupted，**批次及gpu1的`completion.json`明确goal_achieved、禁止恢复**；它不是资源中断续训授权。

原20ms两路已关闭，GPU0为491更新、GPU1为305更新；5cm旧批次也保持终态。不要启动旧事件、峰值奖励对照或新训练。新的同时支撑峰值奖励只完成工程检查，未正式训练，交付使用原有奖励。失败对照与设计依据见[7cm探索记录](step7cm_trials.md)。

原5cm成果、zip和证据保持：[5cm交付](delivery5cm.md)、[原任务记录](step5cm.md)。当前源码回放5cm须传`--step-height 0.05 --control-dt 0.02`。全部详细记录在项目内，外层仅同步核心状态。
