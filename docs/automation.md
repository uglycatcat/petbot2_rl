# STEP 项目管理与运行状态

当前暂停，执行权限只看[当前交接](../../../docs/step_current.md)与最新用户指令。upright只供项目管理方式参考，STEP的算法与运行状态独立。

## 版本和目录

- 开发checkout：`demo_project/step_rl_demo`，repo分支`demoproj/step_rl_demo`。修改前检查工作区、同步远程，保留已有修改。
- main管理`default.xml`、demo/rl索引与manifest；嵌套源码不重复提交main。完整修改在独立分支发布，清单记录实际提交。
- 训练仅用发布后冻结的代码、配置和模型副本；运行中不热改。新任务随机初始化，无历史/工程产物迁移。
- `demo_project/runs/`保留当前v2实验；旧系列见[历史定位](../../../docs/step_history.md)，不恢复旧队列或旧路径链接。

## 状态核对和分析

当前唯一控制入口为`demo_project/runs/_step_refactor_campaign_20261104/`。先看`control.json`的enabled/paused_by_user，再核对`active.json`、对应batch/job状态和实际进程。完成事件不是新授权；已处理或已归档事件不重复发批，暂停时不续接。

每批保存plan、revision、batch状态，每路保存初始化来源、训练曲线、检查点及必要的独立评估证据。先读TensorBoard任务进展、回合长度、奖励和损失；针对异常、实质修改或最终能力查看原始轨迹，不默认重跑整套测试。检查请求/执行/实测、接触和动作，不能只用汇总分数判成功。

恢复前需明确新的训练配置：完整训练至少2048环境、1000次iteration，仅容器内GPU0/1。资源评估区分整卡显存、外部基线、STEP进程占用和GPU计算利用率，不把别人的占用算成训练利用率。现有配置/监控还未按讨论修改。

训练代码实质变化按影响过相应容器、PhysX、PPO短训/同次恢复/重载检查；工程通过才可发布，但不等于策略通过。只允许同次受迫中断以同源码配置恢复，不通过恢复入口加载其他运行权重。

## 停止与事件

当前enabled=false、paused_by_user=true，无训练/guard/控制器运行。获重新授权后也应先检查状态避免重复启动；只管理本任务token对应进程，不干扰其他用户。没有周期定时器或goal。保留原始异常和取消记录，不把工程中断解释为策略能力结论。
