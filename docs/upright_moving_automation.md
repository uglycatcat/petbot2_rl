# 运动任务自动迭代流程（2026-10-20）

`tools/campaign.py` 把「等评估 → 分析 → 判定 → 过 gate → 部署下一轮」做成可重复执行的一次循环，
轮次与判定规则登记在 `campaigns/moving.json`。目标不是让脚本代替判断，而是让**判断只发生在需要判断的地方**。

## 分工边界

| 由脚本确定性完成 | 由 agent 完成 |
|---|---|
| 读真实 PID/产物判断批次状态 | 选下一轮要改哪一层、写实现 |
| 聚合 16 次评估、按判据分项归因 | 决定改动是否值得做、如何对照 |
| 按预先登记规则判 A/B 优劣 | 设计新的对照（换初始化、换范围） |
| 契约检查 + 短训 + 重载 + 写 gate | 判断 gate 失败是代码错还是验收过严 |
| 克隆固定副本、守护启动、起评估队列 | 解读失败模式（轨迹、接触、姿态） |
| 生成 `analysis.json` / `decision.json` / `review_request.md` | 写文档、推送、对外结论 |

脚本**从不自己改代码**。需要新实现时它把轮次停在 `needs_review` 并写分层归因，
由 agent 实现后重新登记 commit，再由脚本继续。

## 一轮的时序

```
campaign.py drive
  ├─ 有 in_flight 轮次 → 每 120 s 轮询 eval_finished.json
  │    └─ 收口后 analyze → decide
  │         ├─ keep_new       → 找下一个 planned 轮次
  │         ├─ no_advantage   → park: needs_review + 分层归因
  │         └─ regress        → park: needs_review
  └─ 下一个 planned 轮次
       ├─ 无 commit        → park（先写代码）
       ├─ 无 gate.json     → 提示先跑 gate
       └─ 有 gate          → launch → 新 in_flight
```

## 判定规则（写在 plan 里，不事后改）

- 比较对象：终局检查点、native 范围、两种子合计（分母 128）。
- `gain = B 末2秒保持 − A 末2秒保持`。
- `gain ≥ 8` 且 `B 轨迹MAE ≤ A` 才算 `keep_new`；轨迹变差不算优势。
- `0 < gain < 8` → `no_advantage`；`gain ≤ 0` → `regress`。
- 失败归因按末段判据通过率排序（按各环境末段样本数加权，不是环境平均）。

## 分层归因

`decide` 在判定为 `no_advantage`/`regress` 时输出 `diagnosis.targets`，每条含
「层 / 证据数字 / 可改什么 / 代价 / 风险」，当前规则覆盖：

| 层 | 触发条件 |
|---|---|
| instrumentation | 末段支撑通过低或后轮缺接触比例 ≥1%：现有 0.06 s 单次采样分不清真悬空与单步丢失 |
| action/transition | 支撑中断发生在完全离地之后，配合峰值前向加速度 |
| observation | 末段朝向通过率 <0.85：观测里没有朝向误差积分，累计误差无处表达 |
| initialization/curriculum | 实际接管前向速度跨度过窄（<40% 请求范围） |
| reward/impact、reward | 峰值加速度 >8 m/s²、非零命令下速度停滞比例 >10% |
| generalization | 宽范围保持远低于训练范围 |

## Gate（每轮代码改动必须过）

1. `checks/moving_contract.py`（容器内 CUDA，合成张量断言）。
2. 两组 128×100 短训，退出 0 且日志无 NaN、出现 `Learning iteration`。
3. 64 环境重载，写出 `evaluation.json`。
4. 写 `gate.json`；`launch` 断言 `git diff tested_commit..commit -- upright train_moving.py` 为空，
   gate 之后改了代码就拒绝开训。

## 不自动化的事情

- 放宽成功判据、改任务范围、改 `robot_description/`、改伺服参数。
- 文档提交与推送（`autonomy.commit_docs=false`），避免机器生成的结论直接进主线。
- 多批次并发；批次仍在跑时脚本只等，不插队。

## 用法

```bash
cd demo_project/upright_rl_demo
python3 tools/campaign.py status
python3 tools/campaign.py analyze moving_batch4
python3 tools/campaign.py decide moving_batch4
python3 tools/campaign.py gate   --round moving_batch5 --dry-run
python3 tools/campaign.py drive  --timeout-h 6
```

状态目录 `demo_project/runs/_<batch>/{analysis,decision}.json`；需要设计变更时
`demo_project/runs/_<batch>/review_request.md`。

## 本轮踩到并已修复的坑（2026-10-20/21）

| 现象 | 根因 | 修法 |
|---|---|---|
| 训练/评估进程静默退出，退出码 0，日志停在环境初始化 | `train_moving.py` 的异常被 `finally: app.close()` 吞掉，Isaac 的关闭把退出码变成 0 | 判断失败看**产物**（`evaluation.json`、检查点）而不是退出码；容器内排查时在 `except` 里写文件 |
| 64 次评估里 14 次失败，全部指向 GPU2 | 训练 envelope 里带了 `--initialize-from`，评估复用该 envelope 后同时收到初始化和检查点，脚本主动报错退出 | envelope 只存"训练参数"，初始化源单独传（`init_args`） |
| 公共宽范围评估对带转向配比的运行失败 | `evaluate_moving.py` 重建参数时只搬运 wheel-mode/slew/feedback-mode | 增加 `--turn-share` 搬运 |
| 计划里的 100/175 检查点从不出现 | 保存间隔硬编码 250 | `--save-interval` 参数化，由 plan 传入 |
| 批次被判断为 `in_flight` 而拒绝重开 | 启动器在最后写 status，被中断时状态已落盘 | 重开前把 round 状态改回 `planned` 并清掉 `launch_status.json` |
| 带 `--frame-min=-0.3` 的重评参数解析错位 | 按两个 token 一组配对，等号写法让后续全部移位 | 按 flag 逐 token 解析 |
