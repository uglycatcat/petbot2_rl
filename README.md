# upright_rl_demo

PetBot 起立训练项目。分支 `demoproj/upright_rl_demo`，由根仓库 `default.xml` 物化到 `demo_project/upright_rl_demo`。

2026-10-15 登记。训练代码还没写。

任务是从默认姿态维持只有两个后轮着地，位置最好不变，左右执行器对称，脖子和耳朵不动。本机参照是 `robot_controller_demo/jump_demo`，那个动作太僵硬，不是模仿数据。说明在主仓库 `docs/upright.md`（嵌套检出时相对路径 `../../docs/upright.md`）。

完整跳跃已放弃。`jump_rl_demo` 仍保留，不要删。与 `recovery_agile_01` 的摔倒恢复训练无关，不要改那条线的代码或运行副本。
