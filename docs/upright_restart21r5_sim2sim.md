# 21r5 seed102：用户请求的MuJoCo体验预览

2026-11-04。按用户要求，`rl_project/std_s2s_upright_project`默认切换为`restart21r5_s102`：训练提交`8cac727`、seed102、1000更新、随机初始化，加载导出的ONNX。第20轮原ONNX＋交接修正版仍可显式选择。21r6训练保持运行，本发布不混入21r6的LAND髋投影修改。

配套`restart_v21r5`控制器逐字来自固定训练副本，manifest配置与该运行完全相同，30ms策略/2ms物理；共同髋与两轮观测/动作接口对齐。保留该轮LAND轮目标投影/回零、正常交接当前输入、异常退出、Q/E及方向键调速。源模型和力矩不改，默认两轮幅度0.15m/s、0.5rad/s，普通完整源范围独立。

发布验证：180组源观测/目标完全一致，27姿态几何、普通全速/QE/控制权与ONNX-only检查通过；不指定`--policy`的混合遥控完成1循环、0中断、755帧有限且目标/力矩合法；默认窗口实际启动4秒通过。独立部署副本此前八种固定MuJoCo场景全部完成，原始准备错误和后续有效结果保留在服务器`demo_project/runs/round21r5_mujoco/`，详见[21r5终局](upright_restart_round21r5_results.md)。发布摘要在[review.json](../rl_project/std_s2s_upright_project/validation/20261104_restart21r5/review.json)。

这是体验预览，非完整能力通过：原生低速完整32/基本31，高范围完整/基本30（各32）；MuJoCo静止后退约0.24cm、后轮轴总移动约1.07cm，请求±0.75时实际约+0.468/−0.520m/s。用户本轮桌面体验尚待反馈。

桌面在主仓库更新main并执行`repo sync rl_project/std_s2s_upright_project`，进入该目录后运行`uv run --locked run.py`，默认即21r5。WASD/Space/QE及方向键含义见[项目README](../rl_project/std_s2s_upright_project/README.md)。需要对照第20轮时添加`--policy policies/restart20_s102`；两者使用相同启动输入幅度比较。
