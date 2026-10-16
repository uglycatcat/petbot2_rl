# 自动起立的服务器诊断

以原 `control.py`、只读MJCF运行12秒；不按Hop/Cross，不修改模型或控制器。无需GUI，EGL离屏渲染可生成视频和关键帧。结果用于理解构型和任务判据，不用作训练轨迹。

从项目根目录以普通用户执行：

```bash
python3 -m pip install --target artifacts/python_tools uv==0.13.1
artifacts/python_tools/bin/uv venv --python /usr/bin/python3 artifacts/mujoco_env
UV_CACHE_DIR=artifacts/uv_cache artifacts/python_tools/bin/uv pip install --python artifacts/mujoco_env/bin/python -r robot_controller_demo/jump_demo/analysis_requirements.txt
MUJOCO_GL=egl MPLCONFIGDIR=artifacts/matplotlib artifacts/mujoco_env/bin/python robot_controller_demo/jump_demo/analyze_upright.py --render
```

环境与输出均在忽略目录artifacts，不修改系统配置。2026-10-16使用Python3.10.12/MuJoCo3.16.0；默认输出 `artifacts/jump_demo_analysis/` 的CSV、JSON、曲线、关键帧和视频。完整解释及下一版设计见 `docs/upright_v2_design.md`。

`test_maneuver.py` 顶层导入tkinter，本服务器系统Python没有tkinter，不能直接运行其unittest模块。诊断脚本直接调用相同控制器，不依赖tkinter。原物理测试的位移、车架范围、连续支撑及末段速度断言已在采集数据上逐项通过；不声称GUI测试通过。
