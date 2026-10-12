"""打印 Go2-W 模型的客观数值（质量、关节/执行器数、力矩上限），用于文档核对。

用法（容器内）：
    /isaac-sim/python.sh /workspace/recovery_env/go2w_model_facts.py [模型路径]
默认模型路径取容器内可用的 Go2-W MJCF 副本（见 recovery_env/models/）。
"""
import sys
import mujoco

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/variants/g1_go2w_paper/assets/go2w/scene.xml"
m = mujoco.MjModel.from_xml_path(path)
print("模型:", path)
print("nq, nv, nu =", m.nq, m.nv, m.nu)
print("关节数 =", m.njnt, "（其中自由/球关节:", sum(1 for i in range(m.njnt) if m.jnt_type[i] != 3), "）")
print("总质量 = %.4f kg" % float(m.body_mass.sum()))
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    cl, ch = m.actuator_ctrlrange[i]
    fl, fh = m.actuator_forcerange[i]
    print(f"  actuator {i:2d} {name:24s} ctrlrange [{cl:8.2f}, {ch:8.2f}]  forcerange [{fl:6.2f}, {fh:6.2f}]")
