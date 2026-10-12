#!/usr/bin/env python3
"""打印 Go2-W MJCF 的关节/执行器/几何信息（用于核对策略映射与 PD 配置）。"""
import mujoco

m = mujoco.MjModel.from_xml_path("/workspace/variants/g1_go2w_paper/assets/go2w/scene.xml")
print("nq", m.nq, "nv", m.nv, "nbody", m.nbody, "nu", m.nu, "dt", m.opt.timestep)
names = lambda obj, i: mujoco.mj_id2name(m, obj, i)
print("bodies:", [names(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
print("joints:")
for i in range(m.njnt):
    print("   %-20s type=%d qposadr=%d dofadr=%d" % (
        names(mujoco.mjtObj.mjOBJ_JOINT, i), m.jnt_type[i], m.jnt_qposadr[i], m.jnt_dofadr[i]))
print("actuators:")
for i in range(m.nu):
    print("   %-20s ctrlrange=%s gear=%s" % (
        names(mujoco.mjtObj.mjOBJ_ACTUATOR, i), m.actuator_ctrlrange[i].tolist(),
        m.actuator_gear[i][0]))
print("geom types:", sorted(set(m.geom_type.tolist())), " ngeom", m.ngeom)
print("total mass:", float(m.body_mass.sum()))
