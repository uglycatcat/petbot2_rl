"""对称的后轮速度和车架力矩能否把前轮抬离地面。

机器人左右对称，四轮着地。本测试只读 MJCF，不打开窗口，也不做时间积分。
它在四轮着地的静止姿态上组装质量矩阵和接触雅可比，解一次约束动力学：

    M qacc = tau - qfrc_bias + J^T f
    J qacc = 0

后轮接地点保持不离地、不纵向打滑；头和双耳加速度保持为 0。前轮不加入约束。
前轮最低点的垂向加速度大于 0，表示地面无法再提供拉力，前轮离地。
离地之后支撑只剩两个后轮，矢状面运动就是绕后轮接地点的俯仰，即倒立摆。

另外在车架角为 0 的对称姿态里寻找质心落在后轮轴正上方的俯仰角，并确认
越过该角后重力矩会把机体继续推离，而不是拉回。那是倒立摆的不稳定平衡，
不是悬挂摆。

这里证明的是：限位内的对称指令能在不打滑的前提下进入这种支撑方式，
并且倒立摆平衡姿态落在关节限位内。没有积分一条摆到该平衡点的轨迹。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parents[3] / "robot_description/mjcf/scene.xml"
WHEELS = (
    "left_rear_wheel",
    "right_rear_wheel",
    "left_front_wheel",
    "right_front_wheel",
)
REAR = ("left_rear_wheel", "right_rear_wheel")
FRONT = ("left_front_wheel", "right_front_wheel")
BODY_GEOMS = (
    "lower_body",
    "head",
    "left_ear",
    "right_ear",
    "left_frame",
    "right_frame",
)
HELD_JOINTS = ("head", "left_ear", "right_ear")


def _ids(model: mujoco.MjModel):
    joint = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
        for i in range(model.njnt)
    }
    qadr = {name: int(model.jnt_qposadr[i]) for name, i in joint.items()}
    dadr = {name: int(model.jnt_dofadr[i]) for name, i in joint.items()}
    return qadr, dadr


def _world_mesh(model: mujoco.MjModel, data: mujoco.MjData, geom: str) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    mesh = int(model.geom_dataid[gid])
    start = int(model.mesh_vertadr[mesh])
    count = int(model.mesh_vertnum[mesh])
    verts = model.mesh_vert[start : start + count]
    rotation = data.geom_xmat[gid].reshape(3, 3)
    return data.geom_xpos[gid] + verts @ rotation.T


def _lowest(model: mujoco.MjModel, data: mujoco.MjData, geom: str) -> np.ndarray:
    cloud = _world_mesh(model, data, geom)
    return cloud[int(np.argmin(cloud[:, 2]))].copy()


def _set_free(data: mujoco.MjData, x: float, z: float, pitch: float, frame: float, qadr):
    """pitch > 0 为抬头（绕世界 +Y 的负转角）。左右车架使用同一个角度。"""
    half = -0.5 * pitch
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:7] = (x, 0.0, z, np.cos(half), 0.0, np.sin(half), 0.0)
    data.qpos[qadr["left_frame"]] = frame
    data.qpos[qadr["right_frame"]] = frame


def _solve_two(residual):
    guess = np.array([0.0, 0.09])
    for _ in range(16):
        value = residual(guess)
        if np.linalg.norm(value) < 1e-10:
            return guess, value
        step = 1e-7
        jacobian = np.zeros((2, 2))
        for index in range(2):
            moved = guess.copy()
            moved[index] += step
            jacobian[:, index] = (residual(moved) - value) / step
        guess = guess - np.linalg.solve(jacobian, value)
    value = residual(guess)
    raise RuntimeError(f"姿态未收敛，残差 {value}")


def _plant_four_wheels(model, data, qadr, frame: float = 0.0):
    """解出左右车架同角时，前轮和后轮最低点同时落在 z=0 的俯仰与高度。"""

    def residual(var):
        _set_free(data, 0.0, var[1], var[0], frame, qadr)
        mujoco.mj_forward(model, data)
        return np.array([
            _lowest(model, data, "left_rear_wheel")[2],
            _lowest(model, data, "left_front_wheel")[2],
        ])

    pitch_z, err = _solve_two(residual)
    _set_free(data, 0.0, pitch_z[1], pitch_z[0], frame, qadr)
    mujoco.mj_forward(model, data)
    return float(pitch_z[0]), float(pitch_z[1]), err


def _plant_rear(model, data, qadr, pitch: float, frame: float):
    """后轮最低点放到原点，机体俯仰和车架角给定。"""

    def residual(var):
        _set_free(data, var[0], var[1], pitch, frame, qadr)
        mujoco.mj_forward(model, data)
        point = _lowest(model, data, "left_rear_wheel")
        return np.array([point[0], point[2]])

    solved, err = _solve_two(residual)
    _set_free(data, solved[0], solved[1], pitch, frame, qadr)
    mujoco.mj_forward(model, data)
    return err


def _jac_row(model, data, body: str, point: np.ndarray, axis: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    mujoco.mj_jac(
        model,
        data,
        jacp,
        jacr,
        np.ascontiguousarray(point, dtype=float),
        body_id,
    )
    return jacp[axis]


def _saturated_torque(model, data, name: str, ctrl: float) -> float:
    """执行器出力，再扣掉该关节的静摩擦。速度为 0。"""
    actuator = model.actuator(name)
    index = int(actuator.id)
    gain = float(model.actuator_gainprm[index, 0])
    bias = model.actuator_biasprm[index]
    # 速度执行器的仿射偏置用第 3 项乘关节速度；静止时这一项为 0。
    force = gain * ctrl + float(bias[0])
    low, high = model.actuator_forcerange[index]
    force = float(np.clip(force, low, high))
    joint_id = int(model.actuator_trnid[index, 0])
    dof = int(model.jnt_dofadr[joint_id])
    friction = float(model.dof_frictionloss[dof])
    if abs(force) <= friction:
        return 0.0
    return float(np.sign(force) * (abs(force) - friction))


def _rear_friction(model, data) -> float:
    saved = data.qpos.copy()
    data.qpos[2] -= 2.0e-4
    mujoco.mj_forward(model, data)
    mus = []
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[k]))
            for k in (0, 1)
        }
        if "floor" in names and names & set(REAR):
            mus.append(float(contact.friction[0]))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    if len(mus) < 2 or max(mus) - min(mus) > 1e-9:
        raise RuntimeError(f"后轮接触摩擦没有成对读到：{mus}")
    return float(mus[0])


def _constrained_acceleration(model, data, tau: np.ndarray, contacts: dict[str, np.ndarray], dadr):
    """后轮不离地、不纵向滑动，头和耳锁住。返回广义加速度和约束力。"""
    nv = model.nv
    rows = []
    for name in REAR:
        rows.append(_jac_row(model, data, name, contacts[name], 2))
        rows.append(_jac_row(model, data, name, contacts[name], 0))
    for name in HELD_JOINTS:
        hold = np.zeros(nv)
        hold[dadr[name]] = 1.0
        rows.append(hold)
    jacobian = np.vstack(rows)
    mass = np.zeros((nv, nv))
    mujoco.mj_fullM(model, mass, data.qM)
    nc = jacobian.shape[0]
    kkt = np.zeros((nv + nc, nv + nc))
    kkt[:nv, :nv] = mass
    kkt[:nv, nv:] = -jacobian.T
    kkt[nv:, :nv] = jacobian
    rhs = np.zeros(nv + nc)
    rhs[:nv] = tau - data.qfrc_bias
    solution = np.linalg.solve(kkt, rhs)
    residual = kkt @ solution - rhs
    if np.linalg.norm(residual) > 1e-6:
        raise RuntimeError(f"约束动力学残差过大：{np.linalg.norm(residual)}")
    return solution[:nv], solution[nv : nv + 4]


def evaluate():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    qadr, dadr = _ids(model)
    pitch, height, plant_err = _plant_four_wheels(model, data, qadr, 0.0)
    if np.linalg.norm(plant_err) > 1e-8:
        raise RuntimeError(f"四轮着地姿态残差 {plant_err}")

    contacts = {name: _lowest(model, data, name) for name in WHEELS}
    mu = _rear_friction(model, data)
    mass = float(np.sum(model.body_mass))
    gravity = float(-model.opt.gravity[2])
    com = data.subtree_com[1].copy()
    rear = contacts["left_rear_wheel"]
    front = contacts["left_front_wheel"]
    axle = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rear_wheel")]
    horizontal = float(com[0] - rear[0])
    com_height = float(com[2] - rear[2])
    radius = float(axle[2] - rear[2])
    wheelbase = float(front[0] - rear[0])

    wheel_ctrl = float(model.actuator("left_rear_wheel").ctrlrange[1])
    frame_ctrl = float(model.actuator("left_frame").ctrlrange[0])
    # 左右指令相同。正的后轮速度是向前驱动；负的车架力矩是能抬前轮的方向。
    tau_wheel = _saturated_torque(model, data, "left_rear_wheel", wheel_ctrl)
    tau_frame = _saturated_torque(model, data, "left_frame", frame_ctrl)
    if tau_wheel <= 0.0 or tau_frame >= 0.0:
        raise RuntimeError(f"饱和力矩符号异常 wheel={tau_wheel} frame={tau_frame}")

    def torque(wheel: float, frame: float) -> np.ndarray:
        applied = np.zeros(model.nv)
        applied[dadr["left_rear_wheel"]] = wheel
        applied[dadr["right_rear_wheel"]] = wheel
        applied[dadr["left_frame"]] = frame
        applied[dadr["right_frame"]] = frame
        return applied

    front_jac = np.vstack([
        _jac_row(model, data, name, contacts[name], 2) for name in FRONT
    ])

    def response(wheel: float, frame: float):
        qacc, force = _constrained_acceleration(
            model, data, torque(wheel, frame), contacts, dadr
        )
        return front_jac @ qacc, force

    passive_a, _ = response(0.0, 0.0)
    wheel_a, wheel_f = response(tau_wheel, 0.0)
    frame_a, frame_f = response(0.0, tau_frame)
    both_a, both_f = response(tau_wheel, tau_frame)
    wrong_a, wrong_f = response(tau_wheel, -tau_frame)

    drive = 2.0 * tau_wheel / radius
    # 四轮仍着地时，对后轮接地点取矩：前轮支持力随驱动力增大而下降。
    front_normal = (mass * gravity * horizontal - drive * com_height) / wheelbase

    def offset(at_pitch: float) -> float:
        _plant_rear(model, data, qadr, at_pitch, 0.0)
        center = data.subtree_com[1]
        axis = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rear_wheel")]
        return float(center[0] - axis[0])

    low, high = 0.7, 1.2
    if offset(low) <= 0.0 or offset(high) >= 0.0:
        raise RuntimeError("车架角为 0 时，质心没有从后轮轴前方跨到后方")
    for _ in range(40):
        mid = 0.5 * (low + high)
        if offset(mid) > 0.0:
            low = mid
        else:
            high = mid
    balance = 0.5 * (low + high)
    _plant_rear(model, data, qadr, balance, 0.0)
    front_clearance = min(float(np.min(_world_mesh(model, data, name)[:, 2])) for name in FRONT)
    body_clearance = min(float(np.min(_world_mesh(model, data, name)[:, 2])) for name in BODY_GEOMS)
    slope = (offset(balance + 0.02) - offset(balance - 0.02)) / 0.04

    return {
        "pitch": pitch,
        "height": height,
        "mass": mass,
        "gravity": gravity,
        "mu": mu,
        "horizontal": horizontal,
        "com_height": com_height,
        "radius": radius,
        "wheelbase": wheelbase,
        "wheel_ctrl": wheel_ctrl,
        "frame_ctrl": frame_ctrl,
        "tau_wheel": tau_wheel,
        "tau_frame": tau_frame,
        "front_normal": front_normal,
        "passive_a": passive_a,
        "wheel_a": wheel_a,
        "frame_a": frame_a,
        "both_a": both_a,
        "both_f": both_f,
        "wrong_a": wrong_a,
        "wrong_f": wrong_f,
        "wheel_f": wheel_f,
        "frame_f": frame_f,
        "contacts": contacts,
        "balance_pitch": balance,
        "front_clearance": front_clearance,
        "body_clearance": body_clearance,
        "slope": slope,
    }


def _friction_ok(force: np.ndarray, mu: float) -> bool:
    # force 顺序：左后 z、左后 x、右后 z、右后 x。正的 z 是地面把轮子向上推。
    for side in (0, 2):
        normal = float(force[side])
        longitudinal = abs(float(force[side + 1]))
        if normal <= 0.0 or longitudinal > mu * normal:
            return False
    return True


def check(report: dict):
    both = report["both_a"]
    assert both[0] > 1.0 and both[1] > 1.0, (
        f"前轮没有向上加速：{both}"
    )
    assert abs(both[0] - both[1]) < 1e-3, f"左右前轮加速度不对称：{both}"
    assert _friction_ok(report["both_f"], report["mu"]), (
        f"后轮支持力或摩擦不够：force={report['both_f']} mu={report['mu']}"
    )
    assert np.all(report["passive_a"] < 0.0), (
        f"不加指令时前轮已经离地：{report['passive_a']}"
    )
    assert np.all(report["wheel_a"] > 0.0), f"后轮力矩单独没有抬前轮：{report['wheel_a']}"
    assert np.all(report["frame_a"] > 0.0), f"车架力矩单独没有抬前轮：{report['frame_a']}"
    assert both[0] > report["wheel_a"][0], "车架力矩没有在轮速之上继续增大前轮加速度"
    assert np.all(report["wrong_a"] < 0.0), (
        f"反向车架力矩仍能抬前轮，符号判断不成立：{report['wrong_a']}"
    )
    assert report["front_normal"] < 0.0, (
        f"准静态前轮支持力仍为正：{report['front_normal']}"
    )
    assert report["horizontal"] / report["com_height"] <= report["mu"], (
        "把全部重量移到后轮时，所需摩擦超过模型后轮摩擦"
    )
    assert report["front_clearance"] > 0.02, (
        f"平衡姿态前轮离地不足：{report['front_clearance']}"
    )
    assert report["body_clearance"] > 0.0, (
        f"平衡姿态机体碰到地面：{report['body_clearance']}"
    )
    assert report["slope"] < 0.0, (
        f"质心偏移对抬头角的斜率不是负的：{report['slope']}"
    )
    left_y = report["contacts"]["left_rear_wheel"][1]
    right_y = report["contacts"]["right_rear_wheel"][1]
    assert left_y > 0.0 and abs(left_y + right_y) < 1e-6, "后轮接触没有左右对称"


def format_report(report: dict) -> str:
    normal = report["both_f"]
    rear_normal = float(normal[0] + normal[2])
    friction = abs(float(normal[1] + normal[3])) / rear_normal
    return "\n".join([
        "对称倒立摆可行性（无窗口，静止约束动力学，一次线性求解）",
        (
            f"四轮着地：俯仰 {report['pitch']:+.4f} rad，"
            f"质心在后轮接地点前 {report['horizontal']:.4f} m、高 {report['com_height']:.4f} m，"
            f"后轮半径 {report['radius']:.4f} m，轴距 {report['wheelbase']:.4f} m"
        ),
        (
            f"指令：左右后轮同为 {report['wheel_ctrl']:+.2f} rad/s，"
            f"左右车架同为 {report['frame_ctrl']:+.2f} N·m；"
            f"扣除静摩擦后每个后轮 {report['tau_wheel']:+.3f} N·m，"
            f"每个车架 {report['tau_frame']:+.3f} N·m"
        ),
        (
            f"合指令：前轮垂向加速度 {report['both_a'][0]:+.2f} m/s^2，"
            f"后轮支持力 {rear_normal:.2f} N，所需摩擦 {friction:.3f}，"
            f"模型后轮摩擦 {report['mu']:.3f}"
        ),
        (
            f"分开看：仅轮速 {report['wheel_a'][0]:+.2f} m/s^2，"
            f"仅车架 {report['frame_a'][0]:+.2f} m/s^2，"
            f"车架反号 {report['wrong_a'][0]:+.2f} m/s^2，"
            f"零指令 {report['passive_a'][0]:+.2f} m/s^2"
        ),
        (
            f"准静态仅轮速：前轮支持力公式给出 {report['front_normal']:+.3f} N"
        ),
        (
            f"倒立摆平衡：车架角 0，抬头 {report['balance_pitch']:.3f} rad，"
            f"前轮离地 {report['front_clearance']:.3f} m，机体离地 {report['body_clearance']:.3f} m，"
            f"质心水平偏移对抬头角的斜率 {report['slope']:+.3f} m/rad"
        ),
        "结论：该对称指令能让前轮离地且后轮保持受压、不打滑；平衡姿态存在且不稳定。",
    ])


class RearWheelInvertedPendulumTest(unittest.TestCase):
    def test_symmetric_wheel_and_frame_torque_lifts_front(self):
        report = evaluate()
        print(format_report(report))
        check(report)


if __name__ == "__main__":
    unittest.main()
