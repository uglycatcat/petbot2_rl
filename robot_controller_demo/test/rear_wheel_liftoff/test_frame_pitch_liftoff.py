"""jump_demo 的后轮支撑姿态上，车架 pitch 力矩能否把后轮也抬离地面。

姿态用 jump_demo 的保持目标：抬头角和左右车架角固定，后轮最低点落在地面，
前轮离地，速度为零。后轮速度命令为 0。左右车架使用同一个力矩，头和双耳
加速度保持为 0。本测试只读 MJCF，不开窗口，也不做时间积分。

后轮还被约束在地面上时，解一次

    M qacc = tau - qfrc_bias + J^T f
    J qacc = 0

垂向乘子是地面把轮子向上推的力。它变成负值时，地面必须提供拉力才能维持接触，
单边接触下后轮会离地。整机受到的向上净力是支反力减去重量。接触力去掉之后，
车架力矩是内力，质心垂向加速度只剩重力。
"""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from robot_controller_demo.jump_demo.control import _FRAME, _PITCH
from robot_controller_demo.test.rear_wheel_inverted_pendulum.test_lift_feasibility import (
    BODY_GEOMS,
    FRONT,
    HELD_JOINTS,
    REAR,
    SCENE,
    _ids,
    _jac_row,
    _lowest,
    _plant_rear,
    _rear_friction,
    _saturated_torque,
    _world_mesh,
)


def _solve(model, data, tau, contacts, dadr, stick: bool):
    """头和耳锁住。stick 为真时后轮接地点不离地、不纵向滑动。"""
    nv = model.nv
    rows = []
    if stick:
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
    count = jacobian.shape[0]
    kkt = np.zeros((nv + count, nv + count))
    kkt[:nv, :nv] = mass
    kkt[:nv, nv:] = -jacobian.T
    kkt[nv:, :nv] = jacobian
    rhs = np.zeros(nv + count)
    rhs[:nv] = tau - data.qfrc_bias
    solution = np.linalg.solve(kkt, rhs)
    if np.linalg.norm(kkt @ solution - rhs) > 1e-6:
        raise RuntimeError("约束动力学残差过大")
    qacc = solution[:nv]
    jac_com = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, 1)
    rear_z = np.array([
        _jac_row(model, data, name, contacts[name], 2) @ qacc for name in REAR
    ])
    return {
        "qacc": qacc,
        "force": solution[nv:nv + 4] if stick else np.zeros(4),
        "com_acc": jac_com @ qacc,
        "rear_z": rear_z,
    }


def _frame_torque(model, data, dadr, ctrl: float) -> np.ndarray:
    each = 0.0 if ctrl == 0.0 else _saturated_torque(model, data, "left_frame", ctrl)
    tau = np.zeros(model.nv)
    tau[dadr["left_frame"]] = each
    tau[dadr["right_frame"]] = each
    return tau, each


def _normal(force: np.ndarray) -> float:
    return float(force[0] + force[2])


def _longitudinal(force: np.ndarray) -> float:
    return float(force[1] + force[3])


def _in_cone(force: np.ndarray, mu: float) -> bool:
    for side in (0, 2):
        normal = float(force[side])
        if normal <= 0.0 or abs(float(force[side + 1])) > mu * normal:
            return False
    return True


def evaluate():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    qadr, dadr = _ids(model)
    plant_err = _plant_rear(model, data, qadr, _PITCH, _FRAME)
    if np.linalg.norm(plant_err) > 1e-8:
        raise RuntimeError(f"后轮着地姿态残差 {plant_err}")

    contacts = {name: _lowest(model, data, name) for name in (*REAR, *FRONT)}
    mu = _rear_friction(model, data)
    mass = float(np.sum(model.body_mass))
    weight = mass * float(-model.opt.gravity[2])
    limit = float(model.actuator("left_frame").ctrlrange[1])

    def response(ctrl: float, stick: bool):
        tau, each = _frame_torque(model, data, dadr, ctrl)
        solved = _solve(model, data, tau, contacts, dadr, stick)
        solved["each"] = each
        solved["ctrl"] = ctrl
        return solved

    held = response(0.0, True)
    press = response(-limit, True)
    unload = response(limit, True)
    released = response(limit, False)

    low, high = 0.0, limit
    for _ in range(50):
        mid = 0.5 * (low + high)
        if _in_cone(response(mid, True)["force"], mu):
            low = mid
        else:
            high = mid
    cone = response(low, True)

    low, high = 0.0, limit
    for _ in range(50):
        mid = 0.5 * (low + high)
        if _normal(response(mid, True)["force"]) > 0.0:
            low = mid
        else:
            high = mid
    zero_normal = response(low, True)

    front_clearance = min(float(contacts[name][2]) for name in FRONT)
    body_clearance = min(
        float(np.min(_world_mesh(model, data, name)[:, 2])) for name in BODY_GEOMS
    )
    return {
        "pitch": _PITCH,
        "frame": _FRAME,
        "mu": mu,
        "mass": mass,
        "weight": weight,
        "limit": limit,
        "contacts": contacts,
        "front_clearance": front_clearance,
        "body_clearance": body_clearance,
        "held": held,
        "press": press,
        "unload": unload,
        "released": released,
        "cone": cone,
        "zero_normal": zero_normal,
    }


def check(report: dict):
    assert report["front_clearance"] > 0.12, (
        f"前轮离地不足：{report['front_clearance']}"
    )
    assert report["body_clearance"] > 0.0, (
        f"机体碰到地面：{report['body_clearance']}"
    )
    rear_z = [report["contacts"][name][2] for name in REAR]
    assert max(abs(z) for z in rear_z) < 1e-8, f"后轮没有落在地面：{rear_z}"

    held_n = _normal(report["held"]["force"])
    assert held_n > 0.0, f"保持姿态本身后轮已经不受压：{held_n}"

    press = report["press"]
    press_n = _normal(press["force"])
    assert press_n > report["weight"], (
        f"负车架力矩没有产生大于重量的向上支反力：{press_n}"
    )
    assert float(press["com_acc"][2]) > 1.0, (
        f"负车架力矩没有让质心向上加速：{press['com_acc'][2]}"
    )
    assert _in_cone(press["force"], report["mu"]), (
        f"向上推时后轮打滑：force={press['force']} mu={report['mu']}"
    )

    unload_n = _normal(report["unload"]["force"])
    assert unload_n < 0.0, f"正车架力矩仍能把后轮压在地上：{unload_n}"
    assert _normal(report["cone"]["force"]) > 1.0, (
        f"摩擦锥内已经能把后轮支反力降到零：{_normal(report['cone']['force'])}"
    )
    assert np.all(report["released"]["rear_z"] > 1.0), (
        f"松开接触后后轮没有向上加速：{report['released']['rear_z']}"
    )
    gravity = -report["weight"] / report["mass"]
    assert abs(float(report["released"]["com_acc"][2]) - gravity) < 1e-8, (
        f"离地后质心垂向加速度不是重力：{report['released']['com_acc'][2]}"
    )


def format_report(report: dict) -> str:
    press_n = _normal(report["press"]["force"])
    unload_n = _normal(report["unload"]["force"])
    cone_n = _normal(report["cone"]["force"])
    zero = report["zero_normal"]
    return "\n".join([
        "后轮支撑姿态上的车架力矩（无窗口，静止约束动力学，一次线性求解）",
        (
            f"jump_demo 保持目标：抬头 {report['pitch']:+.3f} rad，"
            f"车架 {report['frame']:+.3f} rad，前轮离地 {report['front_clearance']:.3f} m，"
            f"重量 {report['weight']:.2f} N，后轮摩擦 {report['mu']:.3f}"
        ),
        (
            f"车架力矩 {report['press']['each']:+.3f} N·m："
            f"后轮支反力 {press_n:.2f} N，质心垂向加速度 {report['press']['com_acc'][2]:+.2f} m/s^2，"
            f"后轮仍受压"
        ),
        (
            f"车架力矩 {report['unload']['each']:+.3f} N·m："
            f"维持接触所需支反力 {unload_n:.2f} N"
        ),
        (
            f"支反力降到 0 需要每个车架 {zero['each']:+.3f} N·m，"
            f"但摩擦锥在 {report['cone']['each']:+.3f} N·m 时已经用尽，"
            f"那时支反力仍有 {cone_n:.2f} N"
        ),
        (
            f"若忽略摩擦、直接松开接触：后轮最低点垂向加速度 "
            f"{report['released']['rear_z'][0]:+.2f} m/s^2，"
            f"质心垂向加速度 {report['released']['com_acc'][2]:+.2f} m/s^2"
        ),
        "结论：向上的净力把后轮压得更紧；把后轮抬起的力矩方向在摩擦锥内到不了离地，松开接触后整机也只受重力。",
    ])


class RearWheelLiftoffTest(unittest.TestCase):
    def test_frame_pitch_does_not_launch_off_the_rear_wheels(self):
        report = evaluate()
        print(format_report(report))
        check(report)


if __name__ == "__main__":
    unittest.main()
