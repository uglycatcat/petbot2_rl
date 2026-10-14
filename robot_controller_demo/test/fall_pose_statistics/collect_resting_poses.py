"""采集 PetBot 在被动沉降后的静止姿态。

执行器全部关闭，只保留重力、关节阻尼、摩擦和接触。大部分样本的初始朝向在
SO(3) 上均匀抽取；另外直接放在左侧躺、右侧躺、倒立、头朝下、头朝上，用来确认
这些朝向在重力下会不会自己倒进别的范围。最低点离地约 1 cm，初速度为 0。
六轴 IMU 在一段时间内不变，且加速度模长接近重力、身体与地面有接触，才记为停稳。
站着停住的样本保留。超时样本单独写出，不计入这一千套。

无窗口。不依赖 robot_controller_demo 里的键盘驾驶代码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parents[3] / "robot_description/mjcf/scene.xml"
POSITION_JOINTS = ("head", "left_ear", "right_ear", "left_frame", "right_frame")
PROBE_KINDS = ("left_side", "right_side", "inverted", "nose_down", "nose_up")

CLEARANCE_M = 0.01
STABLE_WINDOW_S = 0.50
MAX_SIM_S = 12.0
GYRO_ABS_MAX = 0.05
ACCEL_SPAN_MAX = 0.25
REST_ACCEL_MIN = 2.0
SEARCH_Z_MAX = 1.5

DEFINITIONS = {
    "tilt_rad": "身体 +Z 与世界 +Z 的夹角。0 为站立，pi/2 为放平，pi 为倒立。",
    "lean_azimuth_rad": "身体 +Z 在水平面上的朝向 atan2(y, x)。接近站立或倒立时为空。",
    "roll_pitch_yaw_rad": (
        "机体系到世界系的 ZYX 欧拉角，轴为 X 前、Y 左、Z 上。"
        "横滚接近 ±pi/2 是侧躺；俯仰接近 ±pi/2 是头朝下或头朝上，且该处欧拉角奇异。"
        "判断侧躺和倒立时以 tilt_rad 和 gravity_body 为准。"
    ),
    "gravity_body": "世界向下方向在机体系中的单位向量。",
    "sampling": (
        "均匀样本的初始四元数在 SO(3) 上均匀分布。定向样本分别把左侧、右侧、"
        "头顶、机头或机尾朝下，再绕世界竖直轴转过若干偏航角。"
        "位置关节从 0 释放。停稳后仍接近站立的样本保留。"
    ),
}


def haar_quat(rng: np.random.Generator) -> np.ndarray:
    """SO(3) 上的均匀随机四元数，MuJoCo 顺序 wxyz。"""
    u = rng.random()
    sigma1 = np.sqrt(1.0 - u)
    sigma2 = np.sqrt(u)
    theta1 = 2.0 * np.pi * rng.random()
    theta2 = 2.0 * np.pi * rng.random()
    quat = np.array([
        np.cos(theta2) * sigma2,
        np.sin(theta1) * sigma1,
        np.cos(theta1) * sigma1,
        np.sin(theta2) * sigma2,
    ])
    return quat / np.linalg.norm(quat)


def yaw_matrix(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def probe_matrix(kind: str, psi: float) -> np.ndarray:
    """机体系到世界系。定向姿态先摆好朝下的轴，再绕世界竖直轴旋转 psi。"""
    bases = {
        # 左侧贴地：机身 +Y 朝下。
        "left_side": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        # 右侧贴地：机身 +Y 朝上。
        "right_side": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        # 倒立：机身 +Z 朝下。
        "inverted": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        # 头朝下：机身 +X 朝下。俯仰 +pi/2。
        "nose_down": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        # 头朝上：机身 +X 朝上。俯仰 -pi/2。
        "nose_up": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    rotation = yaw_matrix(psi) @ bases[kind]
    if abs(np.linalg.det(rotation) - 1.0) > 1e-8:
        raise RuntimeError(f"{kind} 的旋转矩阵不是正当旋转")
    return rotation


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation).reshape(9))
    return quat / np.linalg.norm(quat)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    flat = np.zeros(9)
    mujoco.mju_quat2Mat(flat, np.ascontiguousarray(quat))
    return flat.reshape(3, 3)


def zyx_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """返回 roll, pitch, yaw。俯仰限制在 [-pi/2, pi/2]。"""
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def orientation_record(quat: np.ndarray, prefix: str) -> dict:
    rotation = matrix_from_quat(quat)
    roll, pitch, yaw = zyx_euler(rotation)
    body_z = rotation[:, 2]
    tilt = float(np.arccos(np.clip(body_z[2], -1.0, 1.0)))
    horizontal = float(np.hypot(body_z[0], body_z[1]))
    lean = None if horizontal < 1e-3 else float(np.arctan2(body_z[1], body_z[0]))
    gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
    return {
        f"{prefix}quat_wxyz": [float(v) for v in quat],
        f"{prefix}roll_rad": roll,
        f"{prefix}pitch_rad": pitch,
        f"{prefix}yaw_rad": yaw,
        f"{prefix}tilt_rad": tilt,
        f"{prefix}lean_azimuth_rad": lean,
        f"{prefix}gravity_body": [float(v) for v in gravity_body],
    }


def check_pose_definitions() -> None:
    left = probe_matrix("left_side", 0.9)
    right = probe_matrix("right_side", -0.4)
    inverted = probe_matrix("inverted", 1.2)
    if not np.allclose(left[:, 1], [0.0, 0.0, -1.0], atol=1e-8):
        raise RuntimeError("左侧躺没有让机身左侧朝下")
    if not np.allclose(right[:, 1], [0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError("右侧躺没有让机身右侧朝下")
    if not np.allclose(inverted[:, 2], [0.0, 0.0, -1.0], atol=1e-8):
        raise RuntimeError("倒立没有让机身 +Z 朝下")
    _, nose_down, _ = zyx_euler(probe_matrix("nose_down", 0.3))
    _, nose_up, _ = zyx_euler(probe_matrix("nose_up", 0.3))
    if abs(nose_down - np.pi / 2) > 1e-6 or abs(nose_up + np.pi / 2) > 1e-6:
        raise RuntimeError("头朝下/头朝上与 ZYX 俯仰不一致")
    quat = haar_quat(np.random.default_rng(0))
    if abs(np.linalg.norm(quat) - 1.0) > 1e-8:
        raise RuntimeError("均匀四元数没有单位化")


def make_plan(rng: np.random.Generator, n_uniform: int, probes_per_kind: int) -> list[dict]:
    yaws = np.linspace(0.0, 2.0 * np.pi, probes_per_kind, endpoint=False)
    plan = []
    for kind in PROBE_KINDS:
        for psi in yaws:
            plan.append({
                "kind": kind,
                "probe_yaw_rad": float(psi),
                "quat": quat_from_matrix(probe_matrix(kind, float(psi))),
            })
    for _ in range(n_uniform):
        plan.append({"kind": "uniform", "probe_yaw_rad": None, "quat": haar_quat(rng)})
    return plan


class RestingPoseSimulator:
    def __init__(self, scene: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        self.data = mujoco.MjData(self.model)
        free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if int(self.model.jnt_type[free_id]) != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("root 不是自由关节")
        if int(self.model.jnt_qposadr[free_id]) != 0:
            raise RuntimeError("自由关节不在 qpos 开头")
        self._assert_actuators_disabled()
        self.floor_id = self.model.geom("floor").id
        self.joint_addresses = np.array(
            [self.model.joint(name).qposadr[0] for name in POSITION_JOINTS])
        self.dt = float(self.model.opt.timestep)
        self.window_steps = max(1, round(STABLE_WINDOW_S / self.dt))
        self.max_steps = round(MAX_SIM_S / self.dt)
        accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.accel_adr = int(self.model.sensor_adr[accel_id])
        self.gyro_adr = int(self.model.sensor_adr[gyro_id])

    def _assert_actuators_disabled(self) -> None:
        """后轮速度执行器若仍生效，零指令会把轮速拉回 0。"""
        mujoco.mj_resetData(self.model, self.data)
        dof = int(self.model.jnt_dofadr[self.model.joint("left_rear_wheel").id])
        self.data.qvel[dof] = 10.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        if float(np.max(np.abs(self.data.qfrc_actuator))) > 1e-8:
            raise RuntimeError("执行器仍在出力，被动沉降不成立")
        mujoco.mj_resetData(self.model, self.data)

    def floor_contacts(self) -> list[str]:
        names = []
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if self.floor_id not in (geom1, geom2):
                continue
            other = geom2 if geom1 == self.floor_id else geom1
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            names.append(name or str(other))
        return sorted(set(names))

    def _write_pose(self, quat: np.ndarray, z: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = (0.0, 0.0, z)
        self.data.qpos[3:7] = quat
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _floor_penetrating(self, quat: np.ndarray, z: float) -> bool:
        self._write_pose(quat, z)
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            if self.floor_id in (geom1, geom2) and self.data.contact[index].dist < 0.0:
                return True
        return False

    def place(self, quat: np.ndarray) -> float:
        if self._floor_penetrating(quat, SEARCH_Z_MAX):
            raise RuntimeError("抬到搜索上限仍然与地面穿透")
        low, high = 0.0, SEARCH_Z_MAX
        for _ in range(28):
            mid = 0.5 * (low + high)
            if self._floor_penetrating(quat, mid):
                low = mid
            else:
                high = mid
        z = high + CLEARANCE_M
        self._write_pose(quat, z)
        if self._floor_penetrating(quat, z):
            raise RuntimeError("留出间隙后仍然与地面穿透")
        return z

    def _imu(self) -> np.ndarray:
        accel = self.data.sensordata[self.accel_adr:self.accel_adr + 3]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        return np.concatenate((accel, gyro))

    def _window_is_still(self, window: np.ndarray) -> bool:
        gyro = window[:, 3:]
        accel = window[:, :3]
        if float(np.max(np.abs(gyro))) > GYRO_ABS_MAX:
            return False
        span = accel.max(axis=0) - accel.min(axis=0)
        if float(np.max(span)) > ACCEL_SPAN_MAX:
            return False
        if float(np.linalg.norm(accel.mean(axis=0))) < REST_ACCEL_MIN:
            return False
        return bool(self.floor_contacts())

    def settle(self, quat: np.ndarray) -> dict:
        drop_z = self.place(quat)
        history: deque[np.ndarray] = deque(maxlen=self.window_steps)
        status = "timeout"
        steps = 0
        for steps in range(1, self.max_steps + 1):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.sensordata).all():
                status = "nonfinite"
                break
            history.append(self._imu())
            if len(history) == self.window_steps and self._window_is_still(np.stack(history)):
                status = "settled"
                break
        final_quat = np.array(self.data.qpos[3:7], dtype=float, copy=True)
        joints = {
            name: float(self.data.qpos[address])
            for name, address in zip(POSITION_JOINTS, self.joint_addresses)
        }
        return {
            "status": status,
            "drop_z": drop_z,
            "sim_time_s": steps * self.dt,
            "final_quat": final_quat,
            "joint_pos_rad": joints,
            "contacts": self.floor_contacts(),
            "lower_body_z": float(self.data.xpos[self.model.body("lower_body").id, 2]),
        }


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    keys = (0, 5, 25, 50, 75, 95, 100)
    return {f"p{key}": float(np.percentile(array, key)) for key in keys}


def histogram(values: list[float], edges: np.ndarray) -> list[dict]:
    if not values:
        return []
    counts, _ = np.histogram(values, bins=edges)
    rows = []
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        rows.append({"from": float(left), "to": float(right), "count": int(count)})
    return rows


def summarize(settled: list[dict], failed: list[dict], meta: dict) -> dict:
    by_kind: dict[str, dict] = {}
    for row in settled:
        slot = by_kind.setdefault(row["kind"], {"settled": 0, "final_tilt_deg": []})
        slot["settled"] += 1
        slot["final_tilt_deg"].append(np.degrees(row["tilt_rad"]))
    for row in failed:
        slot = by_kind.setdefault(row["kind"], {"settled": 0, "final_tilt_deg": []})
        slot[row["status"]] = slot.get(row["status"], 0) + 1
    kind_summary = {}
    for kind, slot in by_kind.items():
        tilts = slot.pop("final_tilt_deg")
        slot["final_tilt_deg"] = percentiles(tilts)
        kind_summary[kind] = slot

    def column(name: str) -> list[float]:
        return [float(row[name]) for row in settled]

    tilt_deg = [np.degrees(v) for v in column("tilt_rad")]
    roll_deg = [np.degrees(v) for v in column("roll_rad")]
    pitch_deg = [np.degrees(v) for v in column("pitch_rad")]
    uniform = [row for row in settled if row["kind"] == "uniform"]
    uniform_tilt = [np.degrees(row["tilt_rad"]) for row in uniform]
    return {
        "definitions": DEFINITIONS,
        "meta": meta,
        "n_settled": len(settled),
        "n_timeout": sum(row["status"] == "timeout" for row in failed),
        "n_nonfinite": sum(row["status"] == "nonfinite" for row in failed),
        "by_kind": kind_summary,
        "tilt_deg": percentiles(tilt_deg),
        "roll_deg": percentiles(roll_deg),
        "pitch_deg": percentiles(pitch_deg),
        "uniform_tilt_deg": percentiles(uniform_tilt),
        "tilt_deg_hist": histogram(tilt_deg, np.arange(0.0, 181.0, 10.0)),
        "roll_deg_hist": histogram(roll_deg, np.arange(-180.0, 181.0, 15.0)),
        "pitch_deg_hist": histogram(pitch_deg, np.arange(-90.0, 91.0, 10.0)),
    }


def record_from_trial(spec: dict, result: dict) -> dict:
    initial = orientation_record(spec["quat"], "")
    final = orientation_record(result["final_quat"], "")
    row = {
        "kind": spec["kind"],
        "probe_yaw_rad": spec["probe_yaw_rad"],
        "status": result["status"],
        "drop_z": result["drop_z"],
        "sim_time_s": result["sim_time_s"],
        "lower_body_z": result["lower_body_z"],
        "joint_pos_rad": result["joint_pos_rad"],
        "contacts": result["contacts"],
        "initial_quat_wxyz": initial["quat_wxyz"],
        "initial_roll_rad": initial["roll_rad"],
        "initial_pitch_rad": initial["pitch_rad"],
        "initial_yaw_rad": initial["yaw_rad"],
        "initial_tilt_rad": initial["tilt_rad"],
        "quat_wxyz": final["quat_wxyz"],
        "roll_rad": final["roll_rad"],
        "pitch_rad": final["pitch_rad"],
        "yaw_rad": final["yaw_rad"],
        "tilt_rad": final["tilt_rad"],
        "lean_azimuth_rad": final["lean_azimuth_rad"],
        "gravity_body": final["gravity_body"],
    }
    return row


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> int:
    check_pose_definitions()
    if not SCENE.is_file():
        print(f"找不到模型：{SCENE}", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settled_path = output / "resting_poses.jsonl"
    failed_path = output / "timeouts.jsonl"
    summary_path = output / "summary.json"
    settled_path.write_text("", encoding="utf-8")
    failed_path.write_text("", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    plan = make_plan(rng, args.uniform, args.probes_per_kind)
    target = len(plan)
    simulator = RestingPoseSimulator(SCENE)
    settled: list[dict] = []
    failed: list[dict] = []
    started = time.perf_counter()
    print(
        f"MuJoCo {mujoco.__version__} | 被动沉降 | 目标停稳 {target} "
        f"| 均匀 {args.uniform} | 每类定向 {args.probes_per_kind} | seed {args.seed}",
        flush=True,
    )

    def dump_summary() -> None:
        meta = {
            "seed": args.seed,
            "uniform": args.uniform,
            "probes_per_kind": args.probes_per_kind,
            "clearance_m": CLEARANCE_M,
            "stable_window_s": STABLE_WINDOW_S,
            "max_sim_s": MAX_SIM_S,
            "gyro_abs_max": GYRO_ABS_MAX,
            "accel_span_max": ACCEL_SPAN_MAX,
            "scene": str(SCENE),
            "mujoco": mujoco.__version__,
            "elapsed_s": time.perf_counter() - started,
        }
        write_json(summary_path, summarize(settled, failed, meta))

    attempt = 0
    max_attempts = target * 5
    spec_iter = iter(plan)
    extra = 0
    while len(settled) < target and attempt < max_attempts:
        spec = next(spec_iter, None)
        if spec is None:
            extra += 1
            spec = {"kind": "uniform_fill", "probe_yaw_rad": None, "quat": haar_quat(rng)}
        attempt += 1
        try:
            result = simulator.settle(spec["quat"])
        except RuntimeError as error:
            failed.append({"kind": spec["kind"], "status": "placement_error", "error": str(error)})
            print(f"放置失败 {spec['kind']}: {error}", flush=True)
            continue
        row = record_from_trial(spec, result)
        if result["status"] == "settled":
            row["index"] = len(settled)
            settled.append(row)
            with settled_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{len(settled)}/{target}] {row['kind']} "
                f"tilt {np.degrees(row['tilt_rad']):6.1f} deg "
                f"roll {np.degrees(row['roll_rad']):7.1f} "
                f"pitch {np.degrees(row['pitch_rad']):7.1f} "
                f"t={row['sim_time_s']:.2f}s "
                f"contact={','.join(row['contacts']) or '-'}",
                flush=True,
            )
        else:
            failed.append(row)
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[timeout {len(failed)}] {row['kind']} "
                f"tilt {np.degrees(row['tilt_rad']):6.1f} deg "
                f"after {row['sim_time_s']:.2f}s",
                flush=True,
            )
        if len(settled) % 25 == 0 or result["status"] != "settled":
            dump_summary()

    dump_summary()
    print(
        f"完成：停稳 {len(settled)}，超时 {sum(r['status']=='timeout' for r in failed)}，"
        f"补抽 {extra}，用时 {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(f"数据 {settled_path}", flush=True)
    print(f"汇总 {summary_path}", flush=True)
    return 0 if len(settled) == target else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无窗口采集 PetBot 被动沉降后的静止姿态")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uniform", type=int, default=900, help="均匀朝向样本数")
    parser.add_argument("--probes-per-kind", type=int, default=20, help="五类定向姿态各自的偏航采样数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="jsonl 与 summary.json 的目录",
    )
    args = parser.parse_args()
    if args.uniform < 0 or args.probes_per_kind < 0:
        parser.error("样本数不能为负")
    if args.uniform + args.probes_per_kind * len(PROBE_KINDS) == 0:
        parser.error("至少要有一个样本")
    return args


if __name__ == "__main__":
    try:
        sys.exit(collect(parse_args()))
    except KeyboardInterrupt:
        print("\n已中断，已写出的 jsonl 仍然保留。", file=sys.stderr)
        sys.exit(130)
