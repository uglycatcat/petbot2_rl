"""Rear-wheel balance after a short forward roll.

The frames first pitch the body up, then fold so the front wheels rise while
the rear wheels stay down. Wheel torque then holds that inverted-pendulum
pose and trims away the leftover creep.
"""

import numpy as np

import mujoco


FRONT = ("left_front_wheel", "right_front_wheel")
REAR = ("left_rear_wheel", "right_rear_wheel")
HOLD_JOINTS = (
    ("head", 8.0, 0.35),
    ("left_ear", 4.0, 0.04),
    ("right_ear", 4.0, 0.04),
)

# Piecewise torque, 0.18 s each: symmetric frame torque, total rear-wheel torque.
# Found by search; the signs matter. Do not smooth across the boundaries.
_SCHEDULE = np.array(
    [
        [0.44273422, -0.56871577],
        [0.67190234, -0.10663510],
        [0.61312361, -0.57799277],
        [-0.85969167, 0.57291380],
        [-0.34191565, -0.50885738],
        [-0.10016566, 0.07068474],
    ]
)
_SEGMENT = 0.18
_FRAME = -0.40
_PITCH = 0.614
# Discrete LQR about the planted pose, scaled up slightly for the real entry.
_K = 1.15 * np.array([-4.32355269, -2.17754470, 41.85550149, 2.55137570])


def nose_up(qpos):
    """Nose-up angle in radians. Positive raises the head."""
    w, x, y, z = qpos[3:7]
    r20 = 2 * (x * z - w * y)
    r22 = 1 - 2 * (x * x + y * y)
    r21 = 2 * (y * z + w * x)
    return float(-np.arctan2(-r20, np.hypot(r21, r22)))


class JumpController:
    def __init__(self, model):
        self.model = model
        self.act = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        self.qadr = {}
        self.dadr = {}
        for name in ("head", "left_ear", "right_ear", "left_frame", "right_frame",
                     "left_rear_wheel", "right_rear_wheel"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qadr[name] = model.jnt_qposadr[jid]
            self.dadr[name] = model.jnt_dofadr[jid]
        self.front_ids = [model.body(name).id for name in FRONT]
        self.rear_ids = [model.body(name).id for name in REAR]
        self.kv = float(model.actuator_gainprm[self.act["left_rear_wheel"], 0])
        self.phase = "drive"
        self._phase_time = 0.0
        self._segment = 0
        self._segment_time = 0.0
        self._prev_pitch = None
        self._pitch_rate = 0.0
        self._anchor = None
        self._trim = 0.0
        self.frame_target = np.array([np.nan, np.nan])
        self.wheel_target = np.array([np.nan, np.nan])

    def wheel_height(self, data):
        front = min(float(data.xpos[i][2]) for i in self.front_ids)
        rear = min(float(data.xpos[i][2]) for i in self.rear_ids)
        return front, rear

    def step(self, data, dt):
        pitch = nose_up(data.qpos)
        if self._prev_pitch is None:
            rate = 0.0
        else:
            rate = (pitch - self._prev_pitch) / dt
        self._prev_pitch = pitch
        self._pitch_rate = 0.7 * self._pitch_rate + 0.3 * rate
        frame = float(data.qpos[self.qadr["left_frame"]])
        front_z, rear_z = self.wheel_height(data)
        x = float(data.qpos[0])
        vx = float(data.qvel[0])

        if self.phase == "drive":
            self._posture(data, 0.0, 14.0, 0.45)
            self._wheel_speed(data, 11.0)
            if self._phase_time >= 1.25:
                self._enter("brake")
        elif self.phase == "brake":
            self._posture(data, 0.0, 14.0, 0.45)
            self._wheel_speed(data, 0.0)
            if self._phase_time >= 0.55:
                self._enter("rear")
        elif self.phase == "rear":
            self._open_loop(data)
            if (
                front_z > 0.16
                and rear_z < 0.04
                and -0.45 < frame < -0.22
                and 0.75 < pitch < 1.0
            ):
                self._anchor = x
                self._enter("hold")
                self._balance(data, x, vx, pitch)
            elif self._segment >= len(_SCHEDULE) - 1 and self._segment_time >= _SEGMENT:
                self._anchor = x
                self._enter("hold")
                self._balance(data, x, vx, pitch)
        else:
            self._balance(data, x, vx, pitch)

        self._phase_time += dt
        if self.phase == "rear":
            self._segment_time += dt
            if self._segment_time >= _SEGMENT and self._segment < len(_SCHEDULE) - 1:
                self._segment += 1
                self._segment_time = 0.0

    def _enter(self, phase):
        self.phase = phase
        self._phase_time = 0.0

    def _open_loop(self, data):
        frame_torque, wheel_torque = _SCHEDULE[self._segment]
        self._posture(data, None, 8.0, 0.35)
        self._set_torque(data, "left_frame", frame_torque)
        self._set_torque(data, "right_frame", frame_torque)
        self._wheel_torque(data, wheel_torque)

    def _balance(self, data, x, vx, pitch):
        error = np.array(
            [x - self._anchor, vx, pitch - (_PITCH + self._trim), self._pitch_rate]
        )
        torque = float(np.clip(-(_K @ error), -0.55, 0.55))
        self._trim = float(np.clip(self._trim + 0.0015 * torque, -0.08, 0.08))
        if abs(vx) < 0.08:
            self._anchor = 0.995 * self._anchor + 0.005 * x
        self._posture(data, _FRAME, 32.0, 1.1)
        self._wheel_torque(data, torque)

    def _posture(self, data, frame, frame_kp, frame_kd):
        targets = {name: 0.0 for name, _, _ in HOLD_JOINTS}
        gains = {name: (kp, kd) for name, kp, kd in HOLD_JOINTS}
        if frame is None:
            self.frame_target[:] = np.nan
        else:
            self.frame_target[:] = frame
            targets["left_frame"] = frame
            targets["right_frame"] = frame
            gains["left_frame"] = (frame_kp, frame_kd)
            gains["right_frame"] = (frame_kp, frame_kd)
        for name, target in targets.items():
            kp, kd = gains[name]
            tau = kp * (target - data.qpos[self.qadr[name]]) - kd * data.qvel[self.dadr[name]]
            self._set_torque(data, name, tau)

    def _set_torque(self, data, name, torque):
        lo, hi = self.model.actuator_ctrlrange[self.act[name]]
        data.ctrl[self.act[name]] = float(np.clip(torque, lo, hi))

    def _wheel_speed(self, data, command):
        command = float(command)
        self.wheel_target[:] = command
        data.ctrl[self.act["left_rear_wheel"]] = command
        data.ctrl[self.act["right_rear_wheel"]] = command

    def _wheel_torque(self, data, torque):
        left = data.qvel[self.dadr["left_rear_wheel"]]
        right = data.qvel[self.dadr["right_rear_wheel"]]
        speed = 0.5 * (left + right)
        each = float(np.clip(torque / 2.0, -0.36, 0.36))
        command = float(np.clip(speed + each / self.kv, -25.0, 25.0))
        self._wheel_speed(data, command)
