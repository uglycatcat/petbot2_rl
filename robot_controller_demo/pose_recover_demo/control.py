"""Passive, track, and recover, plus a stand-up sequence that then returns."""

from dataclasses import dataclass

import mujoco
import numpy as np

PASSIVE = "passive"
RECOVER = "recover"
TRACK = "track"
MODES = (PASSIVE, TRACK, RECOVER)
FRAME = "frame"
EAR = "ear"
WHEEL = "wheel"
BARS = (FRAME, EAR, WHEEL)

POSITION_JOINTS = ("head", "left_ear", "right_ear", "left_frame", "right_frame")
WHEEL_ACTUATORS = ("left_rear_wheel", "right_rear_wheel")
# Head has no bar. Each bar commands one pair with the same signed target.
BAR_INDEX = {EAR: (1, 2), FRAME: (3, 4)}

# Torque motors. Gains are high enough to arrive inside the 2 s window and are
# still clipped to each actuator's ctrlrange.
KP = np.array([8.0, 4.0, 4.0, 12.0, 12.0])
KD = np.array([0.35, 0.04, 0.04, 0.35, 0.35])


def smoothstep(u: float) -> tuple[float, float]:
    """Smoothstep value and ds/du. Speed is zero at both ends."""
    u = float(np.clip(u, 0.0, 1.0))
    value = u * u * (3.0 - 2.0 * u)
    slope = 6.0 * u * (1.0 - u)
    return value, slope


def _stand_kind(pitch: float, fallen: bool):
    """Upright poses only home. A fall picks the sequence from the pitch sign."""
    if not fallen:
        return "home"
    if pitch < 0.0:
        return "negative"
    return "positive"


def _arrive(start: float, end: float, elapsed: float, duration: float):
    value, slope = smoothstep(elapsed / duration)
    delta = end - start
    return start + delta * value, delta * (slope / duration)


def _as_pair(position, first: int, second: int):
    return (float(position[first]), float(position[second]))


def _clip_mean(values, limits) -> float:
    low, high = limits
    return float(np.clip(np.mean(values), low, high))


def blend(start: np.ndarray, home: np.ndarray, elapsed: float, duration: float):
    """Position and velocity of a ``duration``-second move from ``start`` to ``home``."""
    value, slope = smoothstep(elapsed / duration)
    delta = home - start
    return start + delta * value, delta * (slope / duration)


@dataclass(frozen=True)
class DemoState:
    mode: str
    progress: float
    rotation: np.ndarray
    rpy: np.ndarray
    tilt: float
    fallen: bool
    frame: tuple[float, float] | None
    ear: tuple[float, float] | None
    wheel: float | None
    manual: tuple[bool, bool, bool]
    dragging: bool
    standing: bool
    stand_total: float


class RecoveryController:
    """Three peer modes, plus a stand-up sequence that takes over and then returns.

    Passive releases every actuator. Recover homes the joints in 2 s. Track holds
    the three bar targets. Bars move only in Track. An upright pose homes every
    joint in 1 s. A fall with negative pitch uses the 4 s motion. A fall with
    positive or zero pitch runs its 4 s motion and then that same 4 s motion.
    """

    DURATION = 2.0
    STAND_HOME = 1.0
    STAND_NEGATIVE = 4.0
    STAND_POSITIVE = 8.0

    def __init__(self, model: mujoco.MjModel):
        self.qadr = np.array([model.joint(name).qposadr[0] for name in POSITION_JOINTS])
        self.dadr = np.array([model.joint(name).dofadr[0] for name in POSITION_JOINTS])
        self.acts = np.array([model.actuator(name).id for name in POSITION_JOINTS])
        self.wheels = np.array([model.actuator(name).id for name in WHEEL_ACTUATORS])
        self.lo = model.actuator_ctrlrange[self.acts, 0]
        self.hi = model.actuator_ctrlrange[self.acts, 1]
        self.home = np.array(model.qpos0[self.qadr], dtype=float)
        self.kp = KP
        self.kd = KD
        # Velocity actuators still brake when ctrl is 0. Passive has to drop the
        # servo gain and the velocity bias, then put them back for Recover.
        self._gain0 = np.array(model.actuator_gainprm[self.wheels, 0], dtype=float)
        self._bias0 = np.array(model.actuator_biasprm[self.wheels, 2], dtype=float)
        self._model = model
        self.mode = RECOVER
        self.q_start = self.home.copy()
        self.elapsed = self.DURATION
        self.limits = {
            FRAME: self._joint_range("left_frame"),
            EAR: self._joint_range("left_ear"),
            WHEEL: (
                float(model.actuator_ctrlrange[self.wheels[0], 0]),
                float(model.actuator_ctrlrange[self.wheels[0], 1]),
            ),
        }
        self.manual = {FRAME: None, EAR: None, WHEEL: None}
        self.head_hold = float(self.home[0])
        self.dragging = False
        self.standing = False
        self._saved_mode = RECOVER
        self._saved_manual = dict(self.manual)
        self._saved_head = self.head_hold
        self._stand_elapsed = 0.0
        self._stand_kind = None
        self._hold_q = None

    @property
    def progress(self) -> float:
        return float(np.clip(self.elapsed / self.DURATION, 0.0, 1.0))

    @property
    def any_manual(self) -> bool:
        return any(value is not None for value in self.manual.values())

    def set_mode(self, mode: str, data: mujoco.MjData, force: bool = False):
        if mode not in MODES:
            raise ValueError(mode)
        if self.standing:
            return
        if mode == self.mode and not force:
            return
        self.manual = {name: None for name in BARS}
        self.mode = mode
        if mode == PASSIVE:
            self._set_wheel_servo(False)
            self.elapsed = 0.0
            return
        if mode == TRACK:
            self._capture_track(data)
            self._set_wheel_servo(True)
            return
        self._set_wheel_servo(True)
        self.q_start = np.array(data.qpos[self.qadr], dtype=float)
        self.elapsed = 0.0

    def set_bar(self, channel: str, value: float):
        if self.mode != TRACK or self.standing or channel not in self.limits:
            return
        low, high = self.limits[channel]
        self.manual[channel] = float(np.clip(value, low, high))

    @property
    def stand_duration(self) -> float:
        if self._stand_kind == "positive":
            return self.STAND_POSITIVE
        if self._stand_kind == "home":
            return self.STAND_HOME
        return self.STAND_NEGATIVE

    def request_stand(self, data: mujoco.MjData, pitch: float, fallen: bool) -> bool:
        """Take over whichever mode is active. A bar drag refuses the click."""
        kind = _stand_kind(pitch, fallen)
        if self.dragging or self.standing:
            return False
        self._saved_mode = self.mode
        self._saved_manual = dict(self.manual)
        self._saved_head = self.head_hold
        self._stand_kind = kind
        self.standing = True
        self._stand_elapsed = 0.0
        self._hold_q = np.array(data.qpos[self.qadr], dtype=float)
        self._set_wheel_servo(True)
        return True

    def command_targets(self):
        """Bar readouts: ``(frame_pair, ear_pair, wheel)``. ``None`` means off."""
        if self.standing and self._hold_q is not None:
            position, _, wheel = self._stand_reference(self._stand_elapsed)
            return _as_pair(position, 3, 4), _as_pair(position, 1, 2), float(wheel)
        if self.mode == TRACK and self.manual[FRAME] is not None:
            return (
                (self.manual[FRAME], self.manual[FRAME]),
                (self.manual[EAR], self.manual[EAR]),
                self.manual[WHEEL],
            )
        if self.mode == RECOVER:
            position, _ = blend(self.q_start, self.home, self.elapsed, self.DURATION)
            return _as_pair(position, 3, 4), _as_pair(position, 1, 2), 0.0
        return None, None, None

    def step(self, data: mujoco.MjData, dt: float):
        if self.standing:
            self._step_stand(data, dt)
            return
        if self.mode == RECOVER:
            self.elapsed = min(self.elapsed + dt, self.DURATION)
            position, velocity = blend(self.q_start, self.home, self.elapsed, self.DURATION)
            self._track(data, np.arange(len(POSITION_JOINTS)), position, velocity)
            data.ctrl[self.wheels] = 0.0
            self._set_wheel_servo(True)
            return
        if self.mode == TRACK:
            self._set_wheel_servo(True)
            self._hold(data, (0,), self.head_hold)
            self._hold(data, BAR_INDEX[EAR], self.manual[EAR])
            self._hold(data, BAR_INDEX[FRAME], self.manual[FRAME])
            data.ctrl[self.wheels] = self.manual[WHEEL]
            return
        data.ctrl[:] = 0.0
        self._set_wheel_servo(False)

    def _capture_track(self, data: mujoco.MjData):
        current = np.array(data.qpos[self.qadr], dtype=float)
        self.head_hold = float(current[0])
        self.manual[EAR] = _clip_mean(current[1:3], self.limits[EAR])
        self.manual[FRAME] = _clip_mean(current[3:5], self.limits[FRAME])
        self.manual[WHEEL] = 0.0

    def _hold(self, data: mujoco.MjData, indexes, target: float):
        self._track(data, np.array(indexes), np.full(len(indexes), target), np.zeros(len(indexes)))

    def _track(self, data: mujoco.MjData, indexes, position, velocity):
        error = position - data.qpos[self.qadr[indexes]]
        torque = self.kp[indexes] * error + self.kd[indexes] * (velocity - data.qvel[self.dadr[indexes]])
        data.ctrl[self.acts[indexes]] = np.clip(torque, self.lo[indexes], self.hi[indexes])

    def _step_stand(self, data: mujoco.MjData, dt: float):
        position, velocity, wheel = self._stand_reference(self._stand_elapsed)
        self._track(data, np.arange(len(POSITION_JOINTS)), position, velocity)
        data.ctrl[self.wheels] = wheel
        limit = self.stand_duration
        self._stand_elapsed = min(self._stand_elapsed + dt, limit)
        if self._stand_elapsed >= limit:
            self._finish_stand(data)

    def _stand_reference(self, time: float):
        if self._stand_kind == "home":
            return self._home_prefix(min(time, 1.0))
        if self._stand_kind == "positive":
            return self._positive_reference(time)
        return self._negative_reference(time)

    def _home_prefix(self, time: float):
        """First second: every position joint returns home and the wheels stop."""
        position = np.zeros(5)
        velocity = np.zeros(5)
        for index in range(5):
            position[index], velocity[index] = _arrive(
                self._hold_q[index], self.home[index], time, 1.0)
        return position, velocity, 0.0

    def _negative_reference(self, time: float):
        """Fallen, negative pitch. Neck stays home after the opening second."""
        if time < 1.0:
            return self._home_prefix(time)
        return self._negative_pose(time - 1.0)

    def _negative_pose(self, pose_time: float):
        """Ears, frames, and wheels after the opening home second."""
        position = self.home.copy()
        velocity = np.zeros(5)
        if pose_time < 1.0:
            position[1], velocity[1] = _arrive(self.home[1], 1.0, pose_time, 1.0)
            position[2], velocity[2] = _arrive(self.home[2], 1.0, pose_time, 1.0)
            position[3], velocity[3] = _arrive(self.home[3], 1.2, pose_time, 1.0)
            position[4], velocity[4] = _arrive(self.home[4], 1.2, pose_time, 1.0)
            return position, velocity, 0.0
        if pose_time < 2.0:
            position[1], velocity[1] = _arrive(1.0, 0.3, pose_time - 1.0, 1.0)
            position[2], velocity[2] = _arrive(1.0, 0.3, pose_time - 1.0, 1.0)
            position[3:] = 1.2
            return position, velocity, 0.0
        position[1:3] = 0.3
        position[3:] = 1.2
        wheel = 5.0 if pose_time < 2.5 else -7.0
        return position, velocity, wheel

    def _positive_reference(self, time: float):
        """Fallen, pitch >= 0. The frame motion, then the negative-pitch motion."""
        if time < 1.0:
            return self._home_prefix(time)
        if time < 4.0:
            return self._positive_frames(time - 1.0)
        follow = time - 4.0
        if follow < 1.0:
            position = self.home.copy()
            velocity = np.zeros(5)
            position[3], velocity[3] = _arrive(-0.4, self.home[3], follow, 1.0)
            position[4], velocity[4] = _arrive(-0.4, self.home[4], follow, 1.0)
            return position, velocity, 0.0
        return self._negative_pose(follow - 1.0)

    def _positive_frames(self, pose_time: float):
        """Both frames to -1, the left frame to +1, then both frames to -0.4."""
        position = self.home.copy()
        velocity = np.zeros(5)
        if pose_time < 1.0:
            position[3], velocity[3] = _arrive(self.home[3], -1.0, pose_time, 1.0)
            position[4], velocity[4] = _arrive(self.home[4], -1.0, pose_time, 1.0)
            return position, velocity, 0.0
        if pose_time < 2.0:
            position[3], velocity[3] = _arrive(-1.0, 1.0, pose_time - 1.0, 1.0)
            position[4] = -1.0
            return position, velocity, 0.0
        position[3], velocity[3] = _arrive(1.0, -0.4, pose_time - 2.0, 1.0)
        position[4], velocity[4] = _arrive(-1.0, -0.4, pose_time - 2.0, 1.0)
        return position, velocity, 0.0

    def _finish_stand(self, data: mujoco.MjData):
        mode = self._saved_mode
        self.standing = False
        if mode == TRACK:
            self.mode = TRACK
            self.manual = dict(self._saved_manual)
            self.head_hold = self._saved_head
            self._set_wheel_servo(True)
            return
        self.set_mode(mode, data, force=True)

    def _joint_range(self, name: str):
        low, high = self._model.jnt_range[self._model.joint(name).id]
        return float(low), float(high)

    def _set_wheel_servo(self, enabled: bool):
        self._model.actuator_gainprm[self.wheels, 0] = self._gain0 if enabled else 0.0
        self._model.actuator_biasprm[self.wheels, 2] = self._bias0 if enabled else 0.0
