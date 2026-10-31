"""Reviewed experiment choices, separate from immutable user requirements."""
from dataclasses import asdict, dataclass, fields, field
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Config:
    schema: str = 'upright_restart_round01_v1'
    series: str = 'upright_restart_20261030'
    round: int = 1
    seed: int = 101
    iterations: int = 1000
    num_envs: int = 2048
    decimation: int = 30  # source timestep 2ms -> policy 60ms
    history: int = 5
    rollout_steps: int = 32
    learning_rate: float = 0.0003
    gae_lambda: float | None = None  # None retains legacy physical-time scaling
    checkpoint_interval: int = 100
    probe_interval: int = 100
    bootstrap_probe_interval: int = 0
    capability_stages: list[list[float]] = field(default_factory=list)
    probe_count: int = 32
    curriculum_pass_rate: float = 0.8
    curriculum_consecutive: int = 2
    low_speed: float = 0.10
    low_yaw: float = 0.30
    initial_hip: float = 0.10
    baseline_fraction: float = 0.2
    boundary_command_fraction: float = 0.0  # training only, moving levels >= 3
    warmup_s: float = 1.5
    warmup_timeout_s: float = 4.0
    rise_timeout_s: float = 12.0
    support_enter_s: float = 0.30
    lifted_confirm_s: float = 0.06
    return_confirm_s: float = 0.12
    ground_confirm_s: float = 0.18
    hold_s: float = 2.0
    balance_train_s: float = 8.0
    short_cycle_fraction: float = 0.0  # training-only moving episode mixture
    short_balance_min_s: float = 4.0
    short_balance_max_s: float = 8.0
    balance_eval_s: float = 30.0
    balance_observation_age_cap_s: float = 0.0
    landing_timeout_s: float = 4.0
    handoff_hold_s: float = 0.6
    handoff_timeout_s: float = 3.0
    handoff_blend_s: float = 0.3
    lift_gap_m: float = 0.025
    support_gap_m: float = 0.075
    ground_gap_m: float = 0.012
    hip_slew_rad_s: float = 6.0
    wheel_slew_rad_s2: float = 150.0
    rise_hip_slew_rad_s: float = 6.0
    rise_wheel_slew_rad_s2: float = 150.0
    rise_prepare_weight: float = 0.0
    rise_partial_lift_weight: float = 0.0
    rise_support_weight: float = 0.0
    rise_prepare_decay_s: float = 2.0
    rise_front_return_penalty: float = 5.0
    balance_velocity_cost: float = 0.0
    balance_velocity_error_tau_s: float = 0.3
    balance_velocity_error_scale_m_s: float = 0.05
    landing_velocity_cost: float = 0.0
    landing_velocity_error_scale_m_s: float = 0.2
    balance_linear_accel_m_s2: float = 0.0  # zero disables reference slew for legacy configs
    balance_yaw_accel_rad_s2: float = 0.0
    balance_yaw_feedforward: float = 0.0
    balance_yaw_feedback: float = 0.0
    balance_yaw_correction_limit_rad_s: float = 4.0
    heading_kp: float = 1.0
    heading_kd: float = 0.1
    heading_wheel_correction_rad_s: float = 2.0
    yaw_quality_rad: float = 0.35
    imu_noise_std: float = 0.0
    joint_noise_std: float = 0.0

    def motion_stages(self):
        # Each row: two-wheel vx/yaw, four-wheel entry vx, common hip initial angle.
        return self.capability_stages or [[self.low_speed, self.low_yaw, self.low_speed, self.initial_hip]]

    @property
    def max_course_level(self):
        return len(self.motion_stages())+1

    def validate(self):
        for key in ('iterations', 'num_envs', 'decimation', 'history', 'rollout_steps',
                    'checkpoint_interval', 'probe_interval', 'probe_count', 'curriculum_consecutive'):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.bootstrap_probe_interval) is not int or self.bootstrap_probe_interval < 0:
            raise ValueError('Invalid bootstrap probe interval')
        if self.gae_lambda is not None and (type(self.gae_lambda) not in (int, float) or not 0 <= self.gae_lambda <= 1):
            raise ValueError('GAE lambda must be finite and within [0, 1]')
        if self.bootstrap_probe_interval and self.probe_interval % self.bootstrap_probe_interval:
            raise ValueError('Bootstrap probe interval must divide motion probe interval')
        previous = None
        for stage in self.motion_stages():
            if len(stage) != 4 or any(type(v) not in (int, float) or not __import__('math').isfinite(v) or v < 0 for v in stage):
                raise ValueError('Capability stage needs four finite nonnegative ranges')
            if stage[0] > .75 or stage[2] > .75 or stage[3] > 1:
                raise ValueError('Capability range exceeds source/required bounds')
            if previous and any(a < b for a, b in zip(stage, previous)):
                raise ValueError('Capability stages must not shrink; regression is sampled separately')
            previous = stage
        if self.motion_stages()[0] != [self.low_speed, self.low_yaw, self.low_speed, self.initial_hip]:
            raise ValueError('First capability stage must preserve the low-range baseline')
        if self.schema != 'upright_restart_round01_v1' or type(self.round) is not int or self.round < 1:
            raise ValueError('Unsupported configuration schema or invalid round number')
        if self.num_envs > 4096 or self.probe_count > self.num_envs:
            raise ValueError('Invalid environment/probe count; maximum 4096 environments')
        if not 0 < self.curriculum_pass_rate <= 1 or not 0 <= self.baseline_fraction < 1:
            raise ValueError('Invalid curriculum probabilities')
        if not 0 <= self.short_cycle_fraction < 1:
            raise ValueError('Short-cycle mixture must retain full-duration training episodes')
        if not 0 <= self.boundary_command_fraction < 1:
            raise ValueError('Boundary commands must retain continuous-range training commands')
        if self.short_cycle_fraction and not (self.hold_s <= self.short_balance_min_s <= self.short_balance_max_s < self.balance_train_s):
            raise ValueError('Short-cycle duration must retain genuine support and be below the long duration')
        if not 0 <= self.low_speed <= .75 or not 0 <= self.initial_hip <= 1:
            raise ValueError('Curriculum range exceeds source/required bounds')
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not __import__('math').isfinite(value) or value < 0):
                raise ValueError(f'Invalid numeric configuration: {key}')
        for key in ('hold_s', 'rise_timeout_s', 'landing_timeout_s', 'support_enter_s',
                    'handoff_blend_s', 'yaw_quality_rad', 'learning_rate', 'rise_prepare_decay_s',
                    'balance_velocity_error_tau_s', 'balance_velocity_error_scale_m_s', 'landing_velocity_error_scale_m_s'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None, **overrides):
    data = json.loads(Path(path or ROOT/'configs/round11.json').read_text())
    unknown = data.keys() - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data).validate()


def require_execution(kind):
    state = json.loads((ROOT/'project.json').read_text())
    key = {'training': 'training_enabled', 'engineering': 'engineering_execution_enabled',
           'evaluation': 'evaluation_enabled'}[kind]
    if not state.get(key, False):
        raise RuntimeError(f'{kind} is not enabled by the current authorization/engineering state. No simulator or training was started.')
    require_gpu_isolation()


def require_gpu_isolation():
    """The container exposes one physical GPU, selected from the user-owned 2/3 allocation."""
    host_gpu = os.environ.get('PETBOT_HOST_GPU')
    uuid = os.environ.get('PETBOT_GPU_UUID', '')
    if host_gpu not in ('2', '3') or not uuid.startswith('GPU-'):
        raise RuntimeError('upright may only execute in a dedicated physical GPU2/GPU3 container')
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') != uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Single-GPU Docker isolation is required, including renderer visibility')
    visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    if visible.strip().splitlines() != [uuid]:
        raise RuntimeError('Visible physical GPUs differ from the assigned single GPU')
    return dict(host_gpu=int(host_gpu), uuid=uuid, logical_device='cuda:0')
