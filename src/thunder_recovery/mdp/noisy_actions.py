"""论文 §IV-A 的动作噪声（action noise, σ = 0.02）。

论文原文（§IV-A Simulation Setup）：
    "Additionally, action noise characterized by a zero mean and a standard
     deviation of 0.02 is incorporated."

实现方式：继承 Isaac Lab 的位置/速度动作项，在 ``process_actions`` 里对**策略输出的
原始动作**叠加零均值高斯噪声（σ=0.02），再交给父类做 scale + default_offset + clip。
这样噪声作用在"动作"上，与论文表述一致；自由落体阶段（零力矩）由
``zero_action_freefall`` 事件直接钉住 PD 目标，不受本项影响。
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointVelocityAction
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    JointVelocityActionCfg,
)
from isaaclab.utils import configclass

ACTION_NOISE_STD = 0.02  # 论文 §IV-A


class _NoisyActionMixin:
    """在动作上叠加零均值高斯噪声（σ 见 cfg.action_noise_std）。"""

    cfg: object

    def process_actions(self, actions: torch.Tensor) -> None:
        std = float(getattr(self.cfg, "action_noise_std", 0.0))
        if std > 0.0:
            actions = actions + torch.randn_like(actions) * std
        super().process_actions(actions)


class NoisyJointPositionAction(_NoisyActionMixin, JointPositionAction):
    """位置动作 + 论文动作噪声。"""


class NoisyJointVelocityAction(_NoisyActionMixin, JointVelocityAction):
    """速度（轮）动作 + 论文动作噪声。"""


@configclass
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = NoisyJointPositionAction
    action_noise_std: float = ACTION_NOISE_STD


@configclass
class NoisyJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = NoisyJointVelocityAction
    action_noise_std: float = ACTION_NOISE_STD
