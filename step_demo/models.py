"""RSL-RL models with fixed scale for bounded, rare task-memory inputs."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization


class SensorMemoryNormalization(EmpiricalNormalization):
    """Normalize 26 sensor fields; keep the three [0,1] memory fields as supplied.

    Statistics remain in the usual RSL-RL state dict for exact same-run resume.
    Memory statistics are recorded but never divide or center these inputs.
    """

    def __init__(self, shape):
        if shape != 29:
            raise ValueError('Sensor-memory normalization requires 26+3 policy inputs')
        super().__init__(shape)

    def forward(self, x):
        normalized = super().forward(x)
        return torch.cat((normalized[..., :26], x[..., 26:]), dim=-1)


class SensorMemoryMLP(MLPModel):
    """Use the native RSL-RL MLP/distribution, with bounded memory coordinates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.obs_normalization or self.obs_groups != ['policy']:
            raise ValueError('Sensor-memory model requires normalized policy-only inputs')
        self.obs_normalizer = SensorMemoryNormalization(self.obs_dim)
