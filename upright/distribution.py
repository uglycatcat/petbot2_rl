"""Opt-in bounded policy using RSL-RL's native Beta sampling/log-prob/KL contract."""
import math

import torch
from rsl_rl.modules.distribution import BetaDistribution


class UprightBetaDistribution(BetaDistribution):
    """State-dependent spread, initialized around the physical default configuration.

    No environment-side Gaussian clipping mismatch and no uniform exploration floor.
    Learning efficacy is deliberately separate from these numerical contracts.
    """
    def __init__(self, output_dim, initial_mean, initial_std):
        super().__init__(output_dim, action_range=(-1.0, 1.0))
        mean = torch.as_tensor(initial_mean, dtype=torch.float32)
        std = torch.as_tensor(initial_std, dtype=torch.float32)
        if mean.shape != (output_dim,) or std.shape != (output_dim,):
            raise ValueError("One initial mean/std required per action channel")
        if not ((mean.abs() < 1).all() and (std > 0).all()):
            raise ValueError("Invalid initial bounded action distribution")
        unit_mean = (mean+1)/2
        concentration = unit_mean*(1-unit_mean)/(std/2).square()-1
        alpha = unit_mean*concentration
        beta = (1-unit_mean)*concentration
        if not ((alpha > 1).all() and (beta > 1).all()):
            raise ValueError("Initial mean/std must describe a unimodal Beta distribution")
        # Stable inverse of softplus, undoing native Beta's +1 offset.
        parameter = torch.cat((alpha-1, beta-1))
        self.register_buffer("initial_bias", parameter + torch.log(-torch.expm1(-parameter)))

    def init_mlp_weights(self, mlp):
        last = [layer for layer in mlp.modules() if isinstance(layer, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(self.initial_bias.to(last.bias))

    @property
    def entropy(self):
        # Upstream entropy omits the constant Jacobian for rescaling [0,1] -> [-1,1].
        return super().entropy + self.output_dim * math.log(2.)
