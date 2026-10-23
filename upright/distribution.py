"""Opt-in bounded policy using RSL-RL's native Beta sampling/log-prob/KL contract."""
import math

import torch
from rsl_rl.modules.distribution import BetaDistribution


class BoundedBetaMean(torch.nn.Module):
    def __init__(self, span: float = 31.):
        super().__init__(); self.span = span
    def forward(self, x):
        a,b=torch.unbind(x,dim=-2)
        a,b=parameters(a,self.span),parameters(b,self.span)
        return 2*a/(a+b)-1

def parameters(x, span: float = 31.):
    v=torch.nn.functional.softplus(x)
    return 1+span*v/(span+v)


class UprightBetaDistribution(BetaDistribution):
    """State-dependent spread, initialized around the physical default configuration.

    Bounded base alpha/beta and finite channel scales retain nonzero exploration.
    Scaling alpha and beta together preserves the deterministic mean.
    Sampling, likelihood, deterministic evaluation and export share the same transform.
    Learning efficacy is deliberately separate from these numerical contracts.
    """
    def __init__(self, output_dim, initial_mean, initial_std, shape_cap=32., concentration_scale=None):
        super().__init__(output_dim, action_range=(-1.0, 1.0))
        if shape_cap not in (32.,128.): raise ValueError("Unsupported Beta shape cap")
        self.span=float(shape_cap)-1.
        self.concentration_scale=tuple(concentration_scale or [1.]*output_dim)
        if len(self.concentration_scale)!=output_dim or any(not math.isfinite(x) or not 1<=x<=64 for x in self.concentration_scale):raise ValueError("Invalid concentration scale")
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
        if not (parameter<self.span).all(): raise ValueError("Initial Beta parameters exceed exploration bound")
        parameter=self.span*parameter/(self.span-parameter)
        self.register_buffer("initial_bias", parameter + torch.log(-torch.expm1(-parameter)))

    def init_mlp_weights(self, mlp):
        last = [layer for layer in mlp.modules() if isinstance(layer, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(self.initial_bias.to(last.bias))

    def update(self, mlp_output):
        a,b=torch.unbind(mlp_output,dim=-2)
        scale=a.new_tensor(self.concentration_scale)
        self._alpha,self._beta=parameters(a,self.span)*scale,parameters(b,self.span)*scale
        self._distribution=torch.distributions.Beta(self._alpha,self._beta)

    def deterministic_output(self, mlp_output):
        return BoundedBetaMean(self.span)(mlp_output)

    def as_deterministic_output_module(self):
        return BoundedBetaMean(self.span)

    @property
    def entropy(self):
        # Upstream entropy omits the constant Jacobian for rescaling [0,1] -> [-1,1].
        return super().entropy + self.output_dim * math.log(2.)
