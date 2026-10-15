"""Opt-in bounded policy using RSL-RL's native Beta sampling/log-prob/KL contract."""
import math

import torch
from rsl_rl.modules.distribution import BetaDistribution


class PetBotBetaDistribution(BetaDistribution):
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


class PetBotRecoveryBetaDistribution(PetBotBetaDistribution):
    """Preserve learned mean; concentrate exploration near the recovery configuration."""
    def update(self, mlp_output):
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.
        beta = torch.nn.functional.softplus(beta_raw) + 1.
        # The actor recomputes this from raw observations for BOTH rollout and PPO minibatches.
        scale = self.recovery_scale
        concentration = alpha + beta
        multiplier = ((concentration + 1.) / scale.square() - 1.) / concentration
        self._alpha = alpha * multiplier
        self._beta = beta * multiplier
        # Near-zero exploration needs large concentrations: evaluate special functions in fp64.
        self._distribution = torch.distributions.Beta(self._alpha.double(), self._beta.double())

    def sample(self):
        return super().sample().float()

    def log_prob(self, outputs):
        return super().log_prob(outputs.double()).float()

    @property
    def entropy(self):
        return super().entropy.float()

    def kl_divergence(self, old_params, new_params):
        return super().kl_divergence(tuple(p.double() for p in old_params),
                                     tuple(p.double() for p in new_params)).float()


from rsl_rl.models import MLPModel


class PetBotRecoveryActor(MLPModel):
    """State gate uses only deployed observations; deterministic export remains native MLP."""
    def get_latent(self, obs, masks=None, hidden_state=None):
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        if raw.shape[-1] % 27:
            raise ValueError("PetBot actor requires 27-value observation frames")
        current = raw[..., -27:]
        upright = ((-current[..., 5] - .85) / .13).clamp(0., 1.)
        configuration = ((.6 - current[..., 6:11].abs().amax(-1)) / .48).clamp(0., 1.)
        gate = upright * configuration
        gate = gate.square() * (3. - 2. * gate)
        floor = raw.new_tensor([.001] * 5 + [.0005] * 2)
        self.distribution.recovery_scale = 1. - gate.unsqueeze(-1) * (1. - floor)
        return super().get_latent(obs, masks, hidden_state)


class PetBotResidualBetaDistribution(PetBotBetaDistribution):
    """Affine bounded residual about home; scale both mean displacement and noise.

    The scale is observation-only, identical during rollout and PPO recomputation.
    KL is invariant under this shared invertible transform, avoiding the enormous
    concentration-induced KL of the previous variance-only terminal gate.
    """
    def __init__(self, output_dim, initial_mean, initial_std):
        super().__init__(output_dim, initial_mean, initial_std)
        self.register_buffer('home', torch.tensor(initial_mean, dtype=torch.float32))

    def sample(self):
        return self.home + self.recovery_scale * (super().sample() - self.home)

    def log_prob(self, outputs):
        raw = self.home + (outputs-self.home)/self.recovery_scale
        return super().log_prob(raw) - self.recovery_scale.log().sum(-1)

    @property
    def mean(self):
        return self.home + self.recovery_scale * (super().mean-self.home)

    @property
    def std(self):
        return self.recovery_scale * super().std

    @property
    def entropy(self):
        return super().entropy + self.recovery_scale.log().sum(-1)

    def deterministic_output(self, mlp_output):
        return self.home + self.recovery_scale * (super().deterministic_output(mlp_output)-self.home)


class _ResidualDeterministic(torch.nn.Module):
    def __init__(self, base, home):
        super().__init__()
        self.base=base
        self.register_buffer('home',home.clone())
        self.register_buffer('floor',home.new_tensor([.001]*5+[.0005]*2))

    def forward(self, obs):
        current=obs[..., -27:]
        upright=((-current[...,5]-.85)/.13).clamp(0.,1.)
        configuration=((.6-current[...,6:11].abs().amax(-1))/.48).clamp(0.,1.)
        gate=upright*configuration
        gate=gate.square()*(3.-2.*gate)
        scale=1.-gate.unsqueeze(-1)*(1.-self.floor)
        return self.home+scale*(self.base(obs)-self.home)


class PetBotResidualActor(PetBotRecoveryActor):
    """RecoveryActor's raw-observation gate with an affine residual policy."""
    def as_jit(self):
        # Native export contains the untransformed Beta mean. Apply the same raw
        # observation gate outside it, before/independent of observation normalization.
        return _ResidualDeterministic(super().as_jit(),self.distribution.home)

    def as_onnx(self, verbose=False):
        raise NotImplementedError('Residual policy currently supports verified TorchScript export only')
