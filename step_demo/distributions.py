"""Mean-preserving Beta variance ceiling; sampling and PPO use the same density."""
import math

import torch
from rsl_rl.modules import BetaDistribution, GaussianDistribution


class StopCenteredGaussianDistribution(GaussianDistribution):
    """Auxiliary-only local exploration around the known default STOP command.

    All network parameters remain trainable. PPO stores the raw Gaussian
    sample/density; the environment applies its existing [-1,1] actuator
    clipping. This is an initialization, never a state-dependent override.
    Native bounded std prevents the old unconstrained-entropy runaway.
    """
    def __init__(self, output_dim, init_std=.03):
        if output_dim != 2 or not .005 <= init_std <= .05:
            raise ValueError('STOP Gaussian requires two actions and local initial noise')
        super().__init__(output_dim, init_std=init_std, std_type='log',
                         std_range=(.005, .05))

    def init_mlp_weights(self, mlp):
        head = [layer for layer in mlp.modules() if isinstance(layer, torch.nn.Linear)][-1]
        if head.out_features != 2:
            raise ValueError('STOP actor must produce two unconstrained means')
        torch.nn.init.normal_(head.weight, mean=0., std=.001)
        torch.nn.init.zeros_(head.bias)
        with torch.no_grad():
            head.bias[1] = -1.  # Existing map: normalized -1 is wheel target 0.


class FullStepGaussianDistribution(StopCenteredGaussianDistribution):
    """Same zero-speed-accessible mean, with bounded exploration for climbing.

    This fresh full-task actor has no phase switch or auxiliary parameters.
    Wider initial noise allows nonzero wheel and lifting requests. PPO may
    learn every mean and standard deviation, within a finite noise range.
    """
    def __init__(self, output_dim, init_std=.35):
        if output_dim != 2 or not .05 <= init_std <= .5:
            raise ValueError('Full-step Gaussian requires two actions and climbing exploration')
        GaussianDistribution.__init__(self, output_dim, init_std=init_std,
                                      std_type='log', std_range=(.005, .5))


class FiniteStepGaussianDistribution(FullStepGaussianDistribution):
    """Fresh full-task exploration starts with modest forward motion.

    The -0.5 wheel mean maps to 6.25 rad/s; it is only an initialization.
    There is no phase-dependent controller, frozen head, or auxiliary state.
    Raw Gaussian means can still reach/past -1 for exact zero wheel requests.
    """
    def __init__(self, output_dim, init_std=.2):
        super().__init__(output_dim, init_std)

    def init_mlp_weights(self, mlp):
        super().init_mlp_weights(mlp)
        head = [layer for layer in mlp.modules() if isinstance(layer, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.bias[1] = -.5


def std_ceiling_at(schedule, completed_updates):
    """Update-boundary curriculum, independent of terrain truth or episode stage."""
    start, end = schedule['start_update'], schedule['end_update']
    finals = [schedule['final_std']]
    if 'wheel_final_std' in schedule:
        finals.append(schedule['wheel_final_std'])
    if completed_updates <= start:
        return [1., 1.] if len(finals) == 2 else 1.
    fraction = min(1., (completed_updates-start)/(end-start))
    values = [math.exp((1-fraction)*math.log(1/math.sqrt(3.))
                      + fraction*math.log(final)) for final in finals]
    return values if len(values) == 2 else values[0]


class CeilingBetaDistribution(BetaDistribution):
    def __init__(self, output_dim, action_range=(-1., 1.)):
        super().__init__(output_dim, action_range)
        if action_range != (-1., 1.):
            raise ValueError('Step variance ceiling is defined on normalized [-1,1] actions')
        self.register_buffer('std_ceiling', torch.tensor(1.))

    def set_std_ceiling(self, value):
        if not math.isfinite(value) or not .005 <= value <= 1.:
            raise ValueError('Invalid normalized action std ceiling')
        self.std_ceiling.fill_(value)

    def update(self, mlp_output):
        super().update(mlp_output)
        total = self._alpha+self._beta
        mean = self._alpha/total
        # Normalized variance = 4*mean*(1-mean)/(total+1). Raising both
        # concentrations equally preserves the mean and unimodal support.
        required = 4.*mean*(1.-mean)/self.std_ceiling.square()-1.
        scale = torch.maximum(total, required)/total
        self._alpha = self._alpha*scale
        self._beta = self._beta*scale
        self._distribution = torch.distributions.Beta(self._alpha, self._beta)


class ActionwiseCeilingBetaDistribution(CeilingBetaDistribution):
    """Independent hip/wheel ceilings, with unchanged native means and density."""
    def __init__(self, output_dim, action_range=(-1., 1.)):
        if output_dim != 2:
            raise ValueError('Step action ceilings require common hip and wheel actions')
        super().__init__(output_dim, action_range)
        self.std_ceiling = torch.ones(output_dim)

    def set_std_ceiling(self, value):
        value = torch.as_tensor(value, dtype=self.std_ceiling.dtype, device=self.std_ceiling.device)
        if (value.shape != self.std_ceiling.shape or not torch.isfinite(value).all()
                or (value < .005).any() or (value > 1.).any()):
            raise ValueError('Invalid per-action normalized std ceilings')
        self.std_ceiling.copy_(value)


class StopReachableBetaDistribution(ActionwiseCeilingBetaDistribution):
    """Round14 Beta with wheel latent support [-1.1,1], hip unchanged.

    Existing environment clipping maps the short extra interval to zero speed.
    Raw samples, densities, entropy, KL, and PPO storage use this SAME affine
    distribution. Physical limits, slew, and observation mapping are unchanged.
    """
    def __init__(self, output_dim, action_range=(-1., 1.)):
        super().__init__(output_dim, action_range)
        self.register_buffer('_range_scale_tensor', torch.tensor([2., 2.1]))
        self.register_buffer('_range_offset_tensor', torch.tensor([-1., -1.1]))

    def update(self, mlp_output):
        BetaDistribution.update(self, mlp_output)
        total = self._alpha+self._beta
        mean = self._alpha/total
        required = self._range_scale_tensor.square()*mean*(1.-mean)/self.std_ceiling.square()-1.
        scale = torch.maximum(total, required)/total
        self._alpha = self._alpha*scale
        self._beta = self._beta*scale
        self._distribution = torch.distributions.Beta(self._alpha, self._beta)

    def sample(self):
        return self._distribution.sample()*self._range_scale_tensor+self._range_offset_tensor

    def deterministic_output(self, mlp_output):
        alpha, beta = torch.nn.functional.softplus(mlp_output).unbind(-2)
        return (alpha+1.)/(alpha+beta+2.)*self._range_scale_tensor+self._range_offset_tensor

    @property
    def mean(self):
        return self._alpha/(self._alpha+self._beta)*self._range_scale_tensor+self._range_offset_tensor

    @property
    def std(self):
        return self._distribution.stddev*self._range_scale_tensor

    @property
    def entropy(self):
        return (self._distribution.entropy()+self._range_scale_tensor.log()).sum(-1)

    def log_prob(self, outputs):
        raw = ((outputs-self._range_offset_tensor)/self._range_scale_tensor).clamp(1e-6, 1.-1e-6)
        return (self._distribution.log_prob(raw)-self._range_scale_tensor.log()).sum(-1)

    def as_deterministic_output_module(self):
        return _StopReachableBetaMean().to(self._range_scale_tensor.device)


class _StopReachableBetaMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor([2., 2.1]))
        self.register_buffer('offset', torch.tensor([-1., -1.1]))

    def forward(self, x):
        alpha, beta = torch.nn.functional.softplus(x).unbind(-2)
        return (alpha+1.)/(alpha+beta+2.)*self.scale+self.offset
