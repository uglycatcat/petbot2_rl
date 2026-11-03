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
