"""Mean-preserving Beta variance ceiling; sampling and PPO use the same density."""
import math

import torch
from rsl_rl.modules import BetaDistribution


def std_ceiling_at(schedule, completed_updates):
    """Update-boundary curriculum, independent of terrain truth or episode stage."""
    start, end = schedule['start_update'], schedule['end_update']
    if completed_updates <= start:
        return 1.  # Exact native distribution throughout initial exploration.
    fraction = min(1., (completed_updates-start)/(end-start))
    return math.exp((1-fraction)*math.log(1/math.sqrt(3.))
                    + fraction*math.log(schedule['final_std']))


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
