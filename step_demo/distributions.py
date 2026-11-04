"""Bounded raw PPO distribution; fixed clipping permits exact zero wheel speed."""
import torch
from rsl_rl.modules import BetaDistribution


class StepBetaDistribution(BetaDistribution):
    def __init__(self, output_dim, initial_mean=(0.,-.64), initial_std=(.03,.15)):
        if output_dim != 2:
            raise ValueError('STEP has exactly two shared actions')
        super().__init__(output_dim, action_range=(-1., 1.))
        self.register_buffer('scale', torch.tensor([2., 2.1]))
        self.register_buffer('offset', torch.tensor([-1., -1.1]))
        self.register_buffer('initial_mean', torch.tensor(initial_mean))
        self.register_buffer('initial_std', torch.tensor(initial_std))

    def init_mlp_weights(self, mlp):
        # Mean/concentration coordinates keep useful mean gradients even with
        # initially precise hip exploration. This is random initialization only.
        head = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        mean = (self.initial_mean-self.offset)/self.scale
        total = mean*(1.-mean)/(self.initial_std/self.scale).square()-1.
        concentration = total-2.
        mix = (mean*total-1.)/concentration
        if not ((concentration > 0).all() and (mix > 0).all() and (mix < 1).all()):
            raise ValueError('Initial mean/std must define a unimodal Beta')
        torch.nn.init.normal_(head.weight, std=.001)
        with torch.no_grad():
            head.bias[:2] = torch.logit(mix)
            head.bias[2:] = concentration+torch.log(-torch.expm1(-concentration))

    def update(self, mlp_output):
        self._alpha, self._beta = _parameters(mlp_output)
        self._distribution = torch.distributions.Beta(self._alpha, self._beta)

    def sample(self):
        return self._distribution.sample()*self.scale+self.offset

    def deterministic_output(self, mlp_output):
        alpha, beta = _parameters(mlp_output)
        return alpha/(alpha+beta)*self.scale+self.offset

    @property
    def mean(self):
        return self._alpha/(self._alpha+self._beta)*self.scale+self.offset

    @property
    def std(self):
        return self._distribution.stddev*self.scale

    @property
    def entropy(self):
        return (self._distribution.entropy()+self.scale.log()).sum(-1)

    def log_prob(self, outputs):
        eps = torch.finfo(outputs.dtype).eps
        raw = ((outputs-self.offset)/self.scale).clamp(eps, 1.-eps)
        return (self._distribution.log_prob(raw)-self.scale.log()).sum(-1)

    def as_deterministic_output_module(self):
        return _StepMean().to(self.scale.device)


class _StepMean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor([2., 2.1]))
        self.register_buffer('offset', torch.tensor([-1., -1.1]))

    def forward(self, x):
        alpha, beta = _parameters(x)
        return alpha/(alpha+beta)*self.scale+self.offset


def _parameters(x):
    raw_mean, raw_concentration = x.unbind(-2)
    mean = raw_mean.sigmoid()
    concentration = torch.nn.functional.softplus(raw_concentration)
    return 1.+mean*concentration, 1.+(1.-mean)*concentration
