"""Train only an initial lift correction; retain the complete balance controller."""
import copy
import torch
from rsl_rl.utils import unpad_trajectories
from .balance_adapter import BalanceAdapterInference, balance_gate
from .balance_reflex import ReflexAdapterModel
from .distribution import BoundedBetaMean, parameters

RISE_CONTRACT = dict(version='frozen_cycle_rise_adapter_v1', limits=[.15, .25],
    ramp_seconds=.12, fade_balance_seconds=.6, concentration_factor=.25,
    channels=['frame_position', 'common_wheel_speed'],
    frozen='all inherited actor, balance/yaw adapters, normalization and inner feedback',
    observation_stats='frozen_parent_critic', gae_lambda=.95)


def rise_gate(raw: torch.Tensor) -> torch.Tensor:
    last = raw[:, -58:]
    seconds = last[:, 48]*4.
    lift = (last[:, 42]>.5).to(raw.dtype)*(seconds/.12).clamp(0., 1.)
    fade = (last[:, 43]>.5).to(raw.dtype)*(1.-seconds/.6).clamp(0., 1.)
    return lift+fade


class RiseAdapterModel(ReflexAdapterModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.register_buffer('rise_mean', torch.zeros(1, 290))
        self.register_buffer('rise_std', torch.ones(1, 290))
        self.register_buffer('rise_scale', torch.tensor(RISE_CONTRACT['limits']))
        self.rise_adapter = torch.nn.Sequential(torch.nn.Linear(290, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 64), torch.nn.ELU(), torch.nn.Linear(64, 2))
        torch.nn.init.zeros_(self.rise_adapter[-1].weight)
        torch.nn.init.zeros_(self.rise_adapter[-1].bias)

    def initialize_rise(self, parent, critic):
        state = self.state_dict()
        assert set(state)-set(parent) == {k for k in state if k.startswith('rise_')}
        assert not set(parent)-set(state)
        for key, value in parent.items():
            assert value.shape == state[key].shape, key
            state[key] = value.clone()
        state['rise_mean'] = critic['obs_normalizer._mean'].clone()
        state['rise_std'] = critic['obs_normalizer._std'].clone()
        self.load_state_dict(state, strict=True)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        raw = torch.cat([obs[group] for group in self.obs_groups], -1)
        parent = super().forward(obs, stochastic_output=False)
        gate = rise_gate(raw)
        delta = torch.tanh(self.rise_adapter((raw-self.rise_mean)/(self.rise_std+.01)))
        delta = delta*self.rise_scale*gate[:, None]
        delta = torch.cat((delta, torch.zeros_like(delta[:, :1])), -1)
        mean = (parent+delta).clamp(-.99, .99)
        if not stochastic_output:
            return mean
        # Exactly the inherited Beta transform outside the rise gate; only
        # frame/common concentrations are reduced for exploration during lift.
        logits = self.mlp(self.obs_normalizer(raw))
        original = BoundedBetaMean(127.)(logits)
        a, b = logits.unbind(-2); scale = a.new_tensor([4., 16., 16.])
        aa, bb = parameters(a, 127.)*scale, parameters(b, 127.)*scale
        shift = (mean-original)*.5*(aa+bb)
        channels = a.new_tensor([1., 1., 0.])
        factor = 1.+(self.balance_noise_factor-1.)*balance_gate(raw)[:, None]*channels
        factor = factor*(1.-.75*gate[:, None]*channels)
        self.distribution._alpha = (aa+shift)*factor
        self.distribution._beta = (bb-shift)*factor
        self.distribution._distribution = torch.distributions.Beta(self.distribution._alpha, self.distribution._beta)
        return self.distribution.sample()

    def as_jit(self):
        return RiseAdapterInference(self)


class RiseAdapterInference(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.parent = BalanceAdapterInference(actor)
        self.refinement = copy.deepcopy(actor.rise_adapter)
        self.register_buffer('mean', actor.rise_mean.clone())
        self.register_buffer('std', actor.rise_std.clone())
        self.register_buffer('scale', actor.rise_scale.clone())

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.refinement((raw-self.mean)/(self.std+.01)))
        delta = delta*self.scale*rise_gate(raw)[:, None]
        delta = torch.cat((delta, torch.zeros_like(delta[:, :1])), -1)
        return (self.parent(raw)+delta).clamp(-.99, .99)

    @torch.jit.export
    def reset(self):
        pass
