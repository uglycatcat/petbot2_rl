"""Portable inference modules; no simulator dependency."""
import copy
import torch
from torch import nn

OBSERVATION_SIZE = 40
ACTION_SIZE = 5

class RawPolicy(nn.Module):
    """Frozen observation normalization and deterministic actor mean."""
    def __init__(self, actor_critic):
        super().__init__()
        self.normalizer = copy.deepcopy(actor_critic.actor_obs_normalizer).cpu().eval()
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(observation.clamp(-20.0,20.0)))

class TargetPolicy(nn.Module):
    """Outputs head/left-frame/right-frame angles and left/right rear-wheel speeds."""
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.raw(observation))
        angles = torch.where(x[:,1:3]>=0.0,x[:,1:3]*1.2,x[:,1:3])
        return torch.cat((x[:,0:1]*1.3,angles,x[:,3:5]*25.0),dim=-1)


def load_actor(checkpoint, device='cpu'):
    """Reconstruct this project's feedforward RSL-RL policy for evaluation/export only."""
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=saved['infos']['config']
    obs=TensorDict({'policy':torch.zeros(1,OBSERVATION_SIZE)},batch_size=[1])
    ac=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},ACTION_SIZE,
        actor_obs_normalization=True,critic_obs_normalization=True,
        actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64],activation='elu',
        init_noise_std=config['initial_std'],noise_std_type='log')
    ac.load_state_dict(saved['model_state_dict'],strict=True)
    return ac.to(device).eval(),saved
