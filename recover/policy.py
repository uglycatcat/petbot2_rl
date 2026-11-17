"""Four independent actor/critics evaluated with batched matrix products."""
import math
import torch
from torch import nn
from .contract import OBS_DIM, ACTION_DIM, POSITION_LOW, POSITION_HIGH


class GroupLinear(nn.Module):
    def __init__(self, groups, ins, outs, gain=math.sqrt(2)):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(groups,ins,outs))
        self.bias=nn.Parameter(torch.zeros(groups,1,outs))
        for w in self.weight: nn.init.orthogonal_(w,gain=gain)

    def forward(self,x): return torch.bmm(x,self.weight)+self.bias


class Policy(nn.Module):
    def __init__(self, groups=4, width=128, action_mode='absolute', noise_rho=0.):
        super().__init__()
        if not 0 <= noise_rho < 1:
            raise ValueError('noise_rho must be in [0, 1)')
        if noise_rho and action_mode != 'absolute':
            raise ValueError('Correlated exploration currently requires absolute targets')
        self.noise_rho=float(noise_rho)
        self.groups=groups; self.width=width
        self.actor=nn.Sequential(GroupLinear(groups,OBS_DIM,width),nn.ELU(),
                                 GroupLinear(groups,width,width),nn.ELU(),GroupLinear(groups,width,ACTION_DIM,.01))
        self.critic=nn.Sequential(GroupLinear(groups,OBS_DIM,width),nn.ELU(),
                                  GroupLinear(groups,width,width),nn.ELU(),GroupLinear(groups,width,1,1.))
        self.log_std=nn.Parameter(torch.full((groups,1,ACTION_DIM),-.2))
        low=torch.tensor(POSITION_LOW); high=torch.tensor(POSITION_HIGH)
        home=2*(0-low)/(high-low)-1
        center=torch.zeros(1,1,ACTION_DIM); center[:,:,:5]=torch.atanh(home)
        self.register_buffer('action_center',center,persistent=False)
        if action_mode=='absolute':
            with torch.no_grad(): self.actor[-1].bias[:,:, :5].copy_(torch.atanh(home).reshape(1,1,5))
        elif action_mode!='delta': raise ValueError(action_mode)

    def mean(self,obs):
        mean=self.actor(obs)
        if self.noise_rho:
            # Last applied absolute target is already part of the observation.
            # This is a conditional Markov distribution, so PPO evaluates the
            # same likelihood used to collect each action (no hidden noise state).
            previous=torch.atanh(obs[:,:,18:25].clamp(-.99999,.99999))
            mean=mean+self.noise_rho*(previous-self.action_center)
        return mean

    def distribution(self,obs):
        scale=self.log_std.clamp(-2.5,.5).exp()*math.sqrt(1-self.noise_rho**2)
        return torch.distributions.Normal(self.mean(obs),scale)

    def value(self,obs): return self.critic(obs).squeeze(-1)

    def sample(self,obs):
        dist=self.distribution(obs); raw=dist.sample()
        return raw,dist.log_prob(raw).sum(-1),self.value(obs)

    def evaluate(self,obs,raw):
        dist=self.distribution(obs)
        # PPO ratios use pre-tanh actions: the common bijection Jacobian cancels.
        # Entropy includes its Jacobian so noise cannot grow by saturating targets.
        sample=dist.rsample()
        log_jac=2*(math.log(2)-sample-torch.nn.functional.softplus(-2*sample))
        entropy=(dist.entropy()+log_jac).sum(-1)
        return dist.log_prob(raw).sum(-1),entropy,self.value(obs)

    def act(self,obs): return torch.tanh(self.mean(obs))
