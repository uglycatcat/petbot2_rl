"""Frozen full-cycle actor with a bounded, phase-gated balance refinement head."""
import copy
import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from .distribution import parameters, BoundedBetaMean


def balance_gate(raw: torch.Tensor) -> torch.Tensor:
    # Last frame of the existing58x5 observation: phase2 one-hot43, clock48/4.
    last=raw[:, -58:]
    return (last[:,43]>.5).to(raw.dtype)*((last[:,48]*4.-1.2)/.6).clamp(0.,1.)


class BalanceAdapterModel(MLPModel):
    def __init__(self,*args,balance_noise_factor=1.,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.obs_dim==290 and self.distribution.concentration_scale==(4.,16.,16.)
        assert balance_noise_factor in (1.,16.)
        self.balance_noise_factor=float(balance_noise_factor)
        for parameter in self.mlp.parameters():parameter.requires_grad_(False)
        self.register_buffer('adapter_mean',torch.zeros(1,290))
        self.register_buffer('adapter_std',torch.ones(1,290))
        self.register_buffer('adapter_scale',torch.tensor([.03,.08]))
        self.adapter=torch.nn.Sequential(torch.nn.Linear(290,64),torch.nn.ELU(),
            torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,2))
        torch.nn.init.zeros_(self.adapter[-1].weight);torch.nn.init.zeros_(self.adapter[-1].bias)
        self.yaw_adapter=torch.nn.Sequential(torch.nn.Linear(290,64),torch.nn.ELU(),
            torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.yaw_adapter[-1].weight);torch.nn.init.zeros_(self.yaw_adapter[-1].bias)
        self.register_buffer('yaw_adapter_scale',torch.tensor(.8))

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        raw=torch.cat([obs[group] for group in self.obs_groups],-1)
        logits=self.mlp(self.obs_normalizer(raw))
        original=BoundedBetaMean(127.)(logits)
        gate=balance_gate(raw)
        features=(raw-self.adapter_mean)/(self.adapter_std+.01)
        refinement=torch.tanh(self.adapter(features))
        delta=refinement*self.adapter_scale*gate[:,None]
        yaw_delta=torch.tanh(self.yaw_adapter(features))*self.yaw_adapter_scale*gate[:,None]
        delta=torch.cat((delta,yaw_delta),-1)
        mean=(original+delta).clamp(-.99,.99)
        if not stochastic_output:return mean
        a,b=logits.unbind(-2);scale=a.new_tensor([4.,16.,16.])
        aa,bb=parameters(a,127.)*scale,parameters(b,127.)*scale
        # Add equal/opposite masses: zero refinement preserves both old shapes
        # exactly; changing the mean leaves concentration (exploration) explicit.
        shift=(mean-original)*.5*(aa+bb)
        factor=1.+(self.balance_noise_factor-1.)*gate[:,None]*a.new_tensor([1.,1.,0.])
        self.distribution._alpha=(aa+shift)*factor
        self.distribution._beta=(bb-shift)*factor
        self.distribution._distribution=torch.distributions.Beta(self.distribution._alpha,self.distribution._beta)
        return self.distribution.sample()

    def update_normalization(self,obs):
        # Both coordinate systems are frozen. Updating the inherited statistics
        # would change the supposedly frozen base actor during optimization.
        pass

    def as_jit(self):return BalanceAdapterInference(self)

    def as_onnx(self,verbose=False):
        raise NotImplementedError('Use the validated JIT adapter export; ONNX is not implemented')


class BalanceAdapterInference(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.normalizer=copy.deepcopy(model.obs_normalizer)
        self.mlp=copy.deepcopy(model.mlp);self.adapter=copy.deepcopy(model.adapter)
        self.yaw_adapter=copy.deepcopy(model.yaw_adapter)
        self.register_buffer('yaw_scale',model.yaw_adapter_scale.clone())
        self.register_buffer('mean',model.adapter_mean.clone())
        self.register_buffer('std',model.adapter_std.clone())
        self.register_buffer('scale',model.adapter_scale.clone())
        self.mean_output=BoundedBetaMean(127.)

    def forward(self,raw: torch.Tensor) -> torch.Tensor:
        original=self.mean_output(self.mlp(self.normalizer(raw)))
        features=(raw-self.mean)/(self.std+.01);gate=balance_gate(raw)[:,None]
        delta=torch.tanh(self.adapter(features))*self.scale*gate
        yaw_delta=torch.tanh(self.yaw_adapter(features))*self.yaw_scale*gate
        delta=torch.cat((delta,yaw_delta),-1)
        return (original+delta).clamp(-.99,.99)

    @torch.jit.export
    def reset(self):pass


def initialize_adapter(actor,parent_actor,parent_critic):
    """Strictly import the frozen base and matched-input recent coordinates."""
    state=actor.state_dict()
    assert set(parent_actor).issubset(state)
    for key,value in parent_actor.items():
        assert state[key].shape==value.shape,key
        state[key]=value.clone()
    state['adapter_mean']=parent_critic['obs_normalizer._mean'].clone()
    state['adapter_std']=parent_critic['obs_normalizer._std'].clone()
    actor.load_state_dict(state,strict=True)


def initialize_cycle12_adapter(actor,parent_actor):
    """Retain the learned two-channel refinement, adding only a zero yaw head."""
    state=actor.state_dict()
    missing=set(state)-set(parent_actor)
    assert missing=={'yaw_adapter_scale',*[f'yaw_adapter.{i}.{k}' for i in (0,2,4) for k in ('weight','bias')]}
    for key,value in parent_actor.items():
        assert key in state and state[key].shape==value.shape,key
        state[key]=value.clone()
    actor.load_state_dict(state,strict=True)
