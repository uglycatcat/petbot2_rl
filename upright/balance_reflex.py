"""Frozen parent refinement for20ms physical feedback within a60ms PPO action.

The fixed controller is part of the environment dynamics, never the live PPO
actor. Old history/task clocks stay60ms; only current physical features refresh.
"""
import copy
import torch
from .balance_adapter import BalanceAdapterModel, balance_gate
from .distribution import BoundedBetaMean


class FrozenBalanceReflex(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.mlp=copy.deepcopy(actor.mlp)
        self.normalizer=copy.deepcopy(actor.obs_normalizer)
        self.refinement=copy.deepcopy(actor.adapter)
        self.register_buffer('mean',actor.adapter_mean.clone())
        self.register_buffer('std',actor.adapter_std.clone())
        self.register_buffer('scale',actor.adapter_scale.clone())
        self.mean_output=BoundedBetaMean(127.)
        self.requires_grad_(False)

    def forward(self,raw: torch.Tensor) -> torch.Tensor:
        features=(raw-self.mean)/(self.std+.01)
        return torch.tanh(self.refinement(features))*self.scale*balance_gate(raw)[:,None]

    @torch.jit.export
    def base(self,raw: torch.Tensor) -> torch.Tensor:
        return self.mean_output(self.mlp(self.normalizer(raw)))[:,:2]


class ReflexAdapterModel(BalanceAdapterModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Initial values are replaced with the explicitly validated parent.
        # Registering this module makes every checkpoint self-contained.
        self.balance_reflex=FrozenBalanceReflex(self)

    def initialize_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and missing and all(k.startswith('balance_reflex.') for k in missing)
        self.balance_reflex.load_state_dict(FrozenBalanceReflex(self).state_dict(),strict=True)


def attach_balance_reflex(env,actor,enabled,*,full_parent=False):
    """Install fixed inner feedback without changing the rollout action density."""
    if not enabled:return
    from .feedback_task import observations,ensure
    a=env.action_manager.get_term('moving');fixed=actor.parent_balance if full_parent else actor.balance_reflex
    assert env.step_dt==.06 and env.physics_dt==.002 and a.fast_stride==1
    assert not any(p.requires_grad for p in fixed.parameters())
    process=a.process_actions;apply=a.apply_actions;reset=a.reset
    cache={}

    def process_actions(actions):
        raw=a.policy_observation
        assert raw.shape==(env.num_envs,290)
        cache['raw']=raw.clone();cache['frame_anchor']=a.target[:,3].clone()
        with torch.no_grad():
            if full_parent:
                parent=fixed(raw)[:,:2]
            else:
                base=fixed.base(raw);parent=(base+fixed(raw)).clamp(-.99,.99)
                cache['base']=base
            cache['offset']=actions[:,:2].detach()-parent
        process(actions)

    def apply_actions():
        p=ensure(env);mask=(p.phase==2)&(p.phase_time>=1.8)
        if a.inner_index in (10,20) and mask.any():
            raw=cache['raw'].clone();raw[:,-58:]=observations(env)
            with torch.no_grad():
                mean=fixed(raw)[:,:2] if full_parent else cache['base']+fixed(raw)
                action=(mean+cache['offset']).clamp(-.99,.99)
            previous=a.inner_policy.mean(-1);desired=action[:,1]*a.max_speed
            if getattr(env.cfg,'capability_version','legacy')=='v25':
                from .capability import balance_common
                desired=balance_common(action[:,1])
            common=previous+(desired-previous).clamp(-env.cfg.wheel_slew*.02,env.cfg.wheel_slew*.02)
            bound=env.cfg.wheel_slew*env.step_dt;anchor=a.inner_anchor
            lower=(anchor-bound).clamp_min(-a.max_speed).mean(-1)
            upper=(anchor+bound).clamp_max(a.max_speed).mean(-1)
            common=common.clamp(min=lower,max=upper)
            a.inner_policy[mask]=common[mask,None]
            desired=a.lo+(action[:,0]+1)*.5*(a.hi-a.lo)
            previous=a.target[:,3];limit=env.cfg.frame_slew*.02
            frame=previous+(desired-previous).clamp(-limit,limit)
            bound=env.cfg.frame_slew*env.step_dt
            frame=frame.clamp(min=cache['frame_anchor']-bound,max=cache['frame_anchor']+bound).clamp(a.lo,a.hi)
            a.target[mask,3:]=frame[mask,None]
            a.applied[:,0]=2*(a.target[:,3]-a.lo)/(a.hi-a.lo)-1
        apply()

    def reset_actions(env_ids=None):
        reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        for value in cache.values():value[ids]=0.

    a.process_actions=process_actions;a.apply_actions=apply_actions;a.reset=reset_actions
