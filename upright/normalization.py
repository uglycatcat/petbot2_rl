"""Refresh stale observation statistics while preserving the actor's function."""
import torch


def rebase_actor_normalization(actor,recent,*,eps=.01,max_count=10_000_000):
    """Use the parent critic's same-input recent statistics, not its network.

    W' = W (std_new+eps)/(std_old+eps)
    b' = b + W (mean_new-mean_old)/(std_old+eps).
    The stored statistics change coordinates; the actor's represented function
    stays the same up to floating-point roundoff. Future PPO gradients use the
    current input scales. No optimizer state is compatible with this transform.
    """
    prefix='obs_normalizer.'
    required=[prefix+k for k in ['_mean','_var','_std','count']]
    assert all(k in actor and k in recent for k in required)
    weight=actor['mlp.0.weight'];bias=actor['mlp.0.bias']
    old_mean=actor[prefix+'_mean'].double().reshape(-1)
    new_mean=recent[prefix+'_mean'].double().reshape(-1)
    old_scale=actor[prefix+'_std'].double().reshape(-1)+eps
    new_scale=recent[prefix+'_std'].double().reshape(-1)+eps
    assert weight.shape[1]==old_mean.numel()==new_mean.numel()
    assert max_count>0 and int(recent[prefix+'count'])>0
    assert (old_scale>0).all() and (new_scale>0).all()
    assert all(torch.isfinite(state[k]).all() for state in [actor,recent] for k in required)
    result={k:v.clone() for k,v in actor.items()}
    result['mlp.0.weight']=(weight.double()*(new_scale/old_scale)[None]).to(weight)
    result['mlp.0.bias']=(bias.double()+weight.double()@((new_mean-old_mean)/old_scale)).to(bias)
    for key in required:result[key]=recent[key].clone()
    result[prefix+'count'].fill_(min(int(recent[prefix+'count']),max_count))
    return result


def rebase_all_branches(actor,recent):
    """Recondition trainable inputs while preserving outputs and frozen reflex."""
    from tensordict import TensorDict
    state={k:v.clone() for k,v in actor.state_dict().items()}
    device=next(actor.parameters()).device
    recent={k:v.to(device) for k,v in recent.items()}
    raw=(recent['obs_normalizer._mean'].to(device)+torch.randn(1024,290,device=device)*(recent['obs_normalizer._std'].to(device)+.01))
    with torch.no_grad():before=actor(TensorDict({'policy':raw},batch_size=[len(raw)]),stochastic_output=False).clone()
    changed=rebase_actor_normalization(state,recent)
    for prefix,mean_key,std_key in [('adapter','adapter_mean','adapter_std'),('yaw_adapter','adapter_mean','adapter_std'),('rise_adapter','rise_mean','rise_std')]:
        old_mean=state[mean_key].double().reshape(-1);old_scale=state[std_key].double().reshape(-1)+.01
        new_mean=recent['obs_normalizer._mean'].to(device).double().reshape(-1)
        new_scale=recent['obs_normalizer._std'].to(device).double().reshape(-1)+.01
        w=state[prefix+'.0.weight'];b=state[prefix+'.0.bias']
        changed[prefix+'.0.weight']=(w.double()*(new_scale/old_scale)[None]).to(w)
        changed[prefix+'.0.bias']=(b.double()+w.double()@((new_mean-old_mean)/old_scale)).to(b)
        changed[mean_key]=new_mean.reshape_as(state[mean_key]).to(state[mean_key])
        changed[std_key]=(new_scale-.01).reshape_as(state[std_key]).to(state[std_key])
    actor.load_state_dict(changed,strict=True)
    with torch.no_grad():after=actor(TensorDict({'policy':raw},batch_size=[len(raw)]),stochastic_output=False)
    maximum=float((before-after).abs().max())
    assert torch.allclose(before,after,atol=2e-5,rtol=2e-5),maximum
    assert all(torch.equal(state[k],v) for k,v in changed.items() if k.startswith(('balance_reflex.','parent_balance.')))
    return dict(passed=True,samples=len(raw),max_abs_error=maximum,frozen_inner_unchanged=True)
