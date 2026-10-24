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
