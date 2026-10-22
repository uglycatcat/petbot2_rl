"""Explicit actor-only transfer preserving each historical frame's original 41 features."""
import torch

def expand_actor(old,new,frames=5,old_width=41,new_width=49):
    indices=torch.cat([torch.arange(old_width)+i*new_width for i in range(frames)])
    out={k:v.clone() for k,v in new.items()}
    for k,v in old.items():
        if k=='mlp.0.weight':
            assert v.shape[-1]==frames*old_width and out[k].shape[-1]==frames*new_width
            out[k].zero_();out[k][:,indices]=v.to(out[k])
        elif k in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
            out[k][:,indices]=v.to(out[k])
        else:
            assert out[k].shape==v.shape,k
            out[k]=v.to(out[k]).clone()
    return out
