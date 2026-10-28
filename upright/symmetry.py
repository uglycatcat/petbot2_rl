"""Sagittal reflection for every frame of the deployed 58 x 5 observation.

Polar y changes sign; axial x/z change sign. Ear/hip/wheel axes are
mapped using the source MJCF, not a blanket sign flip of joint positions.
Only the auxiliary mean loss uses transformed samples: PPO likelihoods always
refer to actual rollouts. Inner parent asymmetry remains an evaluation concern.
"""
import torch

ODD=(1,4,6,8,10,11,16,24,26,28,30,31,50,52,54,56,57)
PAIRS=((12,13),(14,15),(17,18),(19,20),(21,22),(34,35))
CONTRACT=dict(version='sagittal_58x5_v1', odd=list(ODD), pairs=[list(x) for x in PAIRS],
    action_odd=[2], ppo_data_augmentation=False, mirror_loss_coefficient=.2,
    rollout='paired_counter_streams_with_per_environment_generation',
    frozen_feedback='preserved_not_symmetrized')

def mirror_observation(raw):
    assert raw.shape[-1]==290
    frames=raw.reshape(*raw.shape[:-1],5,58)
    result=frames.clone()
    result[...,list(ODD)] = -frames[...,list(ODD)]
    for left,right in PAIRS:
        result[...,left]=frames[...,right];result[...,right]=frames[...,left]
    return result.reshape_as(raw)

def mirror_action(action):
    result=action.clone();result[...,2]=-result[...,2]
    return result

def augment(env,obs=None,actions=None):
    if obs is not None:
        mirrored=obs.clone()
        for key in obs.keys():mirrored[key]=mirror_observation(obs[key])
        obs=torch.cat((obs,mirrored),dim=0)
    if actions is not None:actions=torch.cat((actions,mirror_action(actions)),dim=0)
    return obs,actions
