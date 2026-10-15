"""Independent simulator/math references and distribution probability contracts."""
import math

import mujoco
import numpy as np
import pytest
import torch

from agile.petbot.core import gravity_from_xyzw, rotate_inverse_xyzw, initial_classes, reset_pool_indices
from agile.petbot.distribution import PetBotBetaDistribution


def test_xyzw_rotation_matches_mujoco_wxyz_reference():
    rng = np.random.default_rng(51)
    q = rng.normal(size=(100, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    vectors = rng.normal(size=(100, 3)); expected = np.empty_like(vectors); gravity = np.empty_like(vectors)
    for i, (xyzw, vector) in enumerate(zip(q, vectors)):
        wxyz_inverse = np.array([xyzw[3], -xyzw[0], -xyzw[1], -xyzw[2]])
        mujoco.mju_rotVecQuat(expected[i], vector, wxyz_inverse)
        mujoco.mju_rotVecQuat(gravity[i], np.array([0., 0., -1.]), wxyz_inverse)
    np.testing.assert_allclose(rotate_inverse_xyzw(torch.tensor(q), torch.tensor(vectors)), expected, atol=1e-12)
    np.testing.assert_allclose(gravity_from_xyzw(torch.tensor(q)), gravity, atol=1e-12)


def test_known_reset_pose_classes_and_pitch_signs():
    c = math.sqrt(.5)
    q = torch.tensor([[0.,0.,0.,1.], [0.,c,0.,c], [0.,-c,0.,c], [c,0.,0.,c], [1.,0.,0.,0.]])
    torch.testing.assert_close(gravity_from_xyzw(q), torch.tensor([[0.,0.,-1.],[1.,0.,0.],[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]), atol=1e-6,rtol=0)
    root = torch.zeros(5,13); root[:,3:7]=q
    assert initial_classes(root).tolist()==[0,2,2,2,1]
    assert reset_pool_indices(root).tolist()==[2]


def bounded_policy():
    mean = [0., -.5, -.5, -1/11, -1/11, 0., 0.]
    std = [.2]*5+[.1]*2
    dist = PetBotBetaDistribution(7, mean, std)
    mlp = torch.nn.Sequential(torch.nn.Linear(27, 32), torch.nn.ELU(), torch.nn.Linear(32,14), torch.nn.Unflatten(-1,(2,7)))
    dist.init_mlp_weights(mlp)
    return dist, mlp, torch.tensor(mean), torch.tensor(std)


def test_bounded_policy_initializes_at_physical_home_and_exports():
    dist,mlp,mean,std=bounded_policy(); raw=mlp(torch.randn(512,27));dist.update(raw)
    torch.testing.assert_close(dist.mean,mean.expand(512,7),atol=1e-6,rtol=0)
    torch.testing.assert_close(dist.std,std.expand(512,7),atol=1e-6,rtol=0)
    exported=torch.jit.script(dist.as_deterministic_output_module())
    torch.testing.assert_close(exported(raw),dist.mean)
    action=dist.sample();assert ((action>-1)&(action<1)).all()
    assert torch.isfinite(dist.log_prob(action)).all()


def test_bounded_policy_probability_kl_entropy_and_ppo_gradient():
    dist,mlp,_,_=bounded_policy(); raw=mlp(torch.randn(64,27));dist.update(raw)
    actions=dist.sample().detach();old=tuple(p.detach().clone() for p in dist.params);old_logp=dist.log_prob(actions).detach()
    reference=torch.distributions.TransformedDistribution(torch.distributions.Beta(*dist.params),[torch.distributions.AffineTransform(-1.,2.)])
    torch.testing.assert_close(dist.log_prob(actions),reference.log_prob(actions).sum(-1))
    torch.testing.assert_close(dist.entropy,torch.distributions.Beta(*dist.params).entropy().sum(-1)+7*math.log(2))
    torch.testing.assert_close(dist.kl_divergence(old,dist.params),torch.zeros(64),atol=1e-6,rtol=0)
    loss=-(torch.exp(dist.log_prob(actions)-old_logp)*torch.linspace(-1,1,64)).mean()-.002*dist.entropy.mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in mlp.parameters());assert mlp[2].weight.grad.abs().max()>0
    # Large raw outputs stay bounded; no many-to-one hard-clipping of Gaussian samples.
    dist.update(torch.randn(64,2,7)*20)
    assert dist.mean.abs().max()<=1
    assert torch.isfinite(dist.log_prob(dist.sample())).all()
