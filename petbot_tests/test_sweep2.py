"""Behavioral regressions for recovery shaping and PPO exploration bounds."""
import ast
import math

import pytest
import torch

from agile.petbot.core import discounted_progress, recovery_potential, reset_pool_indices
from agile.petbot.model import PROJECT_ROOT


def test_potential_reward_cannot_pay_for_stationary_failed_pose():
    gravity = torch.tensor([[0., 0., 0.], [0., 0., 1.]])
    error = torch.ones(2, 5)
    phi = recovery_potential(gravity, error)
    reward = discounted_progress(phi, phi, .995, torch.zeros(2, dtype=torch.bool))
    assert (reward <= 0).all()


def test_discounted_shaping_telescopes_even_when_recovery_first_moves_away():
    gamma = .995
    # A detour is not given an extra net penalty for its intermediate potential.
    paths = ([.4, .1, .2, .8, 1.], [.4, .5, .6, .8, 1.])
    totals = []
    for path in paths:
        values = torch.tensor(path)
        delta = discounted_progress(values[:-1], values[1:], gamma, torch.zeros(4, dtype=torch.bool))
        totals.append(sum(gamma**t * float(r) for t, r in enumerate(delta)))
    assert totals[0] == pytest.approx(totals[1], abs=1e-7)
    assert totals[0] == pytest.approx(-.4 + gamma**4, abs=1e-7)
    assert discounted_progress(torch.tensor([.8]), torch.tensor([1.]), gamma, torch.tensor([True])).item() == pytest.approx(-.8)


def test_curriculum_pool_uses_negative_pitch_fallen_states_only():
    roots = torch.zeros(3, 13)
    roots[:, 3] = math.cos(math.pi/4)
    roots[:, 5] = torch.tensor([-math.sin(math.pi/4), math.sin(math.pi/4), 0.])
    roots[2, 3] = 1
    assert reset_pool_indices(roots).tolist() == [0]


def test_distribution_adapter_forwards_bounds_and_ppo_uses_them():
    tree = ast.parse((PROJECT_ROOT/'agile/rl_env/rsl_rl/rl_cfg.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_distribution_cfg')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'distribution_adapter', 'exec'), ns)
    cfg = ns['_distribution_cfg']({'init_noise_std': .35, 'noise_std_type': 'log', 'noise_std_range': (.15, .6)})
    from rsl_rl.modules.distribution import GaussianDistribution
    cfg.pop('class_name')
    dist = GaussianDistribution(output_dim=7, **cfg)
    for parameter, expected in ((-100., .15), (100., .6)):
        with torch.no_grad():
            dist.log_std_param.fill_(parameter)
        dist.update(torch.zeros(16, 7))
        torch.testing.assert_close(dist.std, torch.full((16, 7), expected))
        samples = dist.sample()
        assert torch.isfinite(dist.log_prob(samples)).all()
        expected_logp = torch.distributions.Normal(torch.zeros_like(samples), expected).log_prob(samples).sum(-1)
        torch.testing.assert_close(dist.log_prob(samples), expected_logp)
