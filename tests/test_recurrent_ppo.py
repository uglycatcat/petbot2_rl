"""Real RSL-RL recurrent storage/density checks, CPU only inside the container."""
from copy import deepcopy
import unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import RNNModel, MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from step_demo.config import PROJECT_ROOT
from step_demo.runtime import load_training
from step_demo.ppo import runner_config


class RecurrentPolicyTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(704)
        self.obs = TensorDict({'policy':torch.randn(4,27),'critic':torch.randn(4,132)}, [4])
        self.cfg = runner_config(load_training(PROJECT_ROOT/'configs/engineering.json'))
        actor = deepcopy(self.cfg['actor']); actor.pop('class_name')
        critic = deepcopy(self.cfg['critic']); critic.pop('class_name')
        self.actor = RNNModel(self.obs, self.cfg['obs_groups'], 'actor', 2, **actor)
        self.critic = MLPModel(self.obs, self.cfg['obs_groups'], 'critic', 1, **critic)

    def test_actor_ignores_privileged_state_and_has_resettable_history(self):
        self.make()
        with torch.inference_mode():
            original = self.actor(self.obs)
            self.actor.reset(torch.ones(4,dtype=torch.bool))
            changed = self.obs.clone(); changed['critic'] = torch.randn_like(changed['critic'])*1000.
            torch.testing.assert_close(self.actor(changed), original, rtol=0, atol=0)
            second = self.actor(self.obs)
            self.assertFalse(torch.equal(original, second))
            self.actor.reset(torch.ones(4,dtype=torch.bool))
            torch.testing.assert_close(self.actor(self.obs), original, rtol=0, atol=0)
        self.assertFalse(self.actor.obs_normalization)
        self.assertFalse(self.critic.obs_normalization)

    def test_raw_samples_recurrent_replay_and_optimizer_are_consistent(self):
        self.make()
        storage = RolloutStorage('rl', 4, 16, self.obs, [2])
        cfg = deepcopy(self.cfg['algorithm']); cfg.pop('class_name')
        cfg.update(num_learning_epochs=1, num_mini_batches=2)
        alg = PPO(self.actor, self.critic, storage, **cfg)
        obs = self.obs
        raw_samples = []
        with torch.inference_mode():
            for k in range(16):
                raw = alg.act(obs); raw_samples.append(raw.clone())
                obs = TensorDict({'policy':torch.randn(4,27),'critic':torch.randn(4,132)}, [4])
                dones = torch.tensor([k%3==2,k%5==4,k%7==6,False])
                alg.process_env_step(obs, torch.randn(4), dones, {})
            torch.testing.assert_close(storage.actions, torch.stack(raw_samples), rtol=0, atol=0)
            alg.compute_returns(obs)
            for batch in storage.recurrent_mini_batch_generator(2,1):
                self.actor(batch.observations, masks=batch.masks,
                           hidden_state=batch.hidden_states[0], stochastic_output=True)
                replay = self.actor.get_output_log_prob(batch.actions)
                torch.testing.assert_close(replay, batch.old_actions_log_prob.squeeze(-1), atol=2e-6, rtol=2e-6)
        before = [p.detach().clone() for p in self.actor.parameters()]
        losses = alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,self.actor.parameters())))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.actor.parameters()))

    def test_exact_zero_is_in_raw_support_and_density_has_affine_jacobian(self):
        self.make()
        with torch.inference_mode():
            self.actor(self.obs, stochastic_output=True)
            d = self.actor.distribution
            raw = torch.tensor([[0.,-1.05]]).repeat(4,1)
            unit = (raw-d.offset)/d.scale
            expected = (torch.distributions.Beta(d._alpha,d._beta).log_prob(unit)-d.scale.log()).sum(-1)
            torch.testing.assert_close(d.log_prob(raw), expected)
            self.assertTrue(((raw.clamp(-1,1)[:,1]+1.)*12.5 == 0.).all())

    def test_short_rollout_bootstraps_and_preserves_nonterminal_gru(self):
        self.make()
        storage = RolloutStorage('rl', 4, 16, self.obs, [2])
        cfg = deepcopy(self.cfg['algorithm']); cfg.pop('class_name')
        cfg.update(num_learning_epochs=1, num_mini_batches=2)
        alg = PPO(self.actor, self.critic, storage, **cfg)
        with torch.inference_mode():
            for _ in range(16):
                alg.act(self.obs)
                alg.process_env_step(self.obs, torch.ones(4), torch.zeros(4,dtype=torch.bool), {})
            state = self.actor.get_hidden_state().clone()
            last_value = self.critic(self.obs).clone()
            alg.compute_returns(self.obs)
            torch.testing.assert_close(storage.returns[-1], 1.+last_value)
            torch.testing.assert_close(self.actor.get_hidden_state(), state)
        alg.update()
        torch.testing.assert_close(self.actor.get_hidden_state(), state)
        with torch.inference_mode():
            alg.act(self.obs)
            self.assertFalse(torch.equal(self.actor.get_hidden_state(),state))
            alg.process_env_step(self.obs,torch.ones(4),torch.tensor([True,False,False,False]),{})
            self.assertEqual(float(self.actor.get_hidden_state()[:,0].abs().sum()),0.)
            self.assertGreater(float(self.actor.get_hidden_state()[:,1:].abs().sum()),0.)


if __name__ == '__main__': unittest.main()
