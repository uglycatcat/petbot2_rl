"""Native GAE delayed credit, episode boundaries and the reproducible CLI default."""
import contextlib
import io
import json
from dataclasses import replace
import unittest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from upright.config import Config, load_config, ROOT
from upright.learning import runner_config
from test_contracts import MODEL
import train

class CreditAssignment(unittest.TestCase):
    def test_native_returns_extend_credit_without_crossing_episode_end(self):
        obs = TensorDict({'policy': torch.zeros(2, 1)}, batch_size=[2])
        class ZeroCritic:
            def get_hidden_state(self): return None
            def reset(self, **kwargs): pass
            def __call__(self, obs): return torch.zeros(2, 1)
        initial = []
        for cfg in (Config(), replace(Config(), gae_lambda=.95)):
            parameters = runner_config(cfg, MODEL, 'cpu')['algorithm']
            ppo = object.__new__(PPO)
            ppo.gamma, ppo.lam = parameters['gamma'], parameters['lam']
            ppo.normalize_advantage_per_mini_batch = True
            ppo.critic = ZeroCritic()
            ppo.storage = RolloutStorage('rl', 2, 32, obs, [3])
            ppo.storage.rewards[16] = 1.
            ppo.storage.dones[8, 1] = 1
            ppo.compute_returns(obs)
            initial.append(float(ppo.storage.returns[0, 0, 0]))
            self.assertEqual(float(ppo.storage.returns[:9, 1].abs().max()), 0.)
            self.assertEqual(float(ppo.storage.returns[16, 0, 0]), 1.)
            self.assertTrue(torch.isfinite(ppo.storage.advantages).all())
        self.assertGreater(initial[1], 4*initial[0])

    def test_current_budget_gamma_and_legacy_lambda_are_preserved(self):
        old = load_config(ROOT/'configs/round10.json'); new = load_config()
        before = runner_config(old, MODEL, 'cpu'); after = runner_config(new, MODEL, 'cpu')
        self.assertAlmostEqual(before['algorithm']['lam'], .95**3)
        self.assertEqual(after['algorithm']['lam'], .95)
        self.assertEqual(before['algorithm']['gamma'], after['algorithm']['gamma'])
        for name in ('iterations', 'num_envs', 'rollout_steps', 'learning_rate', 'probe_interval', 'boundary_command_fraction'):
            self.assertEqual(getattr(old, name), getattr(new, name))
        self.assertEqual(before['num_steps_per_env'], after['num_steps_per_env'])

    def test_default_cli_and_explicit_historical_config(self):
        for path in (None, ROOT/'configs/round10.json'):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = train.main(['--dry-run'] + (['--config', str(path)] if path else []))
            report = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertFalse(report['launched'])
            self.assertEqual(report['config'], load_config(path).to_dict())

    def test_invalid_lambda_rejected(self):
        for value in (-.1, 1.01, True, '0.95', float('nan'), float('inf')):
            with self.assertRaises(ValueError): replace(Config(), gae_lambda=value).validate()

if __name__ == '__main__': unittest.main()
