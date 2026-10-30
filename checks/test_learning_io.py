"""Native model and serialization checks with synthetic CPU tensors; no PPO training."""
import contextlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from tensordict import TensorDict
from upright.config import Config
from upright.export import make_actor, export_actor
from upright.runtime import OBSERVATION_DIM, Runtime
from upright.model import load_model
from upright.learning import Runner
from upright.experiment import SCHEMA, atomic_json


class ModelIO(unittest.TestCase):
    def test_bounded_samples_and_finite_gradient(self):
        cfg = Config(num_envs=32)
        with contextlib.redirect_stdout(io.StringIO()): actor = make_actor(cfg)
        obs = TensorDict({'policy': torch.randn(32, OBSERVATION_DIM*cfg.history)}, batch_size=[32])
        samples = actor(obs, stochastic_output=True)
        self.assertTrue(torch.isfinite(samples).all())
        self.assertLessEqual(float(samples.abs().max()), 1.)
        mean = actor(obs)
        mean.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_single_graph_onnx_parity(self):
        cfg = Config(num_envs=32)
        with contextlib.redirect_stdout(io.StringIO()): actor = make_actor(cfg)
        with tempfile.TemporaryDirectory() as name:
            report = export_actor(actor, cfg, Path(name)/'numeric_contract_only', {'synthetic_untrained_actor': True})
            self.assertLessEqual(report['numeric_max_abs_error'], 1e-5)
            self.assertFalse(report['simulation_validated'])

    def test_checkpoint_counter_and_full_state_recovery(self):
        # Native checkpoint orchestration with a small fake optimizer payload.
        # No simulator or PPO update is invoked by this test.
        cfg = Config(num_envs=32)
        model = load_model()
        p = Runtime(2, 'cpu', .06, cfg, model)
        class Algorithm:
            loaded = False
            def save(self):
                return {'actor_state_dict': {'x': torch.ones(1)}, 'critic_state_dict': {'x': torch.ones(1)},
                        'optimizer_state_dict': {'state': {'synthetic': 1}}}
            def load(self, value, load_cfg, strict):
                self.loaded = load_cfg is None and strict and 'optimizer_state_dict' in value
                return True
        with tempfile.TemporaryDirectory() as name:
            runner = object.__new__(Runner)
            runner.folder = Path(name); runner.checkpoints = Path(name)/'checkpoints'; runner.checkpoints.mkdir()
            runner.alg = Algorithm(); runner.runtime = p; runner.device = 'cpu'
            runner.env = types.SimpleNamespace(env=types.SimpleNamespace(reset=lambda: None))
            runner.manifest = dict(schema=SCHEMA, initialization='random', run_id='test', code_sha256='c',
                                   model_sha256='m', config_sha256='f', config=cfg.to_dict())
            runner.current_learning_iteration = 99; runner.completed_updates = 0
            runner.save()
            self.assertEqual(runner.completed_updates, 100)
            runner.current_learning_iteration = 100
            runner.save_boundary()
            saved = torch.load(runner.checkpoints/'update_000100.pt', weights_only=False)
            self.assertEqual(saved['iter'], 99)
            self.assertEqual(saved['restart']['updates_completed'], 100)
            self.assertIn('rng', saved['restart']); self.assertIn('curriculum', saved['restart'])
            self.assertEqual(runner.load_same_run(runner.checkpoints/'update_000100.pt'), 100)
            self.assertTrue(runner.alg.loaded)
            self.assertEqual(runner.current_learning_iteration, 100)
            with self.assertRaises(RuntimeError): runner.load('historical.pt')

    def test_chunk_counts_are_not_replayed(self):
        runner = object.__new__(Runner); runner.completed_updates = 100
        with patch('rsl_rl.runners.OnPolicyRunner.learn') as learn:
            runner.learn_chunk(25)
            self.assertEqual(runner.current_learning_iteration, 100)
            learn.assert_called_once_with(num_learning_iterations=25, init_at_random_ep_len=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
