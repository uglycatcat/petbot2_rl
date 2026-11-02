"""Rare memory transitions must not grow with empirical rarity or disappear on reload."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.modules import EmpiricalNormalization
from step_demo.models import SensorMemoryNormalization, SensorMemoryMLP
from step_demo.ppo import runner_config
from step_demo.runtime import load_training
from step_demo.config import PROJECT_ROOT


class MemoryNormalizationTests(unittest.TestCase):
    def test_rare_memory_keeps_unit_scale_and_sensor_normalization_is_identical(self):
        torch.manual_seed(70)
        n=SensorMemoryNormalization(29);reference=EmpiricalNormalization(29)
        history=torch.randn(4096,29);history[:,26:]=0.;history[-1,26:]=1.
        n.update(history);reference.update(history)
        x=torch.randn(3,29);x[:,26:]=torch.tensor([[0.,0.,0.],[1.,.35,0.],[1.,1.,1.]])
        out=n(x);torch.testing.assert_close(out[:,:26],reference(x)[:,:26],rtol=0,atol=0)
        torch.testing.assert_close(out[:,26:],x[:,26:],rtol=0,atol=0)
        self.assertGreater(float(reference(x)[2,26]),20.)

    def test_memory_jacobian_is_identity_even_after_zero_only_history(self):
        n=SensorMemoryNormalization(29);n.update(torch.zeros(1024,29))
        x=torch.zeros(2,29,requires_grad=True);n(x)[:,26:].sum().backward()
        torch.testing.assert_close(x.grad[:,:26],torch.zeros(2,26))
        torch.testing.assert_close(x.grad[:,26:],torch.ones(2,3))

    def test_state_reload_preserves_prefix_statistics_and_bounded_suffix(self):
        n=SensorMemoryNormalization(29);n.update(torch.randn(200,29));other=SensorMemoryNormalization(29)
        other.load_state_dict(n.state_dict(),strict=True);x=torch.randn(5,29)
        torch.testing.assert_close(other(x),n(x),rtol=0,atol=0)
        torch.testing.assert_close(other(x)[:,26:],x[:,26:],rtol=0,atol=0)

    def test_native_actor_critic_use_new_normalizer_and_reload_exactly(self):
        t=load_training(PROJECT_ROOT/'configs/round20_seed70.json');cfg=runner_config(t)
        obs=TensorDict({'policy':torch.zeros(64,29)},[64])
        for key,output_dim in [('actor',2),('critic',1)]:
            args=copy.deepcopy(cfg[key]);self.assertEqual(args.pop('class_name'),'step_demo.models:SensorMemoryMLP')
            model=SensorMemoryMLP(obs,cfg['obs_groups'],key,output_dim,**args)
            model.update_normalization(obs);obs['policy'][:,26:]=1.
            torch.testing.assert_close(model.get_latent(obs)[:,26:],torch.ones(64,3))
            clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(model(obs),clone(obs),rtol=0,atol=0)
            model(obs).sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mlp.parameters()))

    def test_old_configuration_keeps_native_normalization(self):
        cfg=runner_config(load_training(PROJECT_ROOT/'configs/round19_seed70.json'))
        self.assertEqual(cfg['actor']['class_name'],'MLPModel');self.assertEqual(cfg['critic']['class_name'],'MLPModel')

    def test_round20_changes_only_normalization_selection(self):
        for suffix in ['seed70','seed71','gate','scale_gate']:
            old=load_training(PROJECT_ROOT/f'configs/round19_{suffix}.json');new=load_training(PROJECT_ROOT/f'configs/round20_{suffix}.json')
            self.assertEqual(new.pop('observation_normalization'),'sensor_memory_unit_v20');self.assertEqual(new,old)

    def test_wrong_interface_or_misspelled_configuration_is_rejected(self):
        with self.assertRaises(ValueError):SensorMemoryNormalization(26)
        base=load_training(PROJECT_ROOT/'configs/round20_seed70.json')
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'config.json'
            for overrides in [dict(observation_normalization='typo'),dict(task_memory='none'),dict(critic_observation='reward_state_v11')]:
                f.write_text(json.dumps(dict(base,**overrides)))
                with self.assertRaises(ValueError):load_training(f)


if __name__=='__main__':unittest.main()
