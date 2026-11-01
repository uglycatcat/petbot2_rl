"""Ensure wheel exploration changes without silently broadening hip sampling."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from rsl_rl.modules import BetaDistribution
from step_demo.config import PROJECT_ROOT
from step_demo.distributions import ActionwiseCeilingBetaDistribution, CeilingBetaDistribution, std_ceiling_at
from step_demo.ppo import distribution_metrics, runner_config
from step_demo.runtime import load_training


class ActionwiseCeilingTests(unittest.TestCase):
    def test_warmup_matches_native_density_and_actions_exactly(self):
        torch.manual_seed(2);raw=torch.randn(128,2,2)*3
        native=BetaDistribution(2);new=ActionwiseCeilingBetaDistribution(2)
        native.update(raw);new.update(raw)
        torch.testing.assert_close(native.mean,new.mean,rtol=0,atol=0)
        torch.testing.assert_close(native.std,new.std,rtol=0,atol=0)
        action=native.sample();torch.testing.assert_close(native.log_prob(action),new.log_prob(action),rtol=0,atol=0)

    def test_wheel_ceiling_does_not_change_hip_marginal_or_means(self):
        torch.manual_seed(8);raw=torch.randn(512,2,2)*4
        old=CeilingBetaDistribution(2);old.set_std_ceiling(.02);old.update(raw)
        new=ActionwiseCeilingBetaDistribution(2);new.set_std_ceiling([.02,.10]);new.update(raw)
        torch.testing.assert_close(old.mean,new.mean)
        for a,b in zip(old.params,new.params):torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
        self.assertTrue(bool((new.std[:,0]<=.020001).all()))
        self.assertTrue(bool((new.std[:,1]<=.100001).all()))
        self.assertGreater(float(new.std[:,1].mean()),float(old.std[:,1].mean()))

    def test_empirical_samples_follow_both_marginal_bounds(self):
        torch.manual_seed(17);d=ActionwiseCeilingBetaDistribution(2)
        d.set_std_ceiling([.02,.1]);d.update(torch.zeros(30000,2,2));sample=d.sample()
        self.assertAlmostEqual(float(sample[:,0].std()),.02,delta=.0005)
        self.assertAlmostEqual(float(sample[:,1].std()),.10,delta=.002)
        self.assertTrue(bool((sample.abs()<=1.).all()))

    def test_already_narrow_native_distribution_is_not_broadened(self):
        raw=torch.full((32,2,2),1e5);native=BetaDistribution(2);new=ActionwiseCeilingBetaDistribution(2)
        new.set_std_ceiling([.02,.1]);native.update(raw);new.update(raw)
        torch.testing.assert_close(native.std,new.std,rtol=0,atol=0)

    def test_likelihood_entropy_and_kl_use_the_actual_vector_density(self):
        d=ActionwiseCeilingBetaDistribution(2);d.set_std_ceiling([.02,.1]);d.update(torch.zeros(64,2,2))
        action=d.sample();old=tuple(x.clone() for x in d.params);reference=torch.distributions.Beta(*old)
        torch.testing.assert_close(d.log_prob(action),(reference.log_prob((action+1)/2)-torch.log(torch.tensor(2.))).sum(-1))
        # RSL-RL reports entropy in Beta's [0,1] coordinates; the omitted
        # affine constant 2*log(2) has no policy gradient or KL effect.
        torch.testing.assert_close(d.entropy,reference.entropy().sum(-1))
        d.update(torch.ones(64,2,2))
        torch.testing.assert_close(d.kl_divergence(old,d.params),torch.distributions.kl_divergence(reference,torch.distributions.Beta(*d.params)).sum(-1))

    def test_vector_buffer_roundtrip_preserves_density(self):
        a=ActionwiseCeilingBetaDistribution(2);a.set_std_ceiling([.02,.1]);b=ActionwiseCeilingBetaDistribution(2);b.load_state_dict(a.state_dict(),strict=True)
        raw=torch.randn(64,2,2);a.update(raw);b.update(raw)
        for x,y in zip(a.params,b.params):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_scalar_checkpoint_cannot_silently_restore_as_vector(self):
        with self.assertRaises(RuntimeError):ActionwiseCeilingBetaDistribution(2).load_state_dict(CeilingBetaDistribution(2).state_dict(),strict=True)

    def test_finite_gradients_at_extreme_outputs(self):
        for v in [0.,20.,100.,1e6]:
            raw=torch.tensor([[[v,-v],[0.,v]]]*32,requires_grad=True);d=ActionwiseCeilingBetaDistribution(2);d.set_std_ceiling([.02,.1]);d.update(raw)
            loss=-d.log_prob(d.sample()).mean()-.01*d.entropy.mean();loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)));self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_per_action_logging_detects_a_hip_violation_even_below_wheel_cap(self):
        d=ActionwiseCeilingBetaDistribution(2);d.set_std_ceiling([.05,.1]);d.update(torch.zeros(16,2,2));d.set_std_ceiling([.02,.1])
        with self.assertRaises(RuntimeError):distribution_metrics(d)
        d.update(torch.zeros(16,2,2));stats=distribution_metrics(d)
        self.assertEqual(len(stats['std_ceiling']),2);self.assertLess(stats['std_max_per_action'][0],.020001)

    def test_curriculum_preserves_hip_and_changes_only_wheel_tail(self):
        old={'start_update':150,'end_update':400,'final_std':.02};new=dict(old,wheel_final_std=.1)
        for step in [0,150,151,250,399,400,500]:
            h,w=std_ceiling_at(new,step);self.assertEqual(h,std_ceiling_at(old,step));self.assertGreaterEqual(w,h)
        self.assertEqual(std_ceiling_at(new,150),[1.,1.]);self.assertAlmostEqual(std_ceiling_at(new,500)[1],.1)

    def test_config_only_changes_wheel_ceiling_and_selects_vector_distribution(self):
        for name in ['seed70','seed71','gate','scale_gate']:
            old=load_training(PROJECT_ROOT/f'configs/round13_{name}.json');new=load_training(PROJECT_ROOT/f'configs/round14_{name}.json')
            self.assertTrue(runner_config(new)['actor']['distribution_cfg']['class_name'].endswith(':ActionwiseCeilingBetaDistribution'))
            self.assertEqual(new['beta_std_ceiling'].pop('wheel_final_std'),.1);self.assertEqual(new,old)

    def test_invalid_vector_or_config_cannot_bypass_validation(self):
        d=ActionwiseCeilingBetaDistribution(2)
        for value in [.1,[.02],[.02,.1,.2],[.02,float('nan')],[.02,0.],[.02,2.]]:
            with self.assertRaises(ValueError):d.set_std_ceiling(value)
        cfg=load_training(PROJECT_ROOT/'configs/round14_seed70.json')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            for value in [float('nan'),float('inf'),0.,1.,'bad']:
                cfg['beta_std_ceiling']['wheel_final_std']=value;path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load_training(path)


if __name__=='__main__':unittest.main()
