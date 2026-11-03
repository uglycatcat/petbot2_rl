"""STOP exploration must keep the native raw-action PPO density and full authority."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from step_demo.config import PROJECT_ROOT
from step_demo.distributions import StopCenteredGaussianDistribution
from step_demo.ppo import runner_config,distribution_metrics
from step_demo.runtime import load_training


class StopGaussianTests(unittest.TestCase):
    def test_native_density_uses_raw_samples_including_clipped_tail(self):
        torch.manual_seed(4);d=StopCenteredGaussianDistribution(2)
        raw=torch.tensor([0.,-1.]).expand(50000,-1);d.update(raw);sample=d.sample()
        reference=torch.distributions.Normal(*d.params)
        torch.testing.assert_close(d.log_prob(sample),reference.log_prob(sample).sum(-1))
        self.assertGreater(float((sample[:,1]<-1.).float().mean()),.48)
        self.assertLess(float((sample[:,1]<-1.).float().mean()),.52)
        self.assertGreater(float((d.log_prob(sample)-d.log_prob(sample.clamp(-1,1))).abs().max()),1.)
        torch.testing.assert_close(d.kl_divergence(d.params,d.params),torch.zeros(50000))

    def test_small_noise_is_bounded_even_if_optimizer_pushes_log_std_extreme(self):
        d=StopCenteredGaussianDistribution(2)
        for v,expected in [(-100.,.005),(100.,.05)]:
            with torch.no_grad():d.log_std_param.fill_(v)
            d.update(torch.zeros(32,2));self.assertAlmostEqual(float(d.std.max()),expected,places=6)
            self.assertTrue(torch.isfinite(d.entropy).all());distribution_metrics(d)

    def test_likelihood_gradient_moves_nonzero_requests_toward_default(self):
        torch.manual_seed(7);mean=torch.tensor([.3,-.5],requires_grad=True);d=StopCenteredGaussianDistribution(2)
        d.update(mean.expand(60000,-1));sample=d.sample().detach();applied=sample.clamp(-1,1)
        cost=applied[:,0].square()+((applied[:,1]+1)/2).square()
        advantage=-(cost-cost.mean()).detach();loss=-(d.log_prob(sample)*advantage).mean();loss.backward()
        self.assertTrue(torch.isfinite(mean.grad).all());self.assertGreater(float(mean.grad[0]),.4);self.assertGreater(float(mean.grad[1]),.1)

    def test_initial_head_is_random_trainable_near_stop_and_not_an_output_override(self):
        cfg=runner_config(load_training(PROJECT_ROOT/'configs/restart18_aux_seed70.json'))['actor'];cfg.pop('class_name')
        means=[]
        for seed in (70,71):
            torch.manual_seed(seed);obs=TensorDict({'policy':torch.randn(512,26)},batch_size=[512])
            model=MLPModel(obs,{'actor':['policy']},'actor',2,**deepcopy(cfg));model.eval()
            with torch.no_grad():
                mean=model(obs);means.append(mean.clone());sample=model(obs,stochastic_output=True).clamp(-1,1)
            self.assertTrue(all(x.requires_grad for x in model.parameters()))
            self.assertLess(float(mean[:,0].abs().max()),.05);self.assertLess(float((mean[:,1]+1).abs().max()),.05)
            good=(sample[:,0].abs()<.04)&(sample[:,1]<-.946)
            self.assertGreater(float(good.float().mean()),.65)
        self.assertFalse(torch.equal(*means))
        d=StopCenteredGaussianDistribution(2);raw=torch.tensor([[-2.,2.],[2.,-2.]])
        torch.testing.assert_close(d.deterministic_output(raw),raw)
        torch.testing.assert_close(raw.clamp(-1,1),torch.tensor([[-1.,1.],[1.,-1.]]))

    def test_auxiliary_scope_and_no_legacy_variance_curriculum(self):
        good=load_training(PROJECT_ROOT/'configs/restart18_aux_seed70.json')
        before=load_training(PROJECT_ROOT/'configs/restart17_aux_seed70.json');before.pop('beta_std_ceiling');before.update(policy_distribution='stop_gaussian',initial_std=.03);self.assertEqual(good,before)
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'cfg.json'
            for edit in [{'task_variant':'full'},{'auxiliary_objective':'no'},{'initial_std':.5},{'entropy_coef':.01},{'beta_std_ceiling':{'start_update':1,'end_update':2,'final_std':.01}}]:
                f.write_text(json.dumps(dict(good,**edit)))
                with self.assertRaises(ValueError):load_training(f)


if __name__=='__main__':unittest.main()
