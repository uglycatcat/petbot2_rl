"""Regression for the observed clipped-Gaussian exploration runaway."""
import unittest
try:
    import torch
    from rsl_rl.modules import BetaDistribution
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'RSL-RL policy tests run inside the container')
class BoundedPolicyTests(unittest.TestCase):
    def test_extreme_network_outputs_still_produce_finite_bounded_actions(self):
        from step_demo.ppo import distribution_metrics
        for scale in (.1,1.,100.,1e6):
            raw=torch.tensor([[[-scale,scale],[scale,-scale]],[[scale,scale],[scale,scale]]]).repeat(256,1,1)
            dist=BetaDistribution(2);dist.update(raw)
            for action in (dist.sample(),dist.deterministic_output(raw)):
                self.assertTrue(torch.isfinite(action).all())
                self.assertTrue((action.abs()<=1).all())
                self.assertTrue(torch.isfinite(dist.log_prob(action)).all())
            self.assertLessEqual(distribution_metrics(dist)['std_max'],1.)

    def test_entropy_and_kl_cannot_reward_unbounded_action_variance(self):
        dist=BetaDistribution(2)
        for scale in (-100.,0.,100.):
            raw=torch.full((8,2,2),scale,requires_grad=True);dist.update(raw)
            self.assertTrue(torch.isfinite(dist.entropy).all())
            self.assertTrue((dist.entropy<=1e-5).all())  # native entropy on the unit interval
            torch.testing.assert_close(dist.kl_divergence(dist.params,dist.params),torch.zeros(8))
            (-dist.log_prob(dist.sample()).mean()-.01*dist.entropy.mean()).backward()
            self.assertTrue(torch.isfinite(raw.grad).all())

    def test_round02_selects_native_beta_and_fixed_rate(self):
        import json
        from step_demo.config import PROJECT_ROOT
        from step_demo.runtime import load_training
        from step_demo.ppo import runner_config
        cfg=runner_config(load_training(PROJECT_ROOT/'configs/round02_seed70.json'))
        self.assertEqual(cfg['actor']['distribution_cfg']['class_name'],'BetaDistribution')
        self.assertEqual(cfg['algorithm']['schedule'],'fixed')
        self.assertEqual(cfg['algorithm']['learning_rate'],.0003)


if __name__=='__main__':unittest.main()
