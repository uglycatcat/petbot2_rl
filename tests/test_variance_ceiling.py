"""Regressions for uniform hip sampling behind a zero deterministic mean."""
import unittest
import torch
from rsl_rl.modules import BetaDistribution
from step_demo.distributions import CeilingBetaDistribution,std_ceiling_at


class VarianceCeilingTests(unittest.TestCase):
    def test_warmup_is_native_beta_including_likelihood(self):
        torch.manual_seed(3);raw=torch.randn(128,2,2)*4
        old=BetaDistribution(2);new=CeilingBetaDistribution(2);old.update(raw);new.update(raw)
        torch.testing.assert_close(old.mean,new.mean,rtol=0,atol=0)
        torch.testing.assert_close(old.std,new.std,rtol=0,atol=0)
        actions=old.sample();torch.testing.assert_close(old.log_prob(actions),new.log_prob(actions),rtol=0,atol=0)

    def test_uniform_zero_mean_gets_small_actual_samples(self):
        torch.manual_seed(8);raw=torch.full((20000,2,2),-100.)
        d=CeilingBetaDistribution(2);d.set_std_ceiling(.02);d.update(raw);sample=d.sample()
        torch.testing.assert_close(d.mean,torch.zeros_like(d.mean))
        self.assertLess(float(sample.std(dim=0).max()),.0206)
        self.assertLess(float((sample.abs()>.1).float().mean()),1e-4)
        self.assertTrue(bool((sample.abs()<=1).all()))

    def test_ceiling_preserves_mean_and_narrower_native_variance(self):
        raw=torch.tensor([[[-50.,20.],[-40.,5.]],[[1000.,1000.],[1000.,1000.]]])
        a=BetaDistribution(2);b=CeilingBetaDistribution(2);a.update(raw);b.set_std_ceiling(.03);b.update(raw)
        torch.testing.assert_close(a.mean,b.mean)
        self.assertTrue(bool((b.std<=.030001).all()))
        torch.testing.assert_close(a.std[1],b.std[1])
        torch.testing.assert_close(a.deterministic_output(raw),b.deterministic_output(raw),rtol=0,atol=0)

    def test_log_prob_and_kl_match_the_actual_scaled_density(self):
        d=CeilingBetaDistribution(2);d.set_std_ceiling(.1);d.update(torch.zeros(128,2,2));actions=d.sample();old=tuple(x.clone() for x in d.params)
        reference=torch.distributions.Beta(*d.params)
        expected=(reference.log_prob((actions+1)/2)-torch.log(torch.tensor(2.))).sum(-1)
        torch.testing.assert_close(d.log_prob(actions),expected)
        torch.testing.assert_close(d.kl_divergence(old,d.params),torch.zeros(128))
        d.update(torch.ones(128,2,2));kl=torch.distributions.kl_divergence(torch.distributions.Beta(*old),torch.distributions.Beta(*d.params)).sum(-1)
        torch.testing.assert_close(d.kl_divergence(old,d.params),kl)

    def test_saved_ceiling_restores_exact_distribution(self):
        a=CeilingBetaDistribution(2);a.set_std_ceiling(.02);b=CeilingBetaDistribution(2);b.load_state_dict(a.state_dict())
        raw=torch.randn(32,2,2);a.update(raw);b.update(raw)
        torch.testing.assert_close(a.mean,b.mean);torch.testing.assert_close(a.std,b.std)

    def test_late_distribution_gradients_and_ratios_are_finite(self):
        for magnitude in [0.,20.,100.,1e6]:
            raw=torch.tensor([[[magnitude,-magnitude],[0.,magnitude]]]*64,requires_grad=True)
            d=CeilingBetaDistribution(2);d.set_std_ceiling(.02);d.update(raw);actions=d.sample()
            loss=-d.log_prob(actions).mean()-.01*d.entropy.mean();loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(raw.grad).all())

    def test_curriculum_is_monotonic_and_update_boundary_defined(self):
        cfg={'start_update':150,'end_update':400,'final_std':.02}
        self.assertEqual(std_ceiling_at(cfg,150),1.)
        values=[std_ceiling_at(cfg,u) for u in range(151,501)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertAlmostEqual(values[-1],.02);self.assertAlmostEqual(std_ceiling_at(cfg,400),.02)


if __name__=='__main__':unittest.main()
