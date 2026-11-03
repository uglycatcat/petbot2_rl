"""Return ordering, strict acceptance, and fresh full-task exploration."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from step_demo.config import PROJECT_ROOT
from step_demo.distributions import FiniteStepGaussianDistribution
from step_demo.finite_task import quality_terminates, reward
from step_demo.runtime import load_training


class FiniteTaskTests(unittest.TestCase):
    def test_waiting_does_not_discount_away_failure(self):
        def returns(steps, progress=0., success=False, running=0., physical=0.):
            x=torch.zeros(steps);x[0]=progress
            complete=torch.zeros(steps,dtype=torch.bool);fail=complete.clone()
            complete[-1]=success;fail[-1]=not success
            return float(reward(x,torch.full_like(x,running),torch.full_like(x,physical),
                                torch.zeros_like(x),complete,fail,.02).sum())
        self.assertLess(returns(750), returns(10))
        self.assertAlmostEqual(returns(750), -28., places=4)
        self.assertGreater(returns(100,progress=2.), returns(750))
        self.assertGreater(returns(126,success=True), returns(750))
        self.assertGreater(returns(126,success=True), returns(126,success=True,running=.01,physical=.01))

    def test_movement_and_action_costs_are_not_overwritten(self):
        z=torch.zeros(4);no=torch.zeros(4,dtype=torch.bool)
        r=reward(z,z,torch.tensor([0.,.1,0.,0.]),torch.tensor([0.,0.,1.,0.]),no,no,.02)
        self.assertAlmostEqual(float(r[0]-r[1]),.1,places=6)
        self.assertAlmostEqual(float(r[0]-r[2]),.02,places=6)

    def test_quality_is_latched_without_training_early_reset_but_eval_is_strict(self):
        invalid=torch.tensor([True,False,True]);t=torch.tensor([.2,15.,15.])
        self.assertEqual(quality_terminates(invalid,False,t,15.).tolist(),[False,False,True])
        self.assertEqual(quality_terminates(invalid,True,t,15.).tolist(),[True,False,True])

    def test_fresh_mean_moves_and_zero_speed_remains_reachable(self):
        torch.manual_seed(820)
        d=FiniteStepGaussianDistribution(2)
        mlp=torch.nn.Sequential(torch.nn.Linear(26,32),torch.nn.ELU(),torch.nn.Linear(32,2))
        d.init_mlp_weights(mlp)
        self.assertEqual(mlp[-1].bias.tolist(),[0.,-.5])
        self.assertTrue(all(p.requires_grad for p in mlp.parameters()))
        d.update(torch.tensor([0.,-.5]).expand(60000,-1));raw=d.sample()
        self.assertGreater(float((raw[:,1]<-1).float().mean()),.004)
        self.assertGreater(float((raw[:,1]>-.8).float().mean()),.9)
        torch.testing.assert_close(d.log_prob(raw),torch.distributions.Normal(*d.params).log_prob(raw).sum(-1))
        torch.testing.assert_close(d.deterministic_output(torch.tensor([[0.,-1.5]])),torch.tensor([[0.,-1.5]]))

    def test_finite_configuration_keeps_source_control_and_rejects_auxiliary_reuse(self):
        t=load_training(PROJECT_ROOT/'configs/restart20_full_seed70.json')
        old=load_training(PROJECT_ROOT/'configs/round14_seed70.json')
        for k in ['decimation','hip_kp','hip_kd','head_kp','head_kd','ear_kp','ear_kd',
                  'hip_target_rate_rad_s','wheel_target_accel_rad_s2','hidden_dims','num_envs','iterations']:
            self.assertEqual(t[k],old[k])
        self.assertEqual(t['discount_gamma'],1.);self.assertEqual(t['pretrained_models'],[])
        self.assertEqual(t['initialization'],'random');self.assertNotIn('critic_observation',t)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            for patch in [{'discount_gamma':.99},{'task_variant':'platform_stop_aux'},
                          {'policy_distribution':'stop_gaussian'},{'full_objective':'unknown'},
                          {'entropy_coef':.01},{'pretrained_models':['x.pt']}]:
                p.write_text(json.dumps(dict(t,**patch)))
                with self.assertRaises(ValueError):load_training(p)


if __name__=='__main__':unittest.main()
