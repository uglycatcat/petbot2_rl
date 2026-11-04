"""Regression cases for premature failure, raw PPO density, and STOP access."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from step_demo.config import PROJECT_ROOT
from step_demo.distributions import StopReachableBetaDistribution
from step_demo.ppo import distribution_metrics, runner_config
from step_demo.qualified_task import stopping_progress, certified_front_landing
from step_demo.rewards import component_increment, terminal_reward
from step_demo.runtime import load_training


class QualifiedTaskTests(unittest.TestCase):
    def test_landing_requires_actual_precontact_flight_and_safe_top_contact(self):
        yes=torch.ones(6,2,dtype=torch.bool);no=~yes;count=torch.ones(6,2,dtype=torch.long)
        before=yes.clone();before[1]=False
        count[2]=2;floor=no.clone();floor[3]=True
        hit=torch.zeros(6,dtype=torch.bool);hit[4]=True
        top=yes.clone();top[5]=False
        got=certified_front_landing(before,count,yes,floor,hit,top)
        self.assertEqual(got.all(-1).tolist(),[True,False,False,False,False,False])

    def test_time_alone_does_not_make_early_failure_better(self):
        no=torch.zeros(3,dtype=torch.bool);yes=~no
        r=terminal_reward(no,yes,torch.tensor([.16,7.,15.]))
        torch.testing.assert_close(r,torch.full((3,),-5.))
        self.assertLess(float(r[0]-.05),float(r[-1]))
        self.assertGreater(float(terminal_reward(yes,no,torch.full((3,),15.))[0]),0.)

    def test_invalid_progress_and_repeated_stopping_cannot_pay(self):
        old=torch.zeros(2,8);new=torch.ones(2,8)
        gain,credit=component_increment(old,new,torch.tensor([True,False]))
        self.assertEqual(gain.tolist(),[0.,8.])
        next_gain,_=component_increment(credit,new,torch.tensor([True,False]))
        self.assertEqual(next_gain.tolist(),[0.,0.])
        s=stopping_progress(torch.tensor([.1,.02,0.]),torch.tensor([3.,.66,0.]),torch.ones(3,dtype=torch.bool))
        self.assertTrue(bool((s[1:]>s[:-1]).all()));self.assertEqual(float(s[-1]),6.)
        self.assertEqual(float(stopping_progress(torch.tensor(0.),torch.tensor(0.),False)),0.)

    def test_density_scaling_ceiling_gradients_and_export(self):
        torch.manual_seed(821)
        d=StopReachableBetaDistribution(2);x=torch.zeros(10000,2,2,requires_grad=True)
        d.update(x);a=d.sample()
        self.assertGreater(int((a[:,1]<-1).sum()),0)
        unit=(a-torch.tensor([-1.,-1.1]))/torch.tensor([2.,2.1])
        ref=(torch.distributions.Beta(*d.params).log_prob(unit.clamp(1e-6,1.-1e-6))-torch.tensor([2.,2.1]).log()).sum(-1)
        torch.testing.assert_close(d.log_prob(a),ref)
        torch.testing.assert_close(d.entropy,(torch.distributions.Beta(*d.params).entropy()+torch.tensor([2.,2.1]).log()).sum(-1))
        d.set_std_ceiling([.02,.1]);d.update(x)
        self.assertTrue(bool((d.std<=torch.tensor([.02,.1])+1e-6).all()))
        (-d.log_prob(d.sample()).mean()).backward();self.assertTrue(bool(torch.isfinite(x.grad).all()))
        distribution_metrics(d)
        raw=torch.tensor([[[0.,-10.],[0.,50.]]])
        mean=d.deterministic_output(raw)
        self.assertLess(float(mean[0,1]),-1.)
        torch.testing.assert_close(mean,d.as_deterministic_output_module()(raw))
        restored=StopReachableBetaDistribution(2);restored.load_state_dict(d.state_dict())
        torch.testing.assert_close(restored.deterministic_output(raw),mean)

    def test_configuration_preserves_round14_control_and_no_transfer(self):
        t=load_training(PROJECT_ROOT/'configs/restart21_full_seed70.json')
        old=load_training(PROJECT_ROOT/'configs/round14_seed70.json')
        for key in ('num_envs','iterations','discount_gamma','beta_std_ceiling','decimation',
                    'hip_kp','hip_kd','hip_target_rate_rad_s','wheel_target_accel_rad_s2'):
            self.assertEqual(t[key],old[key])
        self.assertEqual(t['pretrained_models'],[])
        self.assertEqual(runner_config(t)['actor']['distribution_cfg']['class_name'],
                         'step_demo.distributions:StopReachableBetaDistribution')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            for patch in ({'discount_gamma':.99},{'task_variant':'platform_stop_aux'},
                          {'policy_distribution':'beta'},{'pretrained_models':['x.pt']}):
                p.write_text(json.dumps(dict(t,**patch)))
                with self.assertRaises(ValueError):load_training(p)
