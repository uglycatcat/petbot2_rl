"""Preserve fine control/termination while adding a bounded sustained-speed signal."""
from dataclasses import replace
import unittest
import torch
from upright.config import Config
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL,state,truth

CFG=Config(num_envs=32,balance_velocity_cost=2.,balance_velocity_cost_cap=1.5,
           balance_velocity_cost_tail='rational',balance_velocity_coarse_weight=.25,
           balance_velocity_coarse_scale_m_s=.3,landing_velocity_cost=0.)

def ready(cfg=CFG):
    p=Runtime(2,'cpu',.06,cfg,MODEL);p.reset(torch.arange(2));p.level[:]=2
    p.balance_duration[:]=30.;p.control.phase[:]=Phase.BALANCE;p.invoked[:]=True
    p.control.set_command([[.75,0],[-.75,0]])
    return p

class CoarseVelocityCost(unittest.TestCase):
    def test_zero_symmetric_monotone_and_bounded(self):
        p=Runtime(9,'cpu',.06,CFG,MODEL)
        p.balance_velocity_error[:]=torch.tensor([0.,.01,.1,.2,.3,100.,-.1,-.2,-.3])
        c=p.coarse_balance_velocity_cost(torch.ones(9,dtype=torch.bool))/.06
        self.assertEqual(float(c[0]),0.)
        self.assertTrue(torch.isfinite(c).all() and (c>=0).all() and (c<=.25).all())
        self.assertTrue((c[1:6]>c[:5]).all());torch.testing.assert_close(c[2:5],c[6:9])

    def test_only_reward_changes_control_observation_termination_identical(self):
        old=ready(replace(CFG,balance_velocity_coarse_weight=0.));new=ready()
        s=state();s['motion'][:,0]=torch.tensor([.5,-.5]);s['gap'][:]=.1
        for _ in range(20):
            a,b=old.advance(s,truth(raised=True)),new.advance(s,truth(raised=True))
            for k in a:
                if k not in ['reward','balance_velocity_coarse_cost']:torch.testing.assert_close(a[k],b[k])
            torch.testing.assert_close(old.observation(s),new.observation(s))
            torch.testing.assert_close(old.critic_state(),new.critic_state())
            torch.testing.assert_close(old.control.targets(torch.zeros(2,3),s),new.control.targets(torch.zeros(2,3),s))
            torch.testing.assert_close(a['reward']-b['reward'],b['balance_velocity_coarse_cost'])
        self.assertTrue((b['balance_velocity_coarse_cost']>0).all())

    def test_no_cost_in_rise_land_ordinary_or_failure(self):
        for phase in [Phase.ORDINARY,Phase.RISE,Phase.LAND]:
            p=ready();p.control.phase[:]=phase;s=state();s['motion'][:,0]=1.
            r=p.advance(s,truth());self.assertFalse(r['balance_velocity_coarse_cost'].any())
        p=ready();s=state();s['motion'][:,0]=-.75;s['gap'][:]=.1
        for _ in range(10):p.advance(s,truth(raised=True))
        t=truth(raised=True);t['body_contact'][:]=True;r=p.advance(s,t)
        self.assertTrue(r['failure'].all());self.assertFalse(r['balance_velocity_coarse_cost'].any())
        torch.testing.assert_close(r['failure_penalty'],torch.full((2,),5.))

    def test_large_bias_retains_positive_real_support_return(self):
        p=ready();s=state();s['motion'][:,0]=torch.tensor([-.75,.75]);s['gap'][:]=.1
        p.control.executed[:]=torch.tensor([1.2,25.,-25.]);p.control.previous[:]=torch.tensor([-1.,-25.,25.])
        for _ in range(80):r=p.advance(s,truth(raised=True))
        self.assertTrue(r['support_truth'].all() and not r['done'].any())
        self.assertTrue((r['reward']>0).all())
        self.assertTrue(((r['balance_velocity_cost']+r['balance_velocity_coarse_cost'])/.06 < 1.75).all())

    def test_added_signal_for_observed_bias_without_replacing_fine_term(self):
        p=ready();p.balance_velocity_error[:]=torch.tensor([.25,.20])
        c=p.coarse_balance_velocity_cost(torch.ones(2,dtype=torch.bool))/.06
        self.assertGreater(float(c[0]-c[1]),.035)
        self.assertLess(float(c[0]-c[1]),.036)
        p.reset(torch.arange(2));self.assertFalse(p.coarse_balance_velocity_cost(torch.ones(2,dtype=torch.bool)).any())

    def test_validation_and_legacy_default(self):
        self.assertEqual(Config().balance_velocity_coarse_weight,0.)
        for w in [-1.,float('nan'),float('inf'),True,'0.25',.5]:
            with self.assertRaises(ValueError):replace(CFG,balance_velocity_coarse_weight=w).validate()
        with self.assertRaises(ValueError):replace(CFG,balance_velocity_cost_cap=0.).validate()
        for scale in [0.,-.3,.76,float('nan'),float('inf'),True,'0.3']:
            with self.assertRaises(ValueError):replace(CFG,balance_velocity_coarse_scale_m_s=scale).validate()

if __name__=='__main__':unittest.main()
