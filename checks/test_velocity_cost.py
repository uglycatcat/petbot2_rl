"""Reward-only motion costs: sustained bias, measured landing speed, lifecycle and neutrality."""
from dataclasses import replace
import unittest
import torch
from upright.config import Config
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL,state,truth

CFG=Config(num_envs=32,balance_velocity_cost=2.,landing_velocity_cost=2.)
class VelocityCostContracts(unittest.TestCase):
    def test_persistent_bias_is_costly_but_signed_fluctuations_average_out(self):
        p=Runtime(3,'cpu',.06,CFG,MODEL);s=state(3)
        balancing=torch.ones(3,dtype=torch.bool);landing=~balancing
        for j in range(80):
            s['motion'][:,0]=torch.tensor([.03,-.03,.03 if j%2 else -.03])
            cost,_=p.velocity_costs(s,balancing,landing)
        torch.testing.assert_close(cost[0],cost[1])
        self.assertGreater(float(cost[0]),10*float(cost[2]))
        self.assertAlmostEqual(float(cost[0]),.06*2*.5*(.03/.05)**2,places=6)

    def test_landing_uses_actual_entry_speed_and_is_not_zero_speed_objective(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);s=state()
        p.control.transition_motion[:,0]=torch.tensor([.4,-.4]);p.control.set_command([[0,0],[0,0]])
        s['motion'][:,0]=torch.tensor([.4,-.4]);b=torch.zeros(2,dtype=torch.bool)
        _,cost=p.velocity_costs(s,b,~b);self.assertFalse(cost.any())
        s['motion'][:,0]=torch.tensor([.6,-.6]);_,cost=p.velocity_costs(s,b,~b)
        torch.testing.assert_close(cost,torch.full((2,),.06))
        s['motion'][:,0]=0;_,cost=p.velocity_costs(s,b,~b);self.assertTrue((cost>.06).all())

    def test_filter_resets_outside_balance_and_on_episode_reset(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);s=state();s['motion'][:,0]=1;b=torch.ones(2,dtype=torch.bool)
        with torch.inference_mode():p.velocity_costs(s,b,~b)
        self.assertTrue(p.balance_velocity_error.any());p.reset(torch.arange(2));self.assertFalse(p.balance_velocity_error.any())
        p.velocity_costs(s,b,~b);p.velocity_costs(s,~b,b);self.assertFalse(p.balance_velocity_error.any())

    def test_reward_changes_without_changing_observations_actions_or_termination(self):
        old=Runtime(2,'cpu',.06,replace(CFG,balance_velocity_cost=0.,landing_velocity_cost=0.),MODEL)
        new=Runtime(2,'cpu',.06,CFG,MODEL);s=state();s['motion'][:,0]=.05;s['gap'][:]=.1
        for p in (old,new):p.reset(torch.arange(2));p.control.phase[:]=Phase.BALANCE;p.invoked[:]=True
        for _ in range(10):
            a,b=old.advance(s,truth(raised=True)),new.advance(s,truth(raised=True))
            torch.testing.assert_close(old.observation(s),new.observation(s))
            torch.testing.assert_close(old.control.targets(torch.zeros(2,3),s),new.control.targets(torch.zeros(2,3),s))
            for key in ['done','failure','goal','complete','reason']:torch.testing.assert_close(a[key],b[key])
        torch.testing.assert_close(a['reward']-b['reward'],b['balance_velocity_cost'])
        self.assertTrue((b['balance_velocity_cost']>0).all())

    def test_cost_not_applied_to_rise_or_after_failure(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);s=state();p.reset(torch.arange(2));p.control.request_rise(s,torch.arange(2));p.invoked[:]=True;s['motion'][:,0]=1
        a=p.advance(s,truth());self.assertFalse(a['balance_velocity_cost'].any() or a['landing_velocity_cost'].any())
        p.control.phase[:]=Phase.BALANCE;t=truth(raised=True);t['body_contact'][:]=True
        a=p.advance(s,t);self.assertTrue(a['failure'].all());self.assertFalse(a['balance_velocity_cost'].any() or a['landing_velocity_cost'].any())

    def test_invalid_scales_rejected(self):
        for k in ['balance_velocity_error_tau_s','balance_velocity_error_scale_m_s','landing_velocity_error_scale_m_s']:
            with self.assertRaises(ValueError):replace(CFG,**{k:0.}).validate()

    def test_cap_preserves_small_errors_and_is_smooth_monotone_bounded(self):
        p=Runtime(2,'cpu',.06,replace(CFG,balance_velocity_cost_cap=1.5),MODEL)
        raw=torch.tensor([0.,.36,.7499,.7501,1.,3.,10.],requires_grad=True)
        cost=p.limit_balance_velocity_cost(raw)
        torch.testing.assert_close(cost[:3],raw[:3])
        self.assertTrue((cost[1:]>cost[:-1]).all())
        self.assertTrue((cost>=0).all() and (cost<1.5).all())
        slope=torch.autograd.grad(cost.sum(),raw)[0]
        self.assertTrue((slope>=0).all() and (slope<=1).all())
        self.assertLess(float((slope[2]-slope[3]).abs()),.001)
        legacy=Runtime(2,'cpu',.06,CFG,MODEL)
        torch.testing.assert_close(legacy.limit_balance_velocity_cost(raw),raw)

    def test_large_bias_keeps_real_support_positive_and_failure_cost_unchanged(self):
        cfg=replace(CFG,balance_velocity_cost_cap=1.5,landing_velocity_cost=0.)
        p=Runtime(2,'cpu',.06,cfg,MODEL);p.reset(torch.arange(2));p.level[:]=2;p.balance_duration[:]=30.
        p.control.phase[:]=Phase.BALANCE;p.invoked[:]=True;p.control.set_command([[-.75,0],[.75,0]])
        p.control.executed[:]=torch.tensor([1.2,25.,-25.]);p.control.previous[:]=torch.tensor([-1.,-25.,25.])
        s=state();s['motion'][:,0]=torch.tensor([.75,-.75]);s['gap'][:]=.1
        for _ in range(80):a=p.advance(s,truth(raised=True))
        self.assertTrue(a['support_truth'].all() and not a['done'].any())
        self.assertTrue((a['reward']>0).all());self.assertTrue((a['balance_velocity_cost']<=.09).all())
        t=truth(raised=True);t['body_contact'][:]=True;a=p.advance(s,t)
        self.assertTrue(a['failure'].all());torch.testing.assert_close(a['failure_penalty'],torch.full((2,),5.))
        self.assertFalse(a['balance_velocity_cost'].any())

    def test_cap_cannot_exceed_support_reward_or_accept_invalid_values(self):
        for cap in (-1.,2.,float('inf'),float('nan'),True,'1.5'):
            with self.assertRaises(ValueError):replace(CFG,balance_velocity_cost_cap=cap).validate()

if __name__=='__main__':unittest.main()
