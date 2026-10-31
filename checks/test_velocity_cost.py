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

if __name__=='__main__':unittest.main()
