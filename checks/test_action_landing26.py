"""Full balance authority and process-sensitive landing quality contracts."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config, ROOT
from upright.controller import Phase, centered_common_action
from upright.runtime import Runtime
from test_contracts import MODEL, state, truth

CFG=load_config(ROOT/'configs/round26.json')

class ActionLanding26(unittest.TestCase):
    def test_centering_full_range_monotone_mirrored_and_zero_identity(self):
        a=torch.linspace(-1,1,201)[:,None];c=torch.linspace(-.8,.8,17)[None,:]
        u=centered_common_action(a,c)
        self.assertTrue((u.abs()<=1.000001).all())
        torch.testing.assert_close(u[0],-torch.ones(17));torch.testing.assert_close(u[-1],torch.ones(17))
        self.assertTrue((u[1:]-u[:-1]>0).all())
        torch.testing.assert_close(centered_common_action(-a,-c),-u)
        torch.testing.assert_close(centered_common_action(a,torch.zeros_like(c)),a.expand_as(u))
        torch.testing.assert_close(centered_common_action(torch.zeros_like(c),c),c)

    def test_balance_reference_mapping_and_stage_ownership(self):
        p=Runtime(6,'cpu',.06,CFG,MODEL);c=p.control;s=state(6)
        c.phase[:]=torch.tensor([0,1,2,3,4,5]);c.command[:,0]=.75;c.motion_reference[:,0]=.375
        c.executed[:,1:]=7.5;c.handoff_start[:,1:]=7.5;c.age[:]=1
        result=c.targets(torch.zeros(6,3),s).clone()
        # BALANCE uses limited reference .375, not the raw .75 request.
        self.assertAlmostEqual(float(result[2,1]),7.5,places=5)
        self.assertEqual(float(result[1,1]),0);self.assertEqual(float(result[3,1]),0)
        for i in [0,4,5]:self.assertEqual(float(result[i,1]),25)
        c.phase[:]=Phase.BALANCE;c.motion_reference[:,0]=.75;c.executed[:,1:]=0
        a=torch.zeros(6,3);a[:,1]=-1
        self.assertTrue((c.targets(a,s)[:,1:]==-9).all()) # retained full negative demand, existing slew
        for _ in range(3):c.targets(a,s)
        self.assertTrue((c.executed[:,1:]==-25).all())

    def test_ground_window_retains_slide_until_transfer_but_airborne_is_free(self):
        p=Runtime(1,'cpu',.06,CFG,MODEL);s=state(1);t=truth(1);c=p.control
        c.phase[:]=Phase.LAND;s['motion'][:,0]=.6;s['gap'][:]=.1
        p.advance(s,t);self.assertEqual(float(p.landing_quality_min[0]),1)
        s['gap'][:]=.02;p.advance(s,t);bad=p.landing_quality_min.clone()
        self.assertLess(float(bad[0]),.04)
        s['gap'][:]=0;s['motion'][:]=0;c.ground_run[:]=CFG.ground_confirm_s
        m=p.advance(s,t)
        self.assertTrue(m['entered_handoff'].all())
        torch.testing.assert_close(m['landing_quality_bonus'],4*bad)
        torch.testing.assert_close(p.landing_quality_min,bad)
        self.assertFalse(p.advance(s,t)['landing_quality_bonus'].any())
        p.reset(torch.tensor([0]));self.assertEqual(float(p.landing_quality_min[0]),1)

    def test_deceleration_quality_and_critic_memory_stay_private(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);s=state(2)
        p.control.phase[:]=Phase.LAND;p.control.transition_motion[:,0]=torch.tensor([.6,-.6])
        s['motion'][:,0]=torch.tensor([.2,-.2]);s['gap'][:]=.02
        p.advance(s,truth(2));torch.testing.assert_close(p.landing_quality_min,torch.ones(2))
        obs=p.observation(s).clone();p.landing_quality_min[:]=.2
        torch.testing.assert_close(p.observation(s),obs)
        self.assertEqual(p.critic_state().shape,(2,27))
        torch.testing.assert_close(p.critic_state()[:,-1],torch.full((2,),.2))
        legacy=Runtime(2,'cpu',.06,load_config(ROOT/'configs/round25.json'),MODEL)
        self.assertEqual(legacy.critic_state().shape,(2,26))

    def test_disabled_window_keeps_instantaneous_bonus(self):
        p=Runtime(1,'cpu',.06,replace(CFG,landing_quality_window=False),MODEL);s=state(1)
        p.control.phase[:]=Phase.LAND;p.landing_quality_min[:]=.1
        p.control.ground_run[:]=CFG.ground_confirm_s
        m=p.advance(s,truth(1));self.assertEqual(float(m['landing_quality_bonus'][0]),4.)

    def test_config_rejects_invalid_controls(self):
        for value in [-.1,.81,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):replace(CFG,balance_common_center_fraction=value).validate()
        with self.assertRaises(ValueError):replace(CFG,landing_quality_window=1).validate()

if __name__=='__main__':unittest.main()
