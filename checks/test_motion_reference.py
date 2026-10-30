"""Raw requests, reference slew, ordinary authority and paired direction remain distinct."""
from dataclasses import replace
import unittest
import torch
from upright.config import Config
from upright.controller import Controller,Phase
from upright.runtime import Runtime
from test_contracts import MODEL,state

CFG=Config(num_envs=32,balance_linear_accel_m_s2=.5,balance_yaw_accel_rad_s2=1.5)
class MotionReferenceContracts(unittest.TestCase):
    def test_raw_requests_preserved_and_reference_rate_bounded(self):
        c=Controller(2,'cpu',.06,CFG,MODEL,.1);c.phase[:]=Phase.BALANCE
        request=torch.tensor([[.2,.5],[-.2,-.5]]);c.set_command(request)
        c.update_motion_reference();torch.testing.assert_close(c.motion_command(),torch.tensor([[.03,.09],[-.03,-.09]]))
        for _ in range(8):c.update_motion_reference()
        torch.testing.assert_close(c.motion_command(),request)
        c.set_command(-request);before=c.motion_command().clone();c.update_motion_reference()
        torch.testing.assert_close(c.motion_command()-before,torch.tensor([[-.03,-.09],[.03,.09]]))
        c.targets(torch.zeros(2,3),state());torch.testing.assert_close(c.requested_command,-request)
        cached=c.executed_reference.clone();c.update_motion_reference();torch.testing.assert_close(c.executed_reference,cached)

    def test_ordinary_full_range_and_landing_reference_hold(self):
        c=Controller(2,'cpu',.06,CFG,MODEL,.1);s=state()
        c.set_command([[.75,0],[-.75,0]],hip=torch.tensor([1.1,-.9]));c.update_motion_reference()
        target=c.targets(torch.zeros(2,3),s)
        torch.testing.assert_close(target[:,1:],torch.tensor([[25.,25.],[-25.,-25.]]))
        c.phase[:]=Phase.BALANCE;c.motion_reference[:]=torch.tensor([.1,.2]);s['motion'][:]=torch.tensor([.07,.15])
        c.request_land(s,torch.arange(2));c.update_motion_reference()
        torch.testing.assert_close(c.motion_command(),torch.tensor([[.1,.2],[.1,.2]]))
        torch.testing.assert_close(c.transition_motion,torch.tensor([[.07,0],[.07,0]]))
        c.targets(torch.zeros(2,3),s);torch.testing.assert_close(c.effective_command,c.transition_motion)

    def test_disabled_reference_preserves_legacy_and_component_disable(self):
        c=Controller(2,'cpu',.06,Config(),MODEL,.1);c.phase[:]=Phase.BALANCE
        c.set_command([[.2,.5],[-.2,-.5]])
        torch.testing.assert_close(c.motion_command(),c.command)
        c=Controller(2,'cpu',.06,replace(CFG,balance_yaw_accel_rad_s2=0),MODEL,.1)
        c.phase[:]=Phase.BALANCE;c.set_command([[.2,.5],[-.2,-.5]]);c.update_motion_reference()
        torch.testing.assert_close(c.motion_command(),torch.tensor([[.03,.5],[-.03,-.5]]))

    def test_large_reversal_gets_enough_command_dwell(self):
        cfg=replace(CFG,capability_stages=[[.1,.3,.1,.1],[.75,.5,.3,.2]])
        p=Runtime(32,'cpu',.06,cfg,MODEL)
        torch.testing.assert_close(p.stage_dwell[2:],torch.tensor([2.,4.]))
        p.probe_level=3;p.reset(torch.arange(32));p.control.phase[:]=Phase.BALANCE
        p.balance_elapsed[:]=3;p.command_age[:]=3;p.before_action(state(32))
        self.assertLessEqual(float(p.control.command[:,0].abs().max()),.300001)
        p.balance_elapsed[:]=4.02;p.command_age[:]=4.02;p.before_action(state(32))
        torch.testing.assert_close(p.control.command[:,0],torch.full((32,),.75))
        self.assertTrue((p.control.motion_command()[:,0].abs()<.75).all())

if __name__=='__main__':unittest.main()
