"""Counterexamples: phase regression, repeated flight, and early-failure escape."""
import unittest
import torch
from step_demo.staged_task import advance_intent, costs, terminal_value, FrontFlight, MAX_RATE, body_pitch


class RestartTaskTests(unittest.TestCase):
    def test_stop_cannot_be_disabled_by_losing_support_or_home(self):
        intent=torch.tensor([3]);no=torch.tensor([False])
        self.assertEqual(advance_intent(intent,no,no,no).item(),3)

    def test_stop_costs_are_zero_before_stop_and_bounded(self):
        n=2000
        phase=torch.arange(n)%4
        large=torch.randn(n)*100
        joint=torch.randn(n,2)*100
        rate,terms=costs(phase,large.abs(),large,joint,joint,large.abs(),large,
                         large,large,large,large,torch.ones(n,dtype=torch.bool))
        self.assertTrue((rate<=MAX_RATE).all())
        self.assertTrue((terms[phase<3,:2]==0).all())
        self.assertTrue((terms[phase<2,2:4]==0).all())

    def test_failed_early_cannot_save_future_running_cost(self):
        # Even a worst-cost continuation to a later failure is no worse than
        # immediate failure, independently of stage or termination time.
        for first in [0.,.5,3.,10.]:
            for later in [first,12.,15.]:
                early=terminal_value(torch.tensor([False]),torch.tensor([True]),torch.tensor([first]),15.)
                late=terminal_value(torch.tensor([False]),torch.tensor([True]),torch.tensor([later]),15.)
                self.assertAlmostEqual(early.item(),late.item()-MAX_RATE*(later-first),places=5)

    def test_success_dominates_timeout_even_at_max_running_cost(self):
        # Progress is nonnegative and <=24; no terminal bonus for rejected episodes.
        success=terminal_value(torch.tensor([True]),torch.tensor([False]),torch.tensor([15.]),15.).item()-MAX_RATE*15.
        failed=terminal_value(torch.tensor([False]),torch.tensor([True]),torch.tensor([15.]),15.).item()+24.
        # This is an intentionally strict global check, used to select the bonus.
        self.assertGreater(success,failed)

    def test_real_flight_once_vs_contact_noise_and_relift(self):
        f=FrontFlight(1,'cpu');edge=torch.tensor([.3]);p=torch.zeros(1,2,3)
        force=torch.zeros_like(p);top=torch.zeros_like(p)
        # Contact loss at floor is not flight.
        for _ in range(20): f.update(p,force,top,edge,.02,.002)
        self.assertEqual(f.count.tolist(),[[0,0]])
        p[:,:,2]=.03
        for _ in range(11): f.update(p,force,top,edge,.02,.002)
        self.assertEqual(f.count.tolist(),[[1,1]])
        p[:,:,0]=.4;p[:,:,2]=.02;force[:,:,2]=1;top[:,:,2]=1
        for _ in range(11): f.update(p,force,top,edge,.02,.002)
        self.assertTrue(f.complete.item())
        p[:,:,2]=.04;force.zero_();top.zero_()
        for _ in range(11): f.update(p,force,top,edge,.02,.002)
        self.assertTrue(f.invalid.item())
        f.reset(torch.tensor([0]));self.assertFalse(f.invalid.item())

    def test_world_pitch_uses_body_pose_and_world_rate(self):
        import math
        # yaw90 followed by pitch0.3: pure pitch rate is along world -X.
        from step_demo.tensor_math import quat_mul,rotation_step
        q=quat_mul(rotation_step(torch.tensor([[0.,0.,math.pi/2]]),1.),
                   rotation_step(torch.tensor([[0.,.3,0.]]),1.))
        pitch,rate=body_pitch(q,torch.tensor([[-2.,0.,0.]]))
        self.assertAlmostEqual(pitch.item(),.3,places=6)
        self.assertAlmostEqual(rate.item(),2.,places=6)

    def test_auxiliary_cannot_be_launched_as_formal_climb_or_without_real_hold(self):
        import json,tempfile
        from pathlib import Path
        from step_demo.config import PROJECT_ROOT
        from tools.campaign import validate_plan
        with tempfile.TemporaryDirectory(prefix='step-aux-plan-') as directory:
            root=Path(directory);(root/'source/configs').mkdir(parents=True)
            (root/'source/configs/aux.json').write_text((PROJECT_ROOT/'configs/restart15_aux_seed70.json').read_text())
            plan=dict(schema='step_batch_v1',kind='formal',hypothesis='stop diagnostic',
                jobs=[dict(id='seed70',gpu=0,training_config='configs/aux.json',eval_envs=64)])
            with self.assertRaises(ValueError):validate_plan(plan,root)
            plan['kind']='diagnostic'
            with self.assertRaises(ValueError):validate_plan(plan,root)
            plan['jobs'][0]['hold_after_success_s']=2.
            validate_plan(plan,root)


if __name__=='__main__':unittest.main()
