"""Regressions from landing re-lift rewards and lost platform-drive distance."""
from dataclasses import replace
import unittest
import torch

from step_demo.config import load_config
from step_demo.evaluation import CompletionTracker, EvaluationSample, Stage
from step_demo.progress import drive_memory
from step_demo.rewards import staged_progress, component_increment


class StageProgressTests(unittest.TestCase):
    def parts(self, x=.3, front_z=.02, landed=True, rear_z=0., contact=True,
              all_top=False, hip=.7, drive=0., stop=0.):
        return staged_progress(torch.tensor([x]),torch.zeros(1,2),
            torch.tensor([[front_z,front_z,rear_z,rear_z]]),.02,
            torch.full((1,2),landed,dtype=torch.bool),torch.full((1,2),contact,dtype=torch.bool),
            torch.ones(1,2,dtype=torch.bool),torch.tensor([all_top]),torch.tensor([hip]),
            torch.tensor([True]),torch.tensor([drive]),torch.tensor([stop]))

    def test_re_lift_after_landing_cannot_outscore_supported_forward_motion(self):
        supported=self.parts(x=.35,front_z=.02)
        bouncing=self.parts(x=.35,front_z=.04)
        torch.testing.assert_close(supported,bouncing)
        # The old combined maximum hid forward progress when clearance dropped.
        before=self.parts(x=.30,front_z=.07,landed=False)
        gain,best=component_increment(before,supported,torch.tensor([False]))
        self.assertAlmostEqual(float(gain),.5,places=5)

    def test_rear_flight_and_hip_home_before_ascent_have_no_credit(self):
        airborne=self.parts(rear_z=.04,contact=False,hip=0.)
        self.assertEqual(float(airborne[:,3:].sum()),0.)
        supported=self.parts(rear_z=.02,all_top=True,hip=0.)
        self.assertEqual(float(supported[:,3:5].sum()),3.)
        self.assertEqual(float(supported[:,5]),2.)

    def test_invalid_milestones_and_repeated_progress_do_not_pay(self):
        previous=torch.zeros(1,8);current=self.parts(x=10.,rear_z=.02,all_top=True,hip=0.,drive=2.,stop=2.)
        current[:,1:3]=1.
        gain,best=component_increment(previous,current,torch.tensor([True]))
        self.assertEqual(float(gain),0.)
        gain,best=component_increment(previous,current,torch.tensor([False]))
        self.assertEqual(float(gain),16.)
        gain,_=component_increment(best,current,torch.tensor([False]))
        self.assertEqual(float(gain),0.)

    def sample(self, **changes):
        s=EvaluationSample(.02,.5,0.,0.,.2,0.,(0.,0.),(True,True),True,False,False,False,False,True)
        return replace(s,**changes)

    def test_contact_gap_preserves_distance_but_never_counts_as_stopped_support(self):
        t=CompletionTracker(load_config().acceptance,.06)
        t.update(self.sample())
        t.update(self.sample(forward_m=.58,four_wheels_on_top=False))
        self.assertEqual(t.drive_start_m,.5)
        self.assertEqual(t.update(self.sample(forward_m=.67)),Stage.STOPPING)
        for _ in range(24):t.update(self.sample(forward_m=.67,speed_m_s=0.))
        t.update(self.sample(forward_m=.67,speed_m_s=0.,four_wheels_on_top=False))
        self.assertEqual(t.stopped_s,0.)
        for _ in range(25):stage=t.update(self.sample(forward_m=.67,speed_m_s=0.))
        self.assertEqual(stage,Stage.COMPLETE)

    def test_geometry_pose_and_long_support_loss_clear_distance(self):
        for changes in [dict(near_top_during_contact_gap=False),dict(lateral_m=.03),dict(hip_error_rad=(.1,.1))]:
            t=CompletionTracker(load_config().acceptance,.06);t.update(self.sample())
            t.update(self.sample(four_wheels_on_top=False,**changes));self.assertIsNone(t.drive_start_m)
        t=CompletionTracker(load_config().acceptance,.06);t.update(self.sample())
        for _ in range(4):t.update(self.sample(four_wheels_on_top=False))
        self.assertIsNone(t.drive_start_m)

    def test_tensor_memory_matches_scalar_contact_gap_sequence(self):
        t=CompletionTracker(load_config().acceptance,.06);start=torch.tensor([float('nan')]);gap=torch.zeros(1)
        for ready,near in [(True,True),(False,True),(False,True),(True,True),
                           (False,False),(True,True)]:
            s=self.sample(four_wheels_on_top=ready,near_top_during_contact_gap=near);t.update(s)
            start,gap=drive_memory(start,gap,torch.tensor([s.forward_m]),torch.tensor([ready]),
                                  torch.tensor([True]),torch.tensor([near]),s.dt_s)
            if t.drive_start_m is None:self.assertTrue(bool(torch.isnan(start).all()))
            else:self.assertAlmostEqual(float(start),t.drive_start_m)

    def test_physical_failure_remains_terminal_even_inside_contact_grace(self):
        t=CompletionTracker(load_config().acceptance,.06);t.update(self.sample())
        self.assertEqual(t.update(self.sample(rear_support_lost=True)),Stage.FAILED)


if __name__ == '__main__':
    unittest.main()
