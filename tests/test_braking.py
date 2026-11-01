"""Counterexamples from round09's high-speed plateau and finite stop diagnostic."""
import math
import unittest
import torch
from step_demo.rewards import settled_progress, component_increment
from step_demo.runtime import load_training
from step_demo.config import PROJECT_ROOT


class SmoothBrakingTests(unittest.TestCase):
    def parts(self, speed, drove=True, stop=0., smooth=True):
        p=torch.zeros(1,8);p[:,7]=stop
        return settled_progress(p,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([.2]),
            torch.tensor([True]),torch.tensor([drove]),torch.as_tensor([speed]),smooth_braking=smooth)

    def test_observed_high_speeds_have_nonzero_monotonic_signal(self):
        speeds=[.75,.65,.58,.50,.3,.2,.1,.02,0.]
        credits=[self.parts(v)[0,7].item() for v in speeds]
        self.assertGreater(credits[0],0.)
        self.assertTrue(all(b>a for a,b in zip(credits,credits[1:])))
        self.assertEqual(credits[-1],1.)
        self.assertAlmostEqual(credits[2],math.exp(-.58/.3),places=6)

    def test_observed_small_deceleration_is_no_longer_in_dead_zone(self):
        slow=self.parts(.57)[0,7];fast=self.parts(.59)[0,7]
        self.assertGreater(float(slow-fast),0.)
        self.assertEqual(float(self.parts(.57,smooth=False)[0,7]),0.)

    def test_no_credit_before_required_drive(self):
        self.assertEqual(float(self.parts(.01,drove=False)[0,7]),0.)
        self.assertEqual(float(self.parts(0.,drove=False)[0,7]),0.)

    def test_credit_ceiling_and_separate_stop_hold_are_preserved(self):
        self.assertEqual(float(self.parts(0.,stop=1.)[0,7]),2.)
        for v in [0.,.01,.3,.6,2.,100.]:
            self.assertTrue(0.<=float(self.parts(v)[0,7])<=1.)

    def test_repeated_holding_and_braking_cannot_farm(self):
        best=torch.zeros(1,8);total=0.
        for v in [.6,.5,.6,.4,.5,.4,.4]:
            current=self.parts(v);current[:,:7]=0.
            gain,best=component_increment(best,current,torch.tensor([False]));total+=float(gain)
        self.assertAlmostEqual(total,math.exp(-.4/.3),places=6)

    def test_invalid_terminal_cannot_earn_brake_or_stop_credit(self):
        best=torch.zeros(1,8)
        gain,new=component_increment(best,self.parts(0.,stop=1.),torch.tensor([True]))
        self.assertEqual(float(gain),0.);torch.testing.assert_close(new,best)

    def test_other_progress_components_are_unchanged(self):
        old=self.parts(.58,smooth=False);new=self.parts(.58)
        torch.testing.assert_close(old[:,:7],new[:,:7])

    def test_formal_config_changes_only_reward_version(self):
        for seed in [70,71]:
            old=load_training(PROJECT_ROOT/f'configs/round09_seed{seed}.json')
            new=load_training(PROJECT_ROOT/f'configs/round10_seed{seed}.json')
            self.assertEqual(new.pop('reward_version'),'brake_v10')
            old.pop('reward_version');self.assertEqual(old,new)


if __name__=='__main__':unittest.main()
