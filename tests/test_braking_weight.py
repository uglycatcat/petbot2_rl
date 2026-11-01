"""Keep stronger braking bounded and conditional on the original legal drive."""
import math
import unittest
import torch
from step_demo.rewards import settled_progress, component_increment
from step_demo.runtime import load_training
from step_demo.config import PROJECT_ROOT


class BrakingWeightTests(unittest.TestCase):
    def parts(self, speed, drove=True, weight=4., stopped=0.):
        p=torch.zeros(1,8);p[:,7]=stopped
        return settled_progress(p,torch.zeros(1),torch.zeros(1),torch.tensor([.2]),
            torch.tensor([True]),torch.tensor([drove]),torch.tensor([speed]),
            smooth_braking=True,brake_weight=weight)

    def test_no_early_braking_credit_and_other_components_unchanged(self):
        for v in [0.,.02,.3,.6,1.]:
            old=self.parts(v,weight=1.);new=self.parts(v)
            torch.testing.assert_close(old[:,:7],new[:,:7])
            self.assertEqual(float(self.parts(v,drove=False)[0,7]),0.)
            self.assertAlmostEqual(float(new[0,7]),4*math.exp(-v/.3),places=6)

    def test_observed_speed_change_has_stronger_credit(self):
        gain=float(self.parts(.3)[0,7]-self.parts(.6)[0,7])
        self.assertAlmostEqual(gain,4*(math.exp(-1)-math.exp(-2)),places=6)

    def test_repeated_braking_has_finite_total_and_hold_is_separate(self):
        best=torch.zeros(1,8);total=0.
        for v in [.6,.3,.6,.3,.1,.3,.1,0.,0.]:
            current=self.parts(v);current[:,:7]=0.
            gain,best=component_increment(best,current,torch.tensor([False]));total+=float(gain)
        self.assertAlmostEqual(total,4.,places=6)
        self.assertEqual(float(self.parts(0.,stopped=1.)[0,7]),5.)

    def test_failed_terminal_does_not_earn_larger_credit(self):
        best=torch.zeros(1,8)
        gain,new=component_increment(best,self.parts(0.,stopped=1.),torch.tensor([True]))
        self.assertEqual(float(gain),0.);torch.testing.assert_close(new,best)

    def test_all_configs_only_change_reward_against_round10(self):
        for name in ['seed70','seed71','gate','scale_gate']:
            old=load_training(PROJECT_ROOT/f'configs/round10_{name}.json')
            new=load_training(PROJECT_ROOT/f'configs/round13_{name}.json')
            self.assertEqual(new.pop('reward_version'),'brake_v13')
            self.assertEqual(old.pop('reward_version'),'brake_v10')
            self.assertEqual(new,old)


if __name__=='__main__':unittest.main()
