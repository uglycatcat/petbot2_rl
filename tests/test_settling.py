"""Reward counterexamples from fast hip crossings and non-braking platform drives."""
import unittest
import torch
from step_demo.rewards import settled_progress, component_increment


class SettlingTests(unittest.TestCase):
    def credit(self, hip=.02, hip_speed=0., hold=0., stable=True, drove=False, speed=.2, stop=0.):
        p=torch.zeros(1,8);p[:,7]=stop
        return settled_progress(p,torch.tensor([hip]),torch.tensor([hip_speed]),torch.tensor([hold]),
            torch.tensor([stable]),torch.tensor([drove]),torch.tensor([speed]))

    def test_fast_zero_crossing_does_not_look_like_settled_home(self):
        transient=self.credit(hip=.02954,hip_speed=2.85)
        held=self.credit(hip=.0133,hip_speed=.1,hold=.2)
        self.assertLess(float(transient[0,5]),.001)
        self.assertGreater(float(held[0,5]),1.9)

    def test_home_dwell_cannot_be_earned_off_platform(self):
        self.assertEqual(float(self.credit(hip=0.,hold=.2,stable=False)[0,5]),0.)

    def test_braking_has_gradient_before_strict_stop_threshold(self):
        p=[float(self.credit(drove=True,speed=s)[0,7]) for s in [.3,.2,.1,.02,0.]]
        self.assertTrue(all(b>a for a,b in zip(p,p[1:])))
        self.assertEqual(p[-1],1.)
        self.assertEqual(float(self.credit(drove=True,speed=0.,stop=1.)[0,7]),2.)

    def test_early_stop_and_platform_departure_have_no_braking_credit(self):
        self.assertEqual(float(self.credit(drove=False,speed=0.)[0,7]),0.)

    def test_repeated_braking_and_invalid_terminal_cannot_farm(self):
        best=torch.zeros(1,8);total=0.
        for speed in [.3,.2,.1,.2,.1,.2,.1]:
            gain,best=component_increment(best,self.credit(stable=False,drove=True,speed=speed),torch.tensor([False]))
            total+=float(gain)
        self.assertAlmostEqual(total,2./3.,places=6)
        gain,new=component_increment(best,self.credit(hip=0.,hold=.2,drove=True,speed=0.,stop=1.),torch.tensor([True]))
        self.assertEqual(float(gain),0.);torch.testing.assert_close(new,best)


if __name__=='__main__':unittest.main()
