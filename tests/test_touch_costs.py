"""Counterexamples: early top impacts and fast rear-ascent hip motion."""
import unittest
import torch
from step_demo.contacts import early_top_contact, forbidden_front_contact
from step_demo.rewards import touch_costs

class TouchCostsTests(unittest.TestCase):
    def costs(self, forces=(10.,10.), velocity=(0.,0.), contact=(True,True), touched=(True,True), four=False, dt=.002, weight=20.):
        return touch_costs(torch.tensor([forces]),torch.tensor([contact]),torch.tensor([weight]),
            torch.tensor([velocity]),torch.tensor([touched]),torch.tensor([four]),dt)

    def test_top_touch_precedes_whole_wheel_clearance_without_accepting_edge_hit(self):
        normal=torch.tensor([[[0.,0.,120.],[0.,0.,100.]]]);point=torch.tensor([[.03,0.]])
        bad=forbidden_front_contact(normal,point,torch.tensor([[-.02,-.02]]),torch.tensor([0.]))
        touch=early_top_contact(normal,point,torch.tensor([0.]),bad)
        self.assertEqual(touch.tolist(),[[True,False]])

    def test_nan_contact_location_cannot_create_touch(self):
        value=early_top_contact(torch.tensor([[[0.,0.,10.]]]),torch.tensor([[float('nan')]]),torch.tensor([0.]),torch.tensor([[False]]))
        self.assertFalse(bool(value.any()))

    def test_static_load_and_unrelated_floor_contact_do_not_pay_impact_cost(self):
        self.assertEqual(float(self.costs()[0]),0.)
        self.assertEqual(float(self.costs(forces=(120.,120.),contact=(False,False))[0]),0.)

    def test_short_force_peak_is_accounted_at_physics_substeps(self):
        impact=sum(float(self.costs(forces=f)[0]) for f in [(10.,10.),(120.,20.),(10.,10.)])
        self.assertAlmostEqual(impact,.002*25.,places=7)
        self.assertEqual(float(self.costs(forces=(10.,10.),dt=.006)[0]),0.)

    def test_repeated_impacts_are_not_hidden_by_a_first_landing_latch(self):
        first=float(self.costs(forces=(60.,20.))[0]);again=float(self.costs(forces=(60.,20.))[0])
        self.assertGreater(first,0.);self.assertEqual(first,again)

    def test_motion_cost_applies_only_during_the_supported_ascent_transition(self):
        self.assertEqual(float(self.costs(velocity=(10.,10.),touched=(True,False))[1]),0.)
        self.assertEqual(float(self.costs(velocity=(10.,10.),four=True)[1]),0.)
        slow=float(self.costs(velocity=(3.,3.))[1]);fast=float(self.costs(velocity=(10.,10.))[1])
        self.assertGreater(fast,slow);self.assertAlmostEqual(fast,.002*.027*100.,places=7)

    def test_side_symmetry_and_load_normalization(self):
        a=self.costs(forces=(60.,10.),velocity=(3.,9.));b=self.costs(forces=(10.,60.),velocity=(9.,3.))
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
        torch.testing.assert_close(a[0],self.costs(forces=(120.,20.),weight=40.)[0])

if __name__=='__main__':unittest.main()
