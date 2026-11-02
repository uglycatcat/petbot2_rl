import unittest
import torch
from step_demo.front_motion import FrontMotion
from step_demo.stop_memory import wheel_request


class FrontMotionTests(unittest.TestCase):
    def setUp(self):
        self.m = FrontMotion(2, 'cpu')

    def step(self, z, x=-.1, force=0., platform=0., count=10):
        v = torch.zeros(2, 2, 3); v[..., 0] = x; v[..., 2] = z
        f = torch.zeros_like(v); f[..., 2] = force
        p = torch.zeros_like(v); p[..., 2] = platform
        for _ in range(count):self.m.update(v, f, p, torch.zeros(2), .02, .002)

    def test_short_contact_drop_and_near_surface_are_not_liftoff(self):
        self.step(.001, count=30);self.step(.005, count=9);self.step(0., force=1.)
        self.assertFalse(self.m.count.any());self.assertFalse(self.m.invalid.any())

    def test_one_flight_and_top_landing_are_required(self):
        self.step(.025)
        self.assertTrue((self.m.count==1).all());self.assertFalse(self.m.complete.any())
        self.step(.02, x=.1, force=1., platform=1., count=9)
        self.assertFalse(self.m.complete.any())
        self.step(.02, x=.1, force=1., platform=1., count=1)
        self.assertTrue(self.m.complete.all());self.assertFalse(self.m.invalid.any())

    def test_ground_return_rejects_the_first_wasted_lift(self):
        self.step(.015);self.step(0., force=1.)
        self.assertTrue(self.m.returned_floor.all());self.assertTrue(self.m.invalid.all())
        self.step(.03);self.assertTrue((self.m.count==2).all())

    def test_liftoff_after_top_landing_is_rejected(self):
        self.step(.025);self.step(.02, x=.1, force=1., platform=1.)
        self.step(.025, x=.1)
        self.assertTrue(self.m.invalid.all());self.assertFalse(self.m.complete.any())

    def test_one_bad_side_invalidates_episode_and_reset_is_local(self):
        self.step(.025);self.step(0., force=1.)
        self.m.reset(torch.tensor([0]))
        self.assertEqual(self.m.invalid.tolist(), [False, True])
        self.assertEqual(self.m.count[0].tolist(), [0, 0])
        self.m.reset(torch.tensor([1]));self.m.returned_floor[1, 0] = True
        self.assertEqual(self.m.invalid.tolist(), [False, True])

    def test_platform_force_without_top_geometry_cannot_confirm_landing(self):
        self.step(.03);self.step(.02, x=-.01, force=1., platform=1.)
        self.assertFalse(self.m.complete.any())

    def test_zero_margin_preserves_full_range_and_exact_stop(self):
        a = torch.tensor([-1., -.95, -.9, -.89, 0., 1.])
        v = wheel_request(a, 'zero_margin')
        self.assertTrue((v[:3]==0).all());self.assertGreater(float(v[3]), 0.)
        self.assertAlmostEqual(float(v[4]), 25*.9/1.9, places=5)
        self.assertEqual(float(v[-1]), 25.)


if __name__ == '__main__':unittest.main()
