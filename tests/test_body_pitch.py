import math
import unittest
import torch
from step_demo.body_pitch import world_pitch_state, sensor_pitch_state, pitch_costs
from step_demo.tensor_math import quat_mul, inverse_rotate


def rotation(axis, angle):
    q = torch.zeros(1, 4, dtype=torch.float64)
    q[0, axis] = math.sin(angle/2); q[0, 3] = math.cos(angle/2)
    return q


class BodyPitchTests(unittest.TestCase):
    def state(self, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate):
        q = quat_mul(quat_mul(rotation(2, yaw), rotation(1, pitch)), rotation(0, roll))
        # Independent Euler rotation axes expressed in the world frame.
        wx = roll_rate*math.cos(yaw)*math.cos(pitch)-pitch_rate*math.sin(yaw)
        wy = roll_rate*math.sin(yaw)*math.cos(pitch)+pitch_rate*math.cos(yaw)
        wz = yaw_rate-roll_rate*math.sin(pitch)
        return q, torch.tensor([[wx, wy, wz]], dtype=torch.float64)

    def test_compound_pose_and_motion_recover_world_pitch_derivative(self):
        for angles in [(.4, -.5, 1.1), (-.7, .9, -1.5), (0., .2, math.pi/2)]:
            q, w = self.state(*angles, .7, -1.3, .8)
            pitch, rate = world_pitch_state(q, w)
            ep, er = sensor_pitch_state(q, inverse_rotate(q, w))
            self.assertAlmostEqual(pitch.item(), angles[1], places=12)
            self.assertAlmostEqual(rate.item(), -1.3, places=12)
            torch.testing.assert_close(ep, pitch);torch.testing.assert_close(er, rate)

    def test_roll_rotation_does_not_become_pitch_rate_at_nonzero_yaw(self):
        q, w = self.state(.3, .5, .8, 2., 0., 0.)
        self.assertGreater(abs(w[0, 1].item()), .5)
        self.assertAlmostEqual(world_pitch_state(q, w)[1].item(), 0., places=12)

    def test_yaw_rotation_does_not_become_pitch_rate_at_nonzero_roll(self):
        q, w = self.state(.7, .3, .8, 0., 0., 2.)
        gyro = inverse_rotate(q, w)
        self.assertGreater(abs(gyro[0, 1].item()), .5)
        self.assertAlmostEqual(sensor_pitch_state(q, gyro)[1].item(), 0., places=12)

    def test_quaternion_sign_does_not_change_state(self):
        q, w = self.state(.2, -.4, .7, 1., 2., 3.)
        for a, b in zip(world_pitch_state(q, w), world_pitch_state(-q, w)):
            torch.testing.assert_close(a, b)

    def test_cost_is_soft_symmetric_and_zero_at_quiet_level(self):
        pitch = torch.tensor([0., .2, -.2, .6, -.6])
        rate, angle = pitch_costs(pitch, torch.zeros(5), .02)
        self.assertTrue((rate==0).all());self.assertTrue((angle[:3]==0).all())
        self.assertGreater(angle[3].item(), 0.);self.assertEqual(angle[3], angle[4])
        a, b = pitch_costs(pitch, torch.full((5,), 2.), .02)
        c, d = pitch_costs(pitch, torch.full((5,), -2.), .02)
        torch.testing.assert_close(a, c);torch.testing.assert_close(b, d)

    def test_physical_substep_cost_retains_short_rate_impulses(self):
        rate = torch.tensor([18.]+[0.]*9)
        cost = pitch_costs(torch.zeros(10), rate, .002)[0].sum()
        self.assertAlmostEqual(cost.item(), .0648, places=6)
        coarse = pitch_costs(torch.zeros(1), rate.mean().view(1), .02)[0]
        self.assertGreater(cost.item(), 9*coarse.item())

    def test_duration_scaling_and_no_negative_cost(self):
        p = torch.linspace(-1.2, 1.2, 50);v = torch.linspace(-20., 20., 50)
        for a, b in zip(pitch_costs(p, v, .002), pitch_costs(p, v, .02)):
            self.assertTrue((a>=0).all());torch.testing.assert_close(a*10, b)


if __name__ == '__main__':unittest.main()
