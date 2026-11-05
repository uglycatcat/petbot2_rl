"""Task-contract regressions derived from the independent audit.

Synthetic physics samples test state semantics, not simulator reachability.
Real PhysX validation remains a separate gate.
"""
import unittest

import torch

from step_demo.config import load_config
from step_demo.task import Failure, Phase, PhysicsSample, StepTask


def sample(n=1):
    z = lambda *s: torch.zeros(n, *s)
    return PhysicsSample(z(), z(), z(), z(2), z(2), z(2), z(), z(), z(), z(),
                         z(4, 3), z(4, 3), z(4), z(4, 3), z(4, 3),
                         torch.full((n, 4, 3), float('nan')), torch.zeros(n, dtype=torch.bool))


def task(n=1):
    c = load_config()
    return StepTask(n, 'cpu', c.acceptance, c.terrain, torch.full((n,), .3), torch.zeros(n))


def put_on_top(s, x=.5):
    s.forward[:] = x
    s.minimum[:, :, 0] = x-.06
    s.maximum[:, :, 0] = x+.06
    s.minimum[:, :, 2] = .02
    s.maximum[:, :, 2] = .08
    s.force[:, :, 2] = 1.
    s.platform_force[:, :, 2] = 1.
    s.contact_point[:] = 0.
    s.contact_point[:, :, 0] = x
    s.contact_point[:, :, 2] = .02
    s.clearance[:] = 0.


def fly_front(s):
    s.clearance[:, :2] = .01
    s.minimum[:, :2, 2] = .03
    s.force[:, :2] = 0.
    s.platform_force[:, :2] = 0.
    s.contact_point[:, :2] = float('nan')


def route_to_stop(t, s):
    # A physically specified event sequence, not direct writes to task history.
    s.minimum[:, :, 0] = .1
    s.maximum[:, :, 0] = .16
    s.force[:, 2:, 2] = 1.
    fly_front(s)
    rewards = []
    for _ in range(15): rewards.append(t.advance(s).clone())
    put_on_top(s, .5)
    rewards.append(t.advance(s).clone())
    for k in range(1, 17):
        s.forward[:] = .5+k*.01
        s.speed[:] = .5
        rewards.append(t.advance(s).clone())
    assert (t.phase == Phase.STOP).all()
    s.speed[:] = 0.
    return rewards


class TaskContractTests(unittest.TestCase):
    def edge_flight(self):
        t, s = task(), sample()
        s.forward[:] = .16
        s.minimum[:, :2, 0] = .21
        s.maximum[:, :2, 0] = .30
        s.force[:, 2:, 2] = 1.
        fly_front(s)
        t.advance(s)
        return t, s

    def test_vertical_hop_does_not_exhaust_crossing_progress(self):
        t, s = self.edge_flight()
        hop = t.potential(s).item()
        self.assertLess(hop, 3.)
        s.minimum[:, :2, 2] = .10  # Extra height without crossing cannot pay.
        self.assertAlmostEqual(t.potential(s).item(), hop, places=5)
        s.minimum[:, :2, 0] += .045; s.maximum[:, :2, 0] += .045
        self.assertGreater(t.potential(s).item(), hop+.5)

    def test_crossing_requires_both_fronts_clear_and_rear_support(self):
        t, s = self.edge_flight()
        s.minimum[:, :2, 0] += .045; s.maximum[:, :2, 0] += .045
        supported = t.potential(s).item()
        s.clearance[:, 2] = .002
        self.assertLess(t.potential(s).item(), supported)
        s.clearance[:, 2] = .005; s.force[:, 2] = 0.
        self.assertAlmostEqual(t.potential(s).item(), 2., places=5)
        best = t.best_legal_potential.item()
        t.advance(s)
        self.assertEqual(t.failure.item(), Failure.REAR_AIR)
        self.assertEqual(t.best_legal_potential.item(), best)
        t, s = self.edge_flight()
        s.minimum[:, :2, 0] += .045; s.maximum[:, :2, 0] += .045
        s.minimum[:, 0, 2] = .002  # One low wheel must not earn crossing credit.
        self.assertLess(t.potential(s).item(), 2.5)

    def test_actual_front_landing_strictly_exceeds_any_airborne_pose(self):
        t, s = self.edge_flight()
        s.minimum[:, :2, 0] = .40; s.maximum[:, :2, 0] = .49
        t.advance(s)
        airborne = t.potential(s).item()
        self.assertAlmostEqual(airborne, 4., places=5)
        s.minimum[:, :2, 2] = .02
        s.force[:, :2, 2] = 1.; s.platform_force[:, :2, 2] = 1.
        s.contact_point[:, :2] = torch.tensor([.44, 0., .02])
        t.advance(s)
        self.assertEqual(t.phase.item(), Phase.REAR_ASCENT)
        self.assertGreaterEqual(t.potential(s).item()-airborne, .99)
        self.assertAlmostEqual(t.potential(s).item(), 5., places=5)

    def test_rolling_forward_without_flight_cannot_earn_crossing_credit(self):
        t, s = task(), sample()
        s.forward[:] = .16
        s.minimum[:, :2, 0] = .40; s.maximum[:, :2, 0] = .49
        self.assertAlmostEqual(t.potential(s).item(), 2., places=5)

    def test_loaded_contact_not_rejected_by_aabb_penetration(self):
        t, s = task(), sample()
        route_to_stop(t, s)
        s.minimum[:, :2, 2] = torch.tensor([.017984977, .017961534])
        s.force[:, :2, 2] = torch.tensor([2.807093, 4.159694])
        s.platform_force.copy_(s.force)
        for _ in range(1250): t.advance(s)
        self.assertTrue(t.completed.item())
        self.assertAlmostEqual(t.drive_distance.item(), .15, places=6)
        self.assertEqual(t.failure.item(), Failure.NONE)

    def test_stop_memory_survives_long_support_gap_and_bad_hip(self):
        t, s = task(), sample()
        route_to_stop(t, s)
        for _ in range(100): t.advance(s)
        for _ in range(50):
            s.force[:, 2] = 0.; s.platform_force[:, 2] = 0.
            s.contact_point[:, 2] = float('nan')
            s.hip[:] = .06
            t.advance(s)
        self.assertEqual(t.phase.item(), Phase.STOP)
        self.assertEqual(t.hold_ticks.item(), 0)
        self.assertAlmostEqual(t.drive_distance.item(), .15, places=6)
        put_on_top(s, .66); s.hip.zero_()
        for _ in range(1250): t.advance(s)
        self.assertTrue(t.completed.item())

    def test_short_loaded_landing_does_not_merge_two_flights(self):
        t, s = task(), sample()
        fly_front(s)
        for _ in range(15): t.advance(s)
        put_on_top(s)
        for _ in range(9): t.advance(s)  # 18 ms, audit counterexample
        fly_front(s)
        for _ in range(15): t.advance(s)
        self.assertEqual(t.flight_count.tolist(), [[2, 2]])
        self.assertEqual(t.failure.item(), Failure.FRONT_FLIGHT)
        self.assertFalse(t.completed.item())

    def test_no_force_at_floor_is_not_flight(self):
        t, s = task(), sample()
        for _ in range(100): t.advance(s)
        self.assertEqual(t.flight_count.tolist(), [[0, 0]])
        self.assertEqual(t.failure.item(), 0)

    def test_even_short_confirmed_relift_and_rear_flight_fail(self):
        t, s = task(), sample()
        route_to_stop(t, s); fly_front(s)
        t.advance(s)  # A single real 2 ms relift cannot be hidden by debounce.
        self.assertEqual(t.failure.item(), Failure.FRONT_FLIGHT)
        t, s = task(), sample()
        s.clearance[:, 2] = .005
        t.advance(s)
        self.assertEqual(t.failure.item(), Failure.REAR_AIR)

    def test_near_vertical_face_does_not_erase_floor_takeoff(self):
        t, s = task(), sample()
        fly_front(s)
        s.clearance[:, :2] = .001  # Close to face, still 30 mm above floor.
        for _ in range(15): t.advance(s)
        self.assertEqual(t.flight_count.tolist(), [[1,1]])
        put_on_top(s)
        t.advance(s)
        self.assertEqual(t.failure.item(), 0)
        self.assertTrue(t.landed_top.all())

    def test_floor_return_and_rolling_onto_top_cannot_succeed(self):
        t, s = task(), sample()
        fly_front(s)
        for _ in range(15): t.advance(s)
        s.minimum[:, :2, 2] = 0.; s.force[:, :2, 2] = 1.; s.clearance[:] = 0.
        t.advance(s)
        self.assertEqual(t.failure.item(), Failure.FRONT_FLIGHT)
        t.reset(torch.tensor([0])); put_on_top(s)
        t.advance(s)
        self.assertEqual(t.failure.item(), Failure.FRONT_FLIGHT)

    def test_top_load_cannot_hide_simultaneous_front_face_normal(self):
        t, s = task(), sample()
        fly_front(s); t.advance(s)
        put_on_top(s, .33)
        s.platform_force[:, 0, 0] = 1.
        t.advance(s)
        self.assertEqual(t.failure.item(), Failure.FRONT_COLLISION)

    def test_physics_substep_velocity_spike_resets_hold(self):
        t, s = task(), sample()
        route_to_stop(t, s)
        for _ in range(1249): t.advance(s)
        self.assertFalse(t.completed.item())
        s.speed[:] = .021; t.advance(s)
        s.speed[:] = 0.; t.advance(s)
        self.assertEqual(t.hold_ticks.item(), 1)
        self.assertFalse(t.completed.item())
        for _ in range(1249): t.advance(s)
        self.assertTrue(t.completed.item())

    def test_exact_deadline_success_has_single_priority(self):
        t, s = task(2), sample(2)
        route_to_stop(t, s)
        t.ticks[:] = t.deadline_ticks-1
        t.hold_ticks[:] = 1249
        s.speed[1] = .021
        t.advance(s)
        self.assertEqual(t.completed.tolist(), [True, False])
        self.assertEqual(t.failure.tolist(), [0, Failure.TIMEOUT])
        self.assertEqual(t.phase.tolist(), [Phase.COMPLETE, Phase.FAILED])

    def test_real_failure_wins_over_simultaneous_success(self):
        t, s = task(), sample()
        route_to_stop(t, s); t.hold_ticks[:] = 1249
        s.body_invalid[:] = True
        t.advance(s)
        self.assertFalse(t.completed.item())
        self.assertEqual(t.failure.item(), Failure.FALL)

    def test_complete_episode_returns_and_terminal_latching(self):
        t, s = task(), sample()
        rewards = route_to_stop(t, s)
        for _ in range(1250): rewards.append(t.advance(s).clone())
        self.assertTrue(t.completed.item())
        self.assertAlmostEqual(torch.stack(rewards).sum().item(), 40., places=4)
        before = t.critic_memory().clone()
        self.assertEqual(t.advance(s).item(), 0.)
        self.assertTrue(torch.equal(before, t.critic_memory()))
        t, s = task(), sample(); rewards = route_to_stop(t, s)
        credit = t.best_legal_potential.item()
        s.body_invalid[:] = True; rewards.append(t.advance(s))
        self.assertAlmostEqual(torch.stack(rewards).sum().item(), -10.+.5*credit, places=4)
        self.assertEqual(t.best_legal_potential.item(), credit)

    def test_qualified_forward_and_unqualified_reverse_cannot_farm_drive(self):
        t, s = task(), sample()
        fly_front(s); t.advance(s); put_on_top(s, .5); t.advance(s)
        self.assertEqual(t.phase.item(), Phase.DRIVE)
        s.speed[:] = .25
        for _ in range(16):
            s.hip.zero_()
            for k in range(1,21):
                s.forward[:] = .5+k*.0005; t.advance(s)
            s.hip[:] = .06
            for k in range(19,-1,-1):
                s.forward[:] = .5+k*.0005; t.advance(s)
            s.hip.zero_(); t.advance(s)
        s.speed.zero_()
        for _ in range(1250): t.advance(s)
        self.assertFalse(t.completed.item())
        self.assertEqual(t.phase.item(), Phase.DRIVE)
        self.assertLess(t.drive_distance.item(), 1e-5)

    def test_stop_milestone_survives_reversal_but_hold_requires_net_distance(self):
        t, s = task(), sample(); route_to_stop(t, s)
        s.forward[:] = .5
        for _ in range(1250): t.advance(s)
        self.assertEqual(t.phase.item(), Phase.STOP)
        self.assertFalse(t.completed.item())
        self.assertEqual(t.hold_ticks.item(), 0)

    def test_failed_legal_progress_beats_idle_without_rewarding_invalid_frame(self):
        def returned(kind):
            t,s=task(),sample(); rewards=[]
            if kind == 'idle':
                t.ticks[:] = t.deadline_ticks-1
                return t.advance(s).item()
            s.forward[:] = .12; s.speed[:] = .1; rewards.append(t.advance(s))
            if kind in ('landed','repeated'):
                fly_front(s); rewards.append(t.advance(s))
                put_on_top(s,.5); rewards.append(t.advance(s))
            best=t.best_legal_potential.item()
            if kind == 'repeated':
                fly_front(s);s.forward[:] = 1.
            else: s.body_invalid[:] = True
            rewards.append(t.advance(s))
            self.assertEqual(t.best_legal_potential.item(),best)
            return torch.stack(rewards).sum().item()
        idle, forward, landed, repeated = [returned(k) for k in ('idle','forward','landed','repeated')]
        self.assertEqual(idle,-10.)
        self.assertGreater(forward,idle)
        self.assertGreater(landed,forward)
        self.assertLessEqual(repeated,landed+1e-5)
        self.assertLess(landed,0.)

    def test_unqualified_stop_cannot_refresh_best_credit(self):
        t,s=task(),sample(); route_to_stop(t,s)
        best=t.best_legal_potential.item()
        s.forward[:]=.5;s.hip[:]=.06;s.speed.zero_()
        for _ in range(100):t.advance(s)
        self.assertFalse(t.ready.item())
        self.assertEqual(t.hold_ticks.item(),0)
        self.assertEqual(t.best_legal_potential.item(),best)

    def test_critic_distinguishes_task_history(self):
        t = task(2)
        t.drive_distance[:] = torch.tensor([.15, .07])
        t.phase[:] = torch.tensor([Phase.STOP, Phase.DRIVE])
        state = t.critic_memory()
        self.assertFalse(torch.equal(state[0], state[1]))
        t.reset(torch.tensor([0, 1]))
        self.assertTrue(torch.equal(t.critic_memory()[0], t.critic_memory()[1]))


if __name__ == '__main__':
    unittest.main()
