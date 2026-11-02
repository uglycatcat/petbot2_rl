"""Counterexamples for the reward-only home penalty after complete ascent."""
import json,unittest
from pathlib import Path
import torch
from step_demo.rewards import post_top_home_cost
from step_demo.runtime import load_training


class PostTopHomeTests(unittest.TestCase):
    def cost(self, values, seen=True, dt=.02):
        x=torch.tensor(values,dtype=torch.float32)
        return post_top_home_cost(x,torch.full((len(x),),seen),.05,dt)

    def test_lifting_before_four_wheel_ascent_is_not_charged(self):
        torch.testing.assert_close(self.cost([[-1.,1.2],[.8,.8]],False),torch.zeros(2))

    def test_default_tolerance_has_neither_cost_nor_holding_bonus(self):
        torch.testing.assert_close(self.cost([[0.,0.],[.05,-.05],[.03,-.01]]),torch.zeros(3))

    def test_departures_in_either_direction_are_charged_equally(self):
        a=self.cost([[.15,.25],[-.15,-.25],[.25,.15]])
        torch.testing.assert_close(a,torch.full((3,),.03))

    def test_cost_increases_with_angle_and_time_outside_tolerance(self):
        c=self.cost([[.06,.06],[.1,.1],[.5,.5],[1.,1.]])
        self.assertTrue(bool((c[1:]>c[:-1]).all()))
        self.assertAlmostEqual(float(c[-1]),.19,places=6)
        self.assertGreater(float(c[-1]*10),float(c[-1]))

    def test_physical_duration_not_policy_frame_count_sets_cost(self):
        a=self.cost([[.4,-.2]],dt=.02)*5
        b=self.cost([[.4,-.2]],dt=.01)*10
        torch.testing.assert_close(a,b)

    def test_reward_calculation_consumes_no_policy_randomness(self):
        state=torch.get_rng_state().clone();self.cost([[.4,-.3]])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_round15_changes_only_reward_from_round13_baseline(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        for suffix in ['seed70','seed71','gate','scale_gate']:
            a=load_training(root/f'round13_{suffix}.json');b=load_training(root/f'round15_{suffix}.json')
            self.assertEqual(b.pop('reward_version'),'home_v15');a.pop('reward_version');self.assertEqual(a,b)
            self.assertNotIn('wheel_final_std',b['beta_std_ceiling'])


if __name__=='__main__':unittest.main()
