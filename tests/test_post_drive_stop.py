"""Counterexamples: stop shaping must not reward stagnation or bypass real stopping."""
import json,tempfile,unittest
from pathlib import Path
import torch
from step_demo.rewards import post_drive_stop_cost
from step_demo.runtime import load_training

class PostDriveStopTests(unittest.TestCase):
    def costs(self,w,v,active,dt=.02):
        return post_drive_stop_cost(torch.tensor(w),torch.tensor(v),torch.tensor(active),.02,dt)

    def test_before_drive_even_large_commands_have_no_cost_or_bonus(self):
        a,b=self.costs([0.,12.,25.],[0.,.3,1.],[False]*3)
        torch.testing.assert_close(a+b,torch.zeros(3))

    def test_command_and_motion_are_distinct_and_both_matter(self):
        a,b=self.costs([0.,12.,0.],[.0,.0,.32],[True]*3)
        self.assertEqual(float((a+b)[0]),0.)
        self.assertGreater(float(a[1]),0.);self.assertEqual(float(b[1]),0.)
        self.assertEqual(float(a[2]),0.);self.assertGreater(float(b[2]),0.)

    def test_quiet_tolerance_and_no_free_reward_for_waiting(self):
        a,b=self.costs([0.,0.,0.],[.0,.01,.02],[True]*3)
        torch.testing.assert_close(a+b,torch.zeros(3))

    def test_monotone_in_both_controls_without_randomness(self):
        state=torch.get_rng_state().clone()
        a,b=self.costs([0.,6.,12.,25.],[.0,.17,.32,.62],[True]*4)
        self.assertTrue(bool((a[1:]>a[:-1]).all()) and bool((b[1:]>b[:-1]).all()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        x,y=self.costs([12.],[.32],[True],.01)
        torch.testing.assert_close(2*(x+y),(a+b)[2:3])

    def test_round21_only_enables_post_drive_reward(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        for suffix in ['seed70','seed71','gate','scale_gate']:
            a=load_training(root/f'round20_{suffix}.json');b=load_training(root/f'round21_{suffix}.json')
            self.assertEqual(b.pop('post_drive_stop_cost'),'command_motion_v21');self.assertEqual(a,b)

    def test_unknown_or_incompatible_cost_rejected(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        b=load_training(root/'round21_gate.json')
        for updates in [{'post_drive_stop_cost':'unknown'},{'reward_version':'clean_v17'},{'task_memory':'none'},{'wheel_mapping':'affine'}]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'bad.json';p.write_text(json.dumps(dict(b,**updates)))
                with self.assertRaises(ValueError):load_training(p)

if __name__=='__main__':unittest.main()
