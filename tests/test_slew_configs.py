"""Round12 task/budget preservation and rollout-wide control configuration."""
import unittest
from step_demo.config import PROJECT_ROOT
from step_demo.runtime import load_training
from step_demo.ppo import runner_config

class SlewConfigTests(unittest.TestCase):
    def test_only_hip_target_slew_changes_from_round10(self):
        for name in ['seed70','seed71','gate','scale_gate']:
            a=load_training(PROJECT_ROOT/f'configs/round10_{name}.json')
            b=load_training(PROJECT_ROOT/f'configs/round12_{name}.json')
            self.assertEqual(a.pop('hip_target_rate_rad_s'),20.)
            self.assertEqual(b.pop('hip_target_rate_rad_s'),6.)
            self.assertEqual(a,b)

    def test_no_privileged_actor_or_critic_input_or_transfer(self):
        for seed in [70,71]:
            t=load_training(PROJECT_ROOT/f'configs/round12_seed{seed}.json')
            self.assertEqual(runner_config(t)['obs_groups'],{'actor':['policy'],'critic':['policy']})
            self.assertEqual(t['initialization'],'random');self.assertEqual(t['pretrained_models'],[])

if __name__=='__main__':unittest.main()
