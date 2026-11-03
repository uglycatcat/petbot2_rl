"""Quality progression must preserve control and survive same-run restoration."""
from dataclasses import replace
import unittest
import torch
from checks.test_contracts import state, truth
from upright.config import load_config
from upright.curriculum import Curriculum
from upright.runtime import Runtime
from upright.model import load_model
from upright.controller import Phase


class LandingCurriculum(unittest.TestCase):
    def runtime(self, config='configs/round21r2_quality.json'):
        cfg=load_config(config, num_envs=32)
        return Runtime(6,'cpu',.03,cfg,load_model())

    def test_pair_differs_only_in_quality_cost_and_schedule(self):
        a=load_config('configs/round21r2_control.json').to_dict()
        b=load_config('configs/round21r2_quality.json').to_dict()
        self.assertEqual({k for k in a if a[k]!=b[k]},
                         {'landing_approach_motion_cost','landing_quality_start_course'})
        self.assertEqual(a['seed'],102)
        for cfg in (a,b):
            self.assertEqual(cfg['iterations'],1000)
            self.assertTrue(cfg['normal_handoff_live_drive'])

    def test_gate_and_full_strength_for_regression_episodes(self):
        p=self.runtime();p.level[:]=1  # static regression continues to receive quality feedback
        for level,scale in [(0,0),(1,0),(2,0),(3,1/6),(5,.5),(8,1.)]:
            p.course.level=level
            self.assertEqual(p.landing_quality_scale(),scale)
        p.cfg=replace(p.cfg,landing_quality_start_course=0)
        p.course.level=0
        self.assertEqual(p.landing_quality_scale(),1.)

    def test_restore_saved_course_reproduces_reward_and_does_not_change_observation(self):
        p=self.runtime();s,t=state(6),truth(6);s['motion'][:,0]=-.3
        p.control.phase[:]=Phase.LAND;p.course.level=5
        obs=p.observation(s).clone();rng=p.generator.get_state().clone()
        saved=p.course.state_dict();a=p.landing_approach_cost(s,t,torch.ones(6,dtype=torch.bool),torch.ones(6))
        p.course=Curriculum(p.cfg);p.course.load_state_dict(saved)
        b=p.landing_approach_cost(s,t,torch.ones(6,dtype=torch.bool),torch.ones(6))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(obs,p.observation(s),rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,p.generator.get_state()))
        p.course.level=8
        full=p.landing_approach_cost(s,t,torch.ones(6,dtype=torch.bool),torch.ones(6))
        torch.testing.assert_close(a,full*.5,rtol=0,atol=0)
        p.course.level=2
        self.assertFalse(p.landing_approach_cost(s,t,torch.ones(6,dtype=torch.bool),torch.ones(6)).any())

    def test_reject_invalid_schedule(self):
        for level in [-1,9,True,2.5]:
            with self.assertRaises(ValueError):load_config('configs/round21r2_quality.json',landing_quality_start_course=level)


if __name__=='__main__':unittest.main()
