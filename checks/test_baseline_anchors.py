"""Preserve high-range exposure and evaluation while keeping basic skills sampled."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.runtime import Runtime
from checks.test_contracts import MODEL, state

class BaselineAnchors(unittest.TestCase):
    def pair(self,n):
        c=load_config('configs/round21r4.json')
        return [Runtime(n,'cpu',.03,replace(c,baseline_anchor_fraction=a),MODEL) for a in (0.,.75)]

    def test_episode_proportions_and_all_old_stages(self):
        p=self.pair(4096)[1];p.course.level=8;counts=torch.zeros(9)
        for _ in range(8):
            p.reset(torch.arange(4096));counts+=torch.bincount(p.level,minlength=9)
            self.assertTrue((p.control.command.abs() <= p.stage_ranges[p.level,:2]+1e-6).all())
        f=counts/counts.sum()
        for level,want in [(1,.0571429),(2,.1071429),(8,.8)]:
            self.assertAlmostEqual(float(f[level]),want,delta=.012)
        self.assertTrue((counts[3:8]>100).all())

    def test_evaluation_states_commands_and_rng_unchanged(self):
        for level in range(9):
            pair=self.pair(32)
            for p in pair:p.probe_level=level;p.course.level=8;p.reset(torch.arange(32))
            for attr in ['level','initial_hip','balance_duration','short_cycle','reversal_episode']:
                torch.testing.assert_close(getattr(pair[0],attr),getattr(pair[1],attr),rtol=0,atol=0)
            torch.testing.assert_close(pair[0].generator.get_state(),pair[1].generator.get_state(),rtol=0,atol=0)
            s=state(32)
            torch.testing.assert_close(pair[0].observation(s),pair[1].observation(s),rtol=0,atol=0)
            for _ in range(5):
                for p in pair:
                    p.control.phase[:]=2;p.command_age[:]=5;p.balance_elapsed+=4.;p.before_action(s)
                torch.testing.assert_close(pair[0].control.command,pair[1].control.command,rtol=0,atol=0)

    def test_early_courses_and_same_run_sampling_restore(self):
        for level in range(3):
            a,b=self.pair(256)
            for p in (a,b):p.course.level=level;p.reset(torch.arange(256))
            torch.testing.assert_close(a.level,b.level,rtol=0,atol=0)
            torch.testing.assert_close(a.generator.get_state(),b.generator.get_state(),rtol=0,atol=0)
        a=self.pair(256)[1];a.course.level=8;a.reset(torch.arange(256));saved=a.generator.get_state().clone()
        b=self.pair(256)[1];b.course.load_state_dict(a.course.state_dict());b.generator.set_state(saved)
        for p in (a,b):p.reset(torch.arange(256))
        for attr in ['level','initial_hip','balance_duration','reversal_episode','reversal_speed']:
            torch.testing.assert_close(getattr(a,attr),getattr(b,attr),rtol=0,atol=0)

    def test_invalid_fraction(self):
        for value in (-.1,1.,True,float('nan'),'0.75'):
            with self.assertRaises(ValueError):load_config('configs/round21r4.json',baseline_anchor_fraction=value)

if __name__=='__main__':unittest.main()
