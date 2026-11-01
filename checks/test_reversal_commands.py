"""Training-only reversal exposure, retained mixtures, reproducibility and isolation."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL, state

CFG = load_config()

class ReversalCommands(unittest.TestCase):
    def test_long_only_mixture_has_symmetric_continuous_and_boundary_amplitudes(self):
        p=Runtime(4096,'cpu',.06,CFG,MODEL);p.course.level=CFG.max_course_level;p.reset(torch.arange(4096))
        eligible=(p.level>=3)&~p.short_cycle
        paired=p.reversal_episode
        self.assertTrue((~paired|eligible).all())
        self.assertTrue(.44<float(paired.sum()/eligible.sum())<.56)
        speed=p.reversal_speed[paired];limit=p.stage_ranges[p.level[paired],0]
        self.assertTrue(((speed.abs()>=.5*limit)&(speed.abs()<=limit)).all())
        self.assertTrue(.2<float((speed.abs()==limit).float().mean())<.3)
        self.assertTrue(.44<float((speed>0).float().mean())<.56)
        self.assertGreater(len(speed.unique()),200)
        self.assertTrue(((p.level>=3)&~paired&~p.short_cycle).any())

    def test_consecutive_commands_reverse_only_selected_ids_without_future_observation(self):
        p=Runtime(64,'cpu',.06,CFG,MODEL);p.course.level=CFG.max_course_level;p.reset(torch.arange(64))
        ids=p.reversal_episode.nonzero().flatten();self.assertGreater(len(ids),0)
        a=p.training_command(ids);b=p.training_command(ids);c=p.training_command(ids)
        torch.testing.assert_close(a,-b);torch.testing.assert_close(a,c)
        self.assertFalse(a[:,1].any())
        before=p.control.command.clone();p.control.set_command(a,ids=ids)
        mask=~p.reversal_episode;torch.testing.assert_close(before[mask],p.control.command[mask])
        s=state(64);obs=p.observation(s);p.reversal_speed*= -1;p.reversal_episode[:]=False
        torch.testing.assert_close(obs,p.observation(s))

    def test_rng_replay_and_subset_reset_preserve_schedule(self):
        p=Runtime(64,'cpu',.06,CFG,MODEL);p.course.level=CFG.max_course_level;ids=torch.arange(64)
        rng=p.generator.get_state().clone();p.reset(ids)
        flag,speed,duration=p.reversal_episode.clone(),p.reversal_speed.clone(),p.balance_duration.clone()
        p.generator.set_state(rng);p.reset(ids)
        torch.testing.assert_close(flag,p.reversal_episode);torch.testing.assert_close(speed,p.reversal_speed)
        torch.testing.assert_close(duration,p.balance_duration)
        p.reset(torch.arange(32));torch.testing.assert_close(flag[32:],p.reversal_episode[32:]);torch.testing.assert_close(speed[32:],p.reversal_speed[32:])

    def test_low_course_and_deterministic_probe_are_unchanged(self):
        for level,probe in [(2,None),(8,8)]:
            runtimes=[Runtime(32,'cpu',.06,replace(CFG,reversal_episode_fraction=f),MODEL) for f in (0.,.5)]
            for p in runtimes:
                p.course.level=level;p.probe_level=probe;p.probe_duration=60. if probe else None;p.reset(torch.arange(32))
                self.assertFalse(p.reversal_episode.any());p.control.phase[:]=Phase.BALANCE
                for age in (4.1,8.2,12.3):
                    p.balance_elapsed[:]=age;p.command_age[:]=4.1;p.before_action(state(32))
            for name in ['generator','initial_hip','balance_duration']:
                x,y=(getattr(p,name) for p in runtimes)
                torch.testing.assert_close(x.get_state() if name=='generator' else x,y.get_state() if name=='generator' else y)
            torch.testing.assert_close(runtimes[0].control.command,runtimes[1].control.command)
            torch.testing.assert_close(runtimes[0].control.motion_reference,runtimes[1].control.motion_reference)

    def test_invalid_fraction_rejected(self):
        for v in (-.1,1.,float('inf'),float('nan'),True,'0.5'):
            with self.assertRaises(ValueError):replace(CFG,reversal_episode_fraction=v).validate()

if __name__=='__main__':unittest.main()
