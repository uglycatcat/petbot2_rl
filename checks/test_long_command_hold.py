"""Training plateau exposure without changing short cycles or validation."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config, ROOT
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL, state

CFG = load_config(ROOT/'configs/round21.json')

class LongCommandHold(unittest.TestCase):
    def test_only_higher_long_training_episodes_extend(self):
        p=Runtime(5,'cpu',.06,CFG,MODEL)
        p.level[:]=torch.tensor([1,2,3,8,8]);p.short_cycle[-1]=True
        torch.testing.assert_close(p.command_dwell(),torch.tensor([2.,2.,4.,6.,4.]))
        p.probe_level=8
        torch.testing.assert_close(p.command_dwell(),p.stage_dwell[p.level])

    def test_full_reversal_has_three_seconds_after_reference_arrival(self):
        holds=[]
        for extra in (0.,2.):
            p=Runtime(1,'cpu',.06,replace(CFG,long_command_hold_extra_s=extra),MODEL)
            p.level[:]=8;p.control.phase[:]=Phase.BALANCE;p.balance_duration[:]=30.
            p.reversal_episode[:]=True;p.reversal_speed[:]=-.75
            p.control.command[:,0]=.75;p.control.motion_reference[:,0]=.75
            p.command_age[:]=p.command_dwell();p.before_action(state(1))
            self.assertAlmostEqual(float(p.control.command[0,0]),-.75)
            settled=0
            for _ in range(150):
                if abs(float(p.control.motion_command()[0,0])+.75)<1e-5:settled+=1
                p.before_action(state(1))
                if p.control.command[0,0]>0:break
            holds.append(settled*.06)
        self.assertLess(holds[0],1.2);self.assertGreaterEqual(holds[1],2.94)

    def test_probe_commands_rng_control_and_observations_remain_identical(self):
        ps=[Runtime(32,'cpu',.06,replace(CFG,long_command_hold_extra_s=e),MODEL) for e in (0.,2.)]
        for p in ps:
            p.probe_level=8;p.probe_duration=60.;p.reset(torch.arange(32));p.control.phase[:]=Phase.BALANCE
        for step in range(210):
            for p in ps:p.balance_elapsed+=.06;p.before_action(state(32))
            for name in ('command','motion_reference'):
                torch.testing.assert_close(getattr(ps[0].control,name),getattr(ps[1].control,name),rtol=0,atol=0)
        torch.testing.assert_close(ps[0].generator.get_state(),ps[1].generator.get_state())
        torch.testing.assert_close(ps[0].observation(state(32)),ps[1].observation(state(32)),rtol=0,atol=0)
        torch.testing.assert_close(ps[0].critic_state(),ps[1].critic_state(),rtol=0,atol=0)

    def test_actual_hold_controls_reversal_eligibility(self):
        p=Runtime(32,'cpu',.06,CFG,MODEL);p.level[:]=8;p.balance_duration[:]=10.
        p.sample_reversal_episodes(torch.arange(32));self.assertFalse(p.reversal_episode.any())
        p.balance_duration[:]=30.;p.sample_reversal_episodes(torch.arange(32))
        self.assertTrue(p.reversal_episode.any())

    def test_actor_isolation_and_critic_uses_actual_command_age(self):
        a,b=[Runtime(2,'cpu',.06,replace(CFG,long_command_hold_extra_s=e),MODEL) for e in (0.,2.)]
        for p in (a,b):p.level[:]=8;p.command_age[:]=3.
        torch.testing.assert_close(a.observation(state()),b.observation(state()),rtol=0,atol=0)
        torch.testing.assert_close(a.critic_state()[:,4],torch.full((2,),.75))
        torch.testing.assert_close(b.critic_state()[:,4],torch.full((2,),.5))

    def test_invalid_values_and_legacy_default(self):
        self.assertEqual(load_config(ROOT/'configs/round18.json').long_command_hold_extra_s,0.)
        for value in (-1.,True,'2',float('nan'),float('inf')):
            with self.assertRaises(ValueError):replace(CFG,long_command_hold_extra_s=value).validate()

if __name__=='__main__':unittest.main()
