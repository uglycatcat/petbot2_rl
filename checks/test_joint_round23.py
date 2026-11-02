"""Wider takeover, stopping/reversal exposure and phase-local preparation objective."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config, ROOT
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL,state,truth
CFG=load_config(ROOT/'configs/round23.json')
OLD=load_config(ROOT/'configs/round22.json')

class JointRound23(unittest.TestCase):
    def test_expanded_takeover_retains_majority_old_range_and_both_signs(self):
        p=Runtime(8192,'cpu',.06,replace(CFG,baseline_fraction=0),MODEL)
        p.course.level=8;p.reset(torch.arange(p.n))
        v=p.control.command[:,0];h=p.initial_hip
        self.assertLessEqual(float(v.abs().max()),.600001);self.assertLessEqual(float(h.abs().max()),.400001)
        self.assertTrue(.12<float(((v.abs()>.3)|(h.abs()>.2)).float().mean())<.18)
        self.assertLess(abs(float(v.mean())),.02);self.assertLess(abs(float(h.mean())),.02)
        rng=p.generator.get_state();p.reset(torch.arange(100));v2=p.control.command.clone()
        p.generator.set_state(rng);p.reset(torch.arange(100));torch.testing.assert_close(v2,p.control.command)

    def test_stops_keep_pending_reverse_direction_and_actor_isolation(self):
        p=Runtime(4096,'cpu',.06,CFG,MODEL);p.level[:]=8;p.reversal_episode[:]=True;p.reversal_speed[:]=.75
        ids=torch.arange(p.n);last=torch.full((p.n,),-.75)
        stops=0
        for _ in range(8):
            old=p.reversal_speed.clone();cmd=p.training_command(ids);zero=cmd[:,0]==0
            self.assertTrue((cmd[:,1]==0).all());torch.testing.assert_close(p.reversal_speed[zero],old[zero])
            torch.testing.assert_close(cmd[~zero,0],-last[~zero]);last[~zero]=cmd[~zero,0];stops+=int(zero.sum())
        self.assertTrue(.23<stops/(8*p.n)<.27)
        obs=p.observation(state(p.n));p.reversal_speed*=-1;torch.testing.assert_close(obs,p.observation(state(p.n)))

    def test_low_course_and_native_validation_state_command_rng_unchanged(self):
        for level,probe in [(2,None),(8,8)]:
            ps=[Runtime(32,'cpu',.06,c,MODEL) for c in (OLD,CFG)]
            for p in ps:
                p.course.level=level;p.probe_level=probe;p.probe_duration=60 if probe else None;p.reset(torch.arange(32))
                p.control.phase[:]=Phase.BALANCE
                for _ in range(150):p.balance_elapsed+=.06;p.before_action(state(32))
            for name in ['initial_hip','balance_duration','reversal_speed']:
                torch.testing.assert_close(getattr(ps[0],name),getattr(ps[1],name),rtol=0,atol=0)
            torch.testing.assert_close(ps[0].generator.get_state(),ps[1].generator.get_state())
            torch.testing.assert_close(ps[0].control.command,ps[1].control.command)
            torch.testing.assert_close(ps[0].observation(state(32)),ps[1].observation(state(32)))

    def test_rise_allows_adjustment_penalizes_fast_pitch_without_changing_land_balance(self):
        for phase in [Phase.RISE,Phase.BALANCE,Phase.LAND]:
            rewards=[];obs=[];targets=[]
            for cfg in (OLD,CFG):
                p=Runtime(3,'cpu',.06,cfg,MODEL);p.control.phase[:]=phase;p.invoked[:]=True;p.takeover_motion[:,0]=.3
                s=state(3);s['gyro'][:,1]=torch.tensor([0.,6.,-6.]);s['motion'][:,0]=0.
                t=truth(3,raised=phase==Phase.BALANCE)
                rewards.append(p.advance(s,t)['reward']);obs.append(p.observation(s));targets.append(p.control.targets(torch.zeros(3,3),s).clone())
            torch.testing.assert_close(obs[0],obs[1]);torch.testing.assert_close(targets[0],targets[1])
            if phase==Phase.RISE:
                self.assertGreater(float(rewards[1][0]),float(rewards[0][0]))
                self.assertLess(float(rewards[1][1]),float(rewards[1][0]));self.assertEqual(float(rewards[1][1]),float(rewards[1][2]))
                self.assertLess(float(rewards[1][0]),0.) # grounded idle never pays
            else:torch.testing.assert_close(rewards[0],rewards[1])

    def test_invalid_mixture_and_nonfinite_reward_config_rejected(self):
        for key in ('takeover_expansion_fraction','reversal_stop_probability'):
            for v in (-.1,1.,float('nan'),True):
                with self.assertRaises(ValueError):replace(CFG,**{key:v}).validate()
        for key in ('rise_time_cost','rise_adjustment_cost','rise_pitch_rate_cost'):
            with self.assertRaises(ValueError):replace(CFG,**{key:float('nan')}).validate()

if __name__=='__main__':unittest.main()
