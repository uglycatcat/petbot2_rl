"""Training coverage without changing deployment or independent protocols."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config,ROOT
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL,state
CFG=load_config(ROOT/'configs/round25.json')

class CapabilitySampling25(unittest.TestCase):
    def make(self,n=4096,cfg=CFG):
        p=Runtime(n,'cpu',.06,cfg,MODEL);p.course.level=8;p.reset(torch.arange(n,dtype=torch.int32));return p

    def test_legacy_switches_default_off(self):
        old=load_config(ROOT/'configs/round24.json')
        self.assertEqual(old.takeover_boundary_fraction,0);self.assertEqual(old.steady_nonreversal_fraction,0)
        self.assertEqual(self.make(4,old).critic_state().shape,(4,24))

    def test_invalid_probabilities(self):
        for k in ['takeover_boundary_fraction','steady_nonreversal_fraction']:
            for x in [-.01,1.01,float('nan')]:
                with self.assertRaises(ValueError):replace(CFG,**{k:x}).validate()

    def test_expanded_edges_both_signs_with_interior_retained(self):
        p=self.make(cfg=replace(CFG,baseline_fraction=0))
        v=p.control.command[:,0];h=p.initial_hip
        for x,edge,interior in [(v,.6,.3),(h,.4,.2)]:
            self.assertTrue((x.abs()<=edge+1e-6).all())
            for sign in [-1,1]:self.assertGreater(int((abs(x-sign*edge)<1e-5).sum()),120)
            self.assertGreater(float((x.abs()<interior-1e-5).float().mean()),.70)

    def test_steady_keeps_reversal_short_and_mixed_coverage(self):
        p=self.make(cfg=replace(CFG,baseline_fraction=0));a=p.steady_episode;r=p.reversal_episode
        self.assertFalse((a&(r|p.short_cycle)).any())
        self.assertTrue(.08<float(a.float().mean())<.17)
        self.assertTrue(.20<float(r.float().mean())<.30)
        self.assertGreater(float((~a&~r&~p.short_cycle).float().mean()),.08)
        self.assertEqual(set(p.steady_speed[a].sign().tolist()),{-1.,1.})
        self.assertTrue((p.steady_speed[a].abs()>=.375).all())
        self.assertGreater(int((p.steady_speed[a].abs()<.749).sum()),100)

    def test_straight_persists_but_closing_and_ordinary_ownership_remain(self):
        p=self.make();ids=p.steady_episode.nonzero().flatten();self.assertGreater(len(ids),0)
        for _ in range(6):
            command=p.training_command(ids)
            torch.testing.assert_close(command[:,0],p.steady_speed[ids]);self.assertFalse(command[:,1].any())
        i=ids[0:1];p.invoked[:]=True;p.control.phase[i]=Phase.BALANCE;p.level[i]=8
        p.landing_case[i]=0;p.balance_elapsed[i]=29;p.balance_duration[i]=30
        p.before_action(state(p.n));self.assertFalse(p.control.command[i].any())
        p.control.phase[i]=Phase.HANDOFF;p.before_action(state(p.n));self.assertFalse(p.control.command[i].any())

    def test_new_sampler_hidden_from_actor_and_present_in_critic(self):
        p=self.make(4);obs=p.observation(state(4));p.steady_episode[:]=True;p.steady_speed[:]=.75
        torch.testing.assert_close(obs,p.observation(state(4)))
        c=p.critic_state();self.assertEqual(c.shape,(4,26));self.assertTrue((c[:,-2:]==1).all())
        self.assertTrue(torch.isfinite(c).all() and (c.abs()<=1).all())

    def test_fixed_probe_unchanged_and_never_enables_steady_sampling(self):
        old=load_config(ROOT/'configs/round24.json');a=Runtime(32,'cpu',.06,old,MODEL);b=Runtime(32,'cpu',.06,CFG,MODEL)
        for p in [a,b]:p.probe_level=8;p.reset(torch.arange(32,dtype=torch.int32))
        torch.testing.assert_close(a.initial_hip,b.initial_hip);torch.testing.assert_close(a.control.command,b.control.command)
        self.assertFalse(b.steady_episode.any());self.assertFalse(b.steady_speed.any())
        torch.testing.assert_close(a.observation(state(32)),b.observation(state(32)))

    def test_independent_held_velocity_has_both_signs_and_interior(self):
        p=Runtime(8,'cpu',.06,CFG,MODEL);p.probe_level=8;p.probe_moving=False;p.probe_steady=True
        p.reset(torch.arange(8,dtype=torch.int32));p.invoked[:]=True;p.control.phase[:]=Phase.BALANCE
        for age in [4.,8.,12.]:
            p.balance_elapsed[:]=age;p.command_age[:]=4.;p.before_action(state(8))
            torch.testing.assert_close(p.control.command[:,0],torch.tensor([.75,-.75,.375,-.375]*2))
            self.assertFalse(p.control.command[:,1].any())
        self.assertFalse((p.landing_case>=0).any())

if __name__=='__main__':unittest.main()
