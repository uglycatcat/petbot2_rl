"""Direction-aware LAND shaping and independent ordinary input ownership."""
from dataclasses import replace
import unittest
import torch
from upright.config import load_config, ROOT
from upright.runtime import Runtime
from upright.controller import Phase
from test_contracts import MODEL, state, truth

CFG = load_config(ROOT/'configs/round24.json')


class LandingRound24(unittest.TestCase):
    def test_current_request_survives_ground_transfer_and_actor_cannot_drive_ordinary(self):
        p = Runtime(3, 'cpu', .06, CFG, MODEL);c=p.control;s=state(3)
        c.phase[:] = Phase.LAND;c.ground_run[:] = CFG.ground_confirm_s
        c.set_command([[.3, 0], [-.3, 0], [0, 0]])
        c.executed[:, 1:] = -12;s['motion'][:, 0] = -.5
        transferred=c.update(s)
        self.assertTrue(transferred['entered_handoff'].all())
        expected=torch.tensor([[.3,0],[-.3,0],[0,0]])
        torch.testing.assert_close(c.command,expected)
        first=c.targets(torch.ones(3,3),s).clone()
        torch.testing.assert_close(first[:,1:],torch.full((3,2),-12.))
        c.age[:] = .4
        a=c.targets(torch.ones(3,3),s).clone()
        b=c.targets(-torch.ones(3,3),s).clone()
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a[:,1:],torch.tensor([[10.,10.],[-10.,-10.],[0.,0.]]))
        c.set_command(torch.zeros(3,2));c.targets(torch.ones(3,3),s)
        self.assertFalse(c.executed[:,1:].any())

    def test_exit_keeps_live_input_and_target_continuity(self):
        p=Runtime(1,'cpu',.06,CFG,MODEL);c=p.control;s=state(1)
        c.phase[:]=Phase.BALANCE;c.executed[:,1:]=8;c.command[:]=0
        s['g'][:,1]=1;s['motion'][:,0]=.6
        self.assertTrue(c.update(s)['failure'].all())
        self.assertFalse(c.command.any())
        torch.testing.assert_close(c.targets(torch.zeros(1,3),s)[:,1:],torch.full((1,2),8.))

    def test_quality_allows_deceleration_is_mirrored_and_rejects_reverse_drift(self):
        p=Runtime(7,'cpu',.06,CFG,MODEL);s=state(7)
        p.control.transition_motion[:,0]=torch.tensor([.5,.5,-.5,-.5,0.,0.,.5])
        s['motion'][:,0]=torch.tensor([.5,.2,-.5,-.2,0.,-.5,-.5])
        q=p.landing_motion_quality(s)
        torch.testing.assert_close(q[:5],torch.ones(5))
        self.assertLess(float(q[5]),.05);self.assertLess(float(q[6]),.1)
        s['motion'][:,0]*=-1;p.control.transition_motion[:,0]*=-1
        torch.testing.assert_close(p.landing_motion_quality(s),q)

    def test_quality_cost_and_bonus_are_bounded_and_land_only(self):
        for phase in (Phase.RISE,Phase.BALANCE,Phase.LAND):
            p=Runtime(2,'cpu',.06,CFG,MODEL);p.control.phase[:]=phase
            s=state();s['motion'][:,0]=-.6
            m=p.advance(s,truth(raised=phase==Phase.BALANCE))
            self.assertTrue((m['landing_quality_cost']<=CFG.landing_quality_cost*.06).all())
            self.assertTrue((m['landing_quality_bonus']<=CFG.landing_quality_bonus).all())
            if phase!=Phase.LAND:self.assertFalse(m['landing_quality_cost'].any())

    def test_closing_sampler_retains_other_episodes_and_hides_future_from_actor(self):
        p=Runtime(2048,'cpu',.06,replace(CFG,baseline_fraction=0),MODEL)
        p.course.level=8;p.reset(torch.arange(p.n))
        selected=p.landing_case>=0
        self.assertTrue((~p.short_cycle[selected]).all())
        self.assertTrue(.18<float(selected.float().mean())<.32)
        self.assertEqual(set(p.landing_case[selected].tolist()),{0,1,2})
        obs=p.observation(state(p.n));p.landing_case=(p.landing_case+1)%3
        torch.testing.assert_close(obs,p.observation(state(p.n)))

    def test_fixed_three_way_probe_commands_before_and_after_transfer(self):
        p=Runtime(3,'cpu',.06,CFG,MODEL);p.probe_level=8;p.probe_landing=True;p.probe_duration=12
        # Isaac passes int32 indices; plain torch.arange defaults to int64.
        p.reset(torch.arange(3,dtype=torch.int32));p.invoked[:]=True;p.control.phase[:]=Phase.BALANCE
        p.balance_elapsed[:]=11;p.before_action(state(3))
        expected=torch.tensor([[0.,0.],[.75,0.],[-.75,0.]])
        torch.testing.assert_close(p.control.command,expected)
        p.control.phase[:]=Phase.HANDOFF;p.control.age[:]=.5
        p.before_action(state(3));p.control.targets(torch.ones(3,3),state(3))
        torch.testing.assert_close(p.control.executed[:,1:],torch.tensor([[0.,0.],[25.,25.],[-25.,-25.]]))

    def test_ordinary_observation_does_not_repeat_completion_reward_or_give_actor_control(self):
        p=Runtime(1,'cpu',.06,CFG,MODEL);p.level[:]=1;p.balance_duration[:]=2
        p.invoked[:]=True;p.balance_seen[:]=True;p.longest_hold[:]=2
        p.control.phase[:]=Phase.HANDOFF;p.control.handoff_run[:]=.6
        p.post_handoff_observe_s=2
        a=p.advance(state(1),truth(1));b=p.advance(state(1),truth(1))
        self.assertTrue(a['completion_event'].all());self.assertFalse(b['completion_event'].any())
        self.assertFalse(a['done'].any());self.assertFalse(b['done'].any())
        self.assertGreater(float(a['reward'][0]-b['reward'][0]),29.)
        p.post_handoff_elapsed[:]=2
        self.assertTrue(p.advance(state(1),truth(1))['done'].all())

    def test_ordinary_success_does_not_hide_landing_reverse_or_failure(self):
        from upright.landing_response import landing_response
        frames=[]
        for j in range(20):
            frames.append(dict(t=.06*(j+1),active=[True]*3,landing_case=[0,1,2],
                phase_before=[3 if j<5 else 0]*3,phase=[4 if j==4 else 0]*3,
                failure=[False,False,j==19],reason=[0,0,1 if j==19 else 0],
                complete=[j>=4]*3,effective_command=[[0,0],[.5,0],[-.5,0]],
                requested_command=[[0,0],[.5,0],[-.5,0]],
                motion=[[-.4 if j<5 else 0,0],[-.4 if j<5 else .49,0],[-.49,0]]))
        result=landing_response(frames,3,.06)
        self.assertEqual(result['groups']['zero']['ordinary_following_pass'],1)
        self.assertGreater(result['cases'][0]['unintended_travel_m'],.1)
        self.assertEqual(result['groups']['forward']['ordinary_following_pass'],1)
        self.assertGreater(result['cases'][1]['unintended_travel_m'],.1)
        self.assertEqual(result['groups']['reverse']['ordinary_following_pass'],0)
        self.assertTrue(result['motion_quality_pass_not_inferred_from_ordinary_following'])


if __name__=='__main__':unittest.main()
