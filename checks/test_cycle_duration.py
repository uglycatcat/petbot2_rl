"""Training Space timing, full evaluation horizon, physical completion and RNG contracts."""
from dataclasses import replace
import unittest
import torch
from upright.config import Config
from upright.runtime import Runtime
from upright.controller import Phase,Reason
from test_contracts import MODEL,state,truth

CFG=Config(num_envs=2048,balance_train_s=30.,short_cycle_fraction=.5,short_balance_min_s=4.,short_balance_max_s=8.)
class CycleDurationContracts(unittest.TestCase):
    def test_training_mixture_retains_long_motion_and_static_cycles(self):
        p=Runtime(2048,'cpu',.06,CFG,MODEL);p.course.level=2;p.reset(torch.arange(2048))
        moving=p.level>=2;short=moving&p.short_cycle;long=moving&~p.short_cycle
        self.assertTrue(short.any() and long.any() and (~moving).any())
        self.assertTrue(((p.balance_duration[short]>=4)&(p.balance_duration[short]<=8)).all())
        self.assertTrue((p.balance_duration[long]==30).all() and (p.balance_duration[~moving]==2).all())
        self.assertTrue(.45<float(short.sum()/moving.sum())<.55)
        self.assertGreater(len(p.balance_duration[short].unique()),100)

    def test_probe_duration_overrides_all_training_sampling(self):
        p=Runtime(32,'cpu',.06,CFG,MODEL);p.probe_level=2;p.probe_duration=60.;before=p.generator.get_state().clone()
        p.reset(torch.arange(32));self.assertTrue((p.balance_duration==60).all());self.assertFalse(p.short_cycle.any())
        torch.testing.assert_close(before,p.generator.get_state())
        p.probe_level=1;p.reset(torch.arange(32));self.assertTrue((p.balance_duration==2).all())
        p.probe_level=2;p.probe_duration=30.;p.reset(torch.arange(32));self.assertTrue((p.balance_duration==30).all())

    def test_sampled_timing_replays_from_scenario_rng_and_subset_reset_is_local(self):
        p=Runtime(32,'cpu',.06,CFG,MODEL);p.course.level=2;ids=torch.arange(32);before=p.generator.get_state().clone()
        with torch.inference_mode():p.reset(ids)
        duration=p.balance_duration.clone();commands=p.control.command.clone();hips=p.initial_hip.clone()
        p.generator.set_state(before);p.reset(ids)
        torch.testing.assert_close(duration,p.balance_duration);torch.testing.assert_close(commands,p.control.command);torch.testing.assert_close(hips,p.initial_hip)
        p.reset(torch.arange(16));torch.testing.assert_close(duration[16:],p.balance_duration[16:])

    def test_space_uses_sampled_time_without_changing_inherited_actual_speed(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);p.course.level=2;p.reset(torch.arange(2));p.level[:]=2;p.balance_duration[:]=torch.tensor([4.,30.]);p.control.phase[:]=Phase.BALANCE;p.invoked[:]=True
        p.balance_elapsed[:]=5;s=state();s['motion'][:,0]=.25;p.before_action(s)
        self.assertEqual(int(p.control.phase[0]),Phase.LAND);self.assertEqual(int(p.control.phase[1]),Phase.BALANCE)
        self.assertAlmostEqual(float(p.control.transition_motion[0,0]),.25)
        p.before_action(s);self.assertEqual(int(p.control.phase[0]),Phase.LAND)

    def test_short_cycle_completion_cannot_satisfy_long_or_independent_horizon(self):
        p=Runtime(3,'cpu',.06,CFG,MODEL);p.reset(torch.arange(3));p.level[:]=2;p.balance_duration[:]=torch.tensor([4.,30.,60.]);p.balance_seen[:]=True;p.longest_hold[:]=4.;p.invoked[:]=True
        p.control.phase[:]=Phase.HANDOFF;p.control.completed[:]=True
        r=p.advance(state(3),truth(3));self.assertEqual(r['complete'].tolist(),[True,False,False]);self.assertEqual(r['failure'].tolist(),[False,True,True])
        self.assertEqual(r['reason'].tolist(),[0,Reason.SUPPORT_MISMATCH,Reason.SUPPORT_MISMATCH])

    def test_schedule_is_not_actor_input_and_disabled_mode_keeps_old_horizon(self):
        p=Runtime(2,'cpu',.06,replace(CFG,short_cycle_fraction=0.),MODEL);p.course.level=2;p.reset(torch.arange(2));self.assertTrue((p.balance_duration[p.level==2]==30).all())
        p.control.phase[:]=Phase.BALANCE;s=state();a=p.observation(s);p.balance_duration[:]=torch.tensor([4.,60.]);p.short_cycle[:]=True
        torch.testing.assert_close(a,p.observation(s))

    def test_demo_budget_can_reach_full_speed_without_weakening_promotion(self):
        from upright.config import load_config
        from upright.curriculum import Curriculum
        cfg=load_config();course=Curriculum(cfg);first_full=None
        for update in range(1,cfg.iterations+1):
            if course.probe_due(update):
                previous=course.level
                course.observe_probe(update,cfg.probe_count,cfg.probe_count,previous)
                if course.level>previous:
                    self.assertEqual([r['level'] for r in course.probes[-2:]],[previous,previous])
                if course.level==cfg.max_course_level and first_full is None:first_full=update
        self.assertIsNotNone(first_full)
        self.assertLess(first_full,cfg.iterations)
        self.assertEqual(cfg.motion_stages()[-1][0],.75)
        self.assertEqual(cfg.curriculum_consecutive,2)
        self.assertEqual(cfg.curriculum_pass_rate,.8)

    def test_invalid_mixture_and_short_duration_rejected(self):
        for kwargs in [dict(short_cycle_fraction=1.),dict(short_cycle_fraction=-.1),dict(short_balance_min_s=1.),dict(short_balance_max_s=31.),dict(short_balance_min_s=9.,short_balance_max_s=8.)]:
            with self.assertRaises(ValueError):replace(CFG,**kwargs).validate()

if __name__=='__main__':unittest.main()
