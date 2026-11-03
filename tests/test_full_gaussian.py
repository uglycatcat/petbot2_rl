"""Full-task exploration and continued acceptance, not synthetic capability claims."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
from step_demo.config import PROJECT_ROOT, load_config
from step_demo.distributions import FullStepGaussianDistribution
from step_demo.evaluation import CompletionTracker, EvaluationSample, Stage
from step_demo.ppo import distribution_metrics
from step_demo.runtime import load_training
from step_demo.regulation import discounted_failure_cost


class FullGaussianTests(unittest.TestCase):
    def test_wide_fresh_exploration_has_drive_lift_and_exact_zero_requests(self):
        torch.manual_seed(19)
        d=FullStepGaussianDistribution(2)
        d.update(torch.tensor([0.,-1.]).expand(60000,-1))
        raw=d.sample();request=raw.clamp(-1,1)
        self.assertGreater(float((request[:,1]==-1.).float().mean()),.48)
        self.assertGreater(float((request[:,1]>-.6).float().mean()),.1)
        self.assertGreater(float((request[:,0].abs()>.5).float().mean()),.1)
        torch.testing.assert_close(d.log_prob(raw),torch.distributions.Normal(*d.params).log_prob(raw).sum(-1))
        self.assertTrue(torch.isfinite(d.log_prob(raw)).all())
        self.assertGreater(float((d.log_prob(raw)-d.log_prob(request)).abs().max()),1.)

    def test_bounded_learned_noise_and_full_mean_authority(self):
        d=FullStepGaussianDistribution(2)
        for x,expected in [(-100.,.005),(100.,.5)]:
            with torch.no_grad():d.log_std_param.fill_(x)
            raw=torch.tensor([[-2.,2.],[2.,-2.]])
            d.update(raw);distribution_metrics(d)
            self.assertAlmostEqual(float(d.std.max()),expected,places=6)
            torch.testing.assert_close(d.deterministic_output(raw),raw)

    def test_full_config_preserves_actuators_and_sensor_only_model(self):
        t=load_training(PROJECT_ROOT/'configs/restart19_full_seed70.json')
        old=load_training(PROJECT_ROOT/'configs/round14_seed70.json')
        for k in ['decimation','hip_kp','hip_kd','head_kp','head_kd','ear_kp','ear_kd',
                  'hip_target_rate_rad_s','wheel_target_accel_rad_s2','hidden_dims','num_envs']:
            self.assertEqual(t[k],old[k])
        self.assertNotIn('critic_observation',t)
        self.assertEqual(t['initialization'],'random');self.assertEqual(t['pretrained_models'],[])
        self.assertEqual(t['iterations'],1000)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            for patch in [{'task_variant':'platform_stop_aux'},{'discount_gamma':1.},
                          {'policy_distribution':'beta'},{'initial_std':1.},{'entropy_coef':.01}]:
                p.write_text(json.dumps(dict(t,**patch)))
                with self.assertRaises(ValueError):load_training(p)

    def test_constant_failure_cost_exceeds_all_discounted_future_running_costs(self):
        cost=discounted_failure_cost(6.,.02,.99)
        self.assertAlmostEqual(cost,15.)
        for delay in (1,25,125,750):
            later=-.12*(1-.99**delay)/.01-.99**delay*cost
            self.assertGreaterEqual(later+1e-9,-cost)

    def test_extra_hold_does_not_latch_original_half_second_success(self):
        cfg=load_config().acceptance
        tracker=CompletionTracker(cfg,.06,2.,True)
        sample=EvaluationSample(.02,0.,0.,0.,0.,0.,(0.,0.),(True,True),True,False,False,False,False,True)
        tracker.update(sample)
        sample=replace(sample,forward_m=.16)
        for _ in range(26):tracker.update(sample)
        self.assertEqual(tracker.stage,Stage.STOPPING)
        tracker.update(replace(sample,speed_m_s=.021))
        self.assertEqual(tracker.stopped_s,0.)
        for _ in range(125):tracker.update(sample)
        self.assertEqual(tracker.stage,Stage.STOPPING)
        self.assertAlmostEqual(tracker.stopped_s,2.4999983310699463)
        self.assertEqual(tracker.update(sample),Stage.COMPLETE)

    def test_extra_hold_still_requires_four_wheels_legal_drive_and_single_flight(self):
        cfg=load_config().acceptance
        sample=EvaluationSample(.02,0.,0.,0.,0.,0.,(0.,0.),(True,True),True,False,False,False,False,True)
        for edit in [dict(four_wheels_on_top=False),dict(hip_error_rad=(.051,0.)),
                     dict(lateral_m=.021),dict(heading_rad=.09),dict(quality_failure=True)]:
            tracker=CompletionTracker(cfg,.06,2.,True);tracker.update(sample)
            broken=replace(sample,forward_m=.16,**edit)
            for _ in range(130):tracker.update(broken)
            self.assertNotEqual(tracker.stage,Stage.COMPLETE)
        tracker=CompletionTracker(cfg,.06,2.,True)
        for _ in range(130):tracker.update(sample)
        self.assertEqual(tracker.stage,Stage.PLATFORM_DRIVE)


if __name__=='__main__':unittest.main()
