"""Regression from real float32 holding:125 steps are2.499998331,not2.5."""
import json
from pathlib import Path
import tempfile
import unittest
from step_demo.stop_review import float32,summarize_auxiliary


class StopReviewTimerTests(unittest.TestCase):
    def test_observed_125_step_boundary_does_not_report_early_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';evaluation=Path(tmp)/'evaluation';run.mkdir();evaluation.mkdir()
            def write(p,x):p.write_text(json.dumps(x))
            acceptance=dict(line_tolerance_m=.02,heading_tolerance_rad=.1,hip_tolerance_rad=.05,
                stop_speed_m_s=.02,stop_yaw_rate_rad_s=.05,stop_hold_s=.5,timeout_s=15.)
            write(run/'origin.json',dict(initialization='random',pretrained_models=[],training=dict(iterations=8,policy_distribution='stop_gaussian'),task=dict(acceptance=acceptance)))
            write(run/'initialization.json',dict(optimizer_states=0,frozen_parameters=0))
            write(run/'status.json',dict(status='complete'));write(run/'latest.json',dict(completed_updates=8,actor_sha256='fixture'))
            write(evaluation/'layout.json',dict(step_dt=.02,trace_ids=[0],hips=[0,1],rear_wheels=[2,3]))
            write(evaluation/'restored_fixture.json',dict(actor_sha256='fixture'))
            write(evaluation/'result.json',dict(status='complete',completed_updates=8,hold_after_success_s=2.,num_envs=1,episodes=[dict(env=0,success=True)]))
            hold=0.;rows=[]
            for step in range(1,127):
                hold=float32(hold+float32(.02))
                if step==125:self.assertEqual(hold,2.4999983310699463)
                rows.append(dict(episode=[1],joint_position=[[0.]*4],joint_velocity=[[0.]*4],velocity=[[0.]*3],intent=[3],
                    hip_target=[0.],wheel_target=[0.],com=[[0.]*3],estimated_com=[[0.]*3],action=[[0.,-1.02]],requested_action=[[0.,-1.]],
                    world_pitch=[0.],world_pitch_rate=[0.],supported_top=[[True]*4],failure=[0],front_flight_count=[[0,0]],
                    yaw=[0.],yaw_rate=[0.],auxiliary_hold=[hold],completed=[step==126],elapsed=[step*.02]))
            (evaluation/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            result=summarize_auxiliary(run,evaluation)['evaluation']
            self.assertEqual(result['successes'],1);self.assertTrue(result['scalar_hold_agrees'])
            self.assertEqual(result['examples'][0]['wheel_request'],0.)
            self.assertEqual(result['raw_action_outside_fraction'],.5)
            self.assertFalse(result['auxiliary_gate_passed'])  # Single fixture is no64-case acceptance.


if __name__=='__main__':unittest.main()
