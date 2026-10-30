import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from step_demo.config import PROJECT_ROOT
from step_demo.evaluation import CompletionTracker, EvaluationSample, Stage
from tools.campaign import queue_event,safe_relative,validate_plan


class AutomationTests(unittest.TestCase):
    def plan_directory(self,directory):
        root=Path(directory);(root/'source/configs').mkdir(parents=True)
        (root/'source/configs/train.json').write_text((PROJECT_ROOT/'configs/train_v1.json').read_text())
        return root,dict(schema='step_batch_v1',kind='formal',round_number=1,hypothesis='baseline',
            jobs=[dict(id='seed70',gpu=0,training_config='configs/train.json',eval_envs=64)])

    def test_gpu_restriction_and_one_job_per_gpu(self):
        with tempfile.TemporaryDirectory(prefix='step-plan-') as directory:
            root,plan=self.plan_directory(directory);validate_plan(plan,root)
            plan['jobs'][0]['gpu']=2
            with self.assertRaises(ValueError):validate_plan(plan,root)
            plan['jobs'][0]['gpu']=0;plan['jobs'].append(dict(plan['jobs'][0],id='seed71'))
            with self.assertRaises(ValueError):validate_plan(plan,root)

    def test_next_round_requires_optimization_decision(self):
        with tempfile.TemporaryDirectory(prefix='step-plan-') as directory:
            root,plan=self.plan_directory(directory);plan['round_number']=2
            with self.assertRaises(ValueError):validate_plan(plan,root)
            plan.update(previous_review='previous/analysis.json',decision=dict(change='reward adjustment',reason='observed idle policy'))
            validate_plan(plan,root)

    def test_plan_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory(prefix='step-plan-') as directory:
            for value in ('../../elsewhere','/tmp/elsewhere'):
                with self.assertRaises(ValueError):safe_relative(directory,value)

    def control(self,directory,enabled=True):
        root=Path(directory);control=root/'control.json'
        control.write_text(json.dumps(dict(enabled=enabled,codex='codex',thread='this-session',cwd=directory)))
        return control,root/'event.json'

    def test_event_is_delivered_only_once_to_same_session(self):
        with tempfile.TemporaryDirectory(prefix='step-event-') as directory:
            control,event=self.control(directory)
            with patch('tools.campaign.subprocess.run',return_value=subprocess.CompletedProcess([],0,'queued','')) as call:
                self.assertEqual(queue_event(control,event,'done')['status'],'queued')
                queue_event(control,event,'done');self.assertEqual(call.call_count,1)
                self.assertEqual(call.call_args.args[0],['codex','queue','--thread','this-session','--message','done'])

    def test_unknown_delivery_is_not_blindly_retried(self):
        with tempfile.TemporaryDirectory(prefix='step-event-') as directory:
            control,event=self.control(directory)
            with patch('tools.campaign.subprocess.run',side_effect=subprocess.TimeoutExpired('codex',60)) as call:
                self.assertEqual(queue_event(control,event,'done')['status'],'delivery_unknown')
                queue_event(control,event,'done');self.assertEqual(call.call_count,1)

    def test_stop_disables_delivery(self):
        with tempfile.TemporaryDirectory(prefix='step-event-') as directory:
            control,event=self.control(directory,False)
            with patch('tools.campaign.subprocess.run') as call:
                self.assertEqual(queue_event(control,event,'done')['status'],'disabled')
                call.assert_not_called();self.assertFalse(event.exists())

    def test_timeout_accumulation_matches_750_policy_steps(self):
        from step_demo.config import load_config
        tracker=CompletionTracker(load_config().acceptance)
        sample=EvaluationSample(.02,0.,0.,0.,0.,0.,(0.,0.),(False,False),False,False,False,False,False)
        for _ in range(749):self.assertEqual(tracker.update(sample),Stage.CROSSING)
        self.assertEqual(tracker.update(sample),Stage.FAILED)

if __name__=='__main__':unittest.main()
