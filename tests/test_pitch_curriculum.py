"""Learning schedule must survive resume and never relax final evaluation."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from step_demo.body_pitch import pitch_cost_scale_at
from step_demo.config import PROJECT_ROOT
from step_demo.runtime import load_training
from step_demo.ppo import StepRunner


class PitchCurriculumTests(unittest.TestCase):
    schedule = dict(start_update=100, end_update=300, initial_scale=.1)

    def test_next_rollout_scale_uses_completed_update_count(self):
        for update, expected in [(0,.1),(100,.1),(200,.55),(300,1.),(500,1.)]:
            self.assertAlmostEqual(pitch_cost_scale_at(self.schedule, update), expected)

    def test_scale_stays_positive_monotone_and_bounded(self):
        values=[pitch_cost_scale_at(self.schedule, i) for i in range(501)]
        self.assertTrue(all(0<x<=1 for x in values))
        self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))
        self.assertTrue(all(x==1 for x in values[300:]))

    def runner(self, schedule):
        runner=StepRunner.__new__(StepRunner)
        runner.origin={'training':{'body_pitch_curriculum':schedule} if schedule else {}}
        runner.env=SimpleNamespace(unwrapped=SimpleNamespace(body_pitch_cost_scale=1.))
        return runner

    def test_evaluation_forces_full_cost_even_from_mid_curriculum(self):
        runner=self.runner(self.schedule)
        for update in [0,100,200,300,500]:
            runner._set_pitch_cost_scale(update,training=True)
            self.assertAlmostEqual(runner.env.unwrapped.body_pitch_cost_scale,pitch_cost_scale_at(self.schedule,update))
            runner._set_pitch_cost_scale(update,training=False)
            self.assertEqual(runner.env.unwrapped.body_pitch_cost_scale,1.)

    def test_legacy_config_retains_full_cost(self):
        runner=self.runner(None)
        for update in [0,200,500]:
            runner._set_pitch_cost_scale(update,training=True)
            self.assertEqual(runner.env.unwrapped.body_pitch_cost_scale,1.)

    def test_configs_only_add_learning_schedule(self):
        for name in ['seed70','seed71','gate','scale_gate']:
            old=load_training(PROJECT_ROOT/f'configs/round18_{name}.json')
            new=load_training(PROJECT_ROOT/f'configs/round19_{name}.json')
            schedule=new.pop('body_pitch_curriculum')
            self.assertEqual(new,old)
            self.assertLess(schedule['end_update'],new['iterations'])

    def test_invalid_schedule_cannot_silently_disable_objective(self):
        base=json.loads((PROJECT_ROOT/'configs/round19_seed70.json').read_text())
        bad=[dict(self.schedule,initial_scale=x) for x in [0.,-1.,1.1,float('nan'),True]]
        bad += [dict(self.schedule,start_update=300),dict(self.schedule,end_update=500),dict(self.schedule,end_update=300.5),{'initial_scale':.1}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train.json'
            for schedule in bad:
                path.write_text(json.dumps(dict(base,body_pitch_curriculum=schedule)))
                with self.assertRaisesRegex(ValueError,'body pitch curriculum'):load_training(path)
            path.write_text(json.dumps(dict(base,reward_version='clean_v17')))
            with self.assertRaises(ValueError):load_training(path)

    def test_missing_or_inconsistent_checkpoint_scale_rejected_before_load(self):
        runner=self.runner(self.schedule);runner.device='cpu'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            base=dict(origin=runner.origin,iter=199,completed_updates=200)
            for data in [base,dict(base,body_pitch_cost_scale=.1)]:
                torch.save(data,path)
                with self.assertRaisesRegex(ValueError,'body pitch curriculum'):runner.restore(path,training=True)


if __name__=='__main__':unittest.main()
