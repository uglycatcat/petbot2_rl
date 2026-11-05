import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from step_demo.video_selection import select_best
from tools.campaign import record_best_video


class VideoSelectionTests(unittest.TestCase):
    def job(self, root, name, success=False, phase=0, reward=-9., status='complete'):
        p=root/name;(p/'train').mkdir(parents=True);(p/'evaluation').mkdir()
        (p/'train/model.pt').write_bytes(b'test-checkpoint')
        (p/'job_status.json').write_text(json.dumps(dict(status=status)))
        (p/'analysis.json').write_text(json.dumps(dict(evaluation=dict(
            independent_physics=dict(all_terminal=True),outcomes=dict(total=1,successes=int(success)),
            phase_frames={str(phase):4,'7':100}))))
        (p/'evaluation/result.json').write_text(json.dumps(dict(checkpoint='model.pt',completed_updates=1000,
            episodes=[dict(success=success,drive_distance=.15 if success else 0.,hold_s=2.5 if success else 0.,
                           return_sum=reward,quality_cost=1.)])))
        return dict(id=name,gpu=0)

    def test_success_wins_over_reward_and_failed_jobs_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=[self.job(root,'success',True,6,35.),self.job(root,'failed',False,5,999.),
                self.job(root,'invalid',True,6,999.,status='failed')]
            result=select_best(root,jobs)
            self.assertEqual(result['selected']['job'],'success')
            self.assertEqual(result['excluded'][0]['job'],'invalid')

    def test_no_success_uses_verified_legal_progress_not_failed_phase(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=[self.job(root,'approach',phase=0,reward=-4.),
                               self.job(root,'landed',phase=2,reward=-9.)]
            self.assertEqual(select_best(root,jobs)['selected']['job'],'landed')
            (root/'landed/train/model.pt').unlink()
            self.assertEqual(select_best(root,jobs)['selected']['job'],'approach')

    def test_video_is_not_recorded_twice_and_pause_blocks_new_recording(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);movie=root/'best.mp4';movie.write_bytes(b'published-video')
            index=dict(status='complete',file=str(movie),sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
            (root/'video_status.json').write_text(json.dumps(index))
            with patch('tools.campaign.subprocess.run') as run:
                self.assertEqual(record_best_video(root,root/'absent-control.json',{}),index)
                run.assert_not_called()
            (root/'video_status.json').unlink();control=root/'control.json'
            control.write_text(json.dumps(dict(enabled=False)))
            self.assertEqual(record_best_video(root,control,{})['status'],'cancelled_by_user')
