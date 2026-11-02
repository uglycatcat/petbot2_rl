"""A real acceptance distinction lost by the old offline-version allowlist."""
import unittest
from step_demo.config import load_config
from step_demo.evaluation import CompletionTracker, EvaluationSample, Stage
from step_demo.review import support_memory_seconds


class ReviewSupportMemoryTests(unittest.TestCase):
    def trajectory(self, memory):
        tracker = CompletionTracker(load_config().acceptance, memory)
        def tick(x, supported):
            return tracker.update(EvaluationSample(.02, x, 0., 0., 0., 0., (0., 0.),
                (True, True), supported, False, False, False, False, True))
        tick(0., True);tick(.1, True);tick(.12, False);tick(.13, False)
        for _ in range(25):stage = tick(.16, True)
        return stage

    def test_post_v10_versions_keep_earned_drive_distance_across_brief_gap(self):
        self.assertNotEqual(self.trajectory(0.), Stage.COMPLETE)
        for version in ('stage_v5', 'settle_v6', 'touch_v8', 'brake_v10', 'brake_v13', 'home_v15', 'stop_v16'):
            self.assertEqual(self.trajectory(support_memory_seconds({'reward_version': version})), Stage.COMPLETE)

    def test_pre_staged_versions_do_not_acquire_new_memory(self):
        for version in ('legacy', 'edge_v3', 'progress_v4'):
            self.assertEqual(support_memory_seconds({'reward_version': version}), 0.)


if __name__ == '__main__':unittest.main()
