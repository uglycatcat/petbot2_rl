import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from step_explore.rewards import motion_costs
from tools.research_logic import can_admit,next_config,descriptor_distance


class ResearchTests(unittest.TestCase):
    def test_local_soft_costs(self):
        q=torch.ones(3,3);qd=torch.zeros_like(q);a=torch.zeros(3,5)
        costs=motion_costs(q,qd,a,a,torch.tensor([.5,.2,.05]),.05)
        self.assertEqual(costs[2,1].item(),0.)
        self.assertGreater(costs[0,1].item(),costs[1,1].item())
        self.assertTrue(torch.equal(costs*torch.zeros(4),torch.zeros_like(costs)))
        self.assertTrue(torch.equal(costs[:,2:],torch.zeros(3,2)))

    def test_rate_cost_uses_seconds(self):
        q=torch.zeros(1,3);a=torch.ones(1,5);zero=torch.zeros_like(a)
        c1=motion_costs(q,q,a*.05,zero,torch.ones(1),.05)
        c2=motion_costs(q,q,a*.025,zero,torch.ones(1),.025)
        self.assertAlmostEqual(float(c1[0,2]),2*float(c2[0,2]),places=6)

    def test_admission_accounts_for_unallocated_peer_peak(self):
        self.assertFalse(can_admit(15000,{'a':9000},{'a':1000},9000,1500))
        self.assertTrue(can_admit(15000,{'a':9000},{'a':9000},9000,1500))

    def test_failed_height_is_retried_not_called_limit(self):
        initial={'seed':205,'research_track':'height','step_height':.10}
        failed={'step_height':.10,'success_rate':0.}
        self.assertEqual(next_config(initial,[failed],0)['step_height'],.10)
        self.assertEqual(next_config(initial,[failed,failed],0)['step_height'],.085)
        self.assertEqual(next_config(initial,[{'step_height':.10,'success_rate':1.}],0)['step_height'],.12)
        self.assertNotIn('checkpoint',next_config(initial,[failed],0))
        self.assertNotEqual(next_config(initial,[failed],0)['seed'],initial['seed'])

    def test_same_behavior_not_novel_just_due_to_seed(self):
        self.assertEqual(descriptor_distance([0,1,2],[0,1,2]),0.)

    def test_quality_adapts_settings_without_loading_weights(self):
        c={'seed':3,'research_track':'quality','head_default_weight':1.}
        passed={'success_rate':1.,'quality_ratio_to_baseline':1.1,'training_config':c}
        self.assertAlmostEqual(next_config(c,[passed],0)['head_default_weight'],1.35)
        failed={'success_rate':0.,'training_config':c}
        self.assertEqual(next_config(c,[failed],0)['head_default_weight'],.5)
        self.assertNotIn('checkpoint',next_config(c,[passed],0))


if __name__=='__main__':unittest.main()
