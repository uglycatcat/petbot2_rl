import unittest
import torch
from step_explore.rewards import TOP_PEAK_VALUES, simultaneous_top_bonus

class PeakSupportRewardTest(unittest.TestCase):
    def test_alternating_pairs_are_only_two_simultaneous_wheels(self):
        peak=torch.zeros(1,dtype=torch.long);levels=torch.tensor(TOP_PEAK_VALUES)
        total=0.
        for contacts in [[True,True,False,False],[False,False,True,True]]:
            reward,peak=simultaneous_top_bonus(torch.tensor([contacts]),peak,levels)
            total+=reward.item()
        self.assertEqual(total,6.)
        self.assertEqual(peak.item(),2)

    def test_losing_and_regaining_support_does_not_pay_again(self):
        peak=torch.zeros(1,dtype=torch.long);levels=torch.tensor(TOP_PEAK_VALUES);rewards=[]
        for count in [2,3,2,3,4,4]:
            contacts=torch.arange(4)[None,:]<count
            reward,peak=simultaneous_top_bonus(contacts,peak,levels);rewards.append(reward.item())
        self.assertEqual(rewards,[6.,6.,0.,0.,6.,0.])

    def test_environments_keep_independent_progress(self):
        contacts=torch.tensor([[True,True,False,False],[True,True,True,False]])
        reward,peak=simultaneous_top_bonus(contacts,torch.tensor([0,3]),torch.tensor(TOP_PEAK_VALUES))
        self.assertEqual(reward.tolist(),[6.,0.])
        self.assertEqual(peak.tolist(),[2,3])

if __name__=='__main__':unittest.main()
