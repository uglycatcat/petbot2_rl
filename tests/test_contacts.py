import unittest
import torch
from step_demo.contacts import forbidden_front_contact


class FrontContactTests(unittest.TestCase):
    def test_top_contact_far_beyond_edge_is_not_a_face_collision(self):
        normal=torch.tensor([[[.0026,.0179,6.57],[.0036,-.0067,6.26]]])
        hit=forbidden_front_contact(normal,torch.tensor([[.15,.15]]),
                                   torch.tensor([[.104,.104]]),torch.tensor([0.]))
        self.assertFalse(bool(hit.any()))

    def test_front_face_and_edge_contact_remain_forbidden(self):
        normal=torch.tensor([[[-4.,0.,3.],[0.,0.,5.]]])
        hit=forbidden_front_contact(normal,torch.tensor([[0.,0.]]),
                                   torch.tensor([[-.03,-.03]]),torch.tensor([0.]))
        self.assertTrue(bool(hit.all()))

    def test_edge_point_evidence_is_not_overridden_by_bounds(self):
        hit=forbidden_front_contact(torch.tensor([[[0.,0.,1.]]]),
            torch.tensor([[0.]]),torch.tensor([[.1]]),torch.tensor([0.]))
        self.assertTrue(bool(hit.all()))


if __name__=='__main__':unittest.main()
