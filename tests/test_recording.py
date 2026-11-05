import unittest
import torch
from step_demo.recording import records_from_tensors


class RecordingTests(unittest.TestCase):
    def test_mixed_types_and_selected_rows_match_scalar_records(self):
        fields=dict(env=torch.tensor([9,3]),episode=torch.tensor([16777217,22]),
                    success=torch.tensor([True,False]),
                    hips=torch.tensor([[.123456,-.123456],[.5,.8]]),
                    flights=torch.tensor([[1,1],[2,1]]))
        expected=[{k:v[i].tolist() for k,v in fields.items()} for i in range(2)]
        self.assertEqual(records_from_tensors(fields),expected)
        self.assertIs(type(records_from_tensors(fields)[0]['success']),bool)
        self.assertEqual(records_from_tensors({k:v[:0] for k,v in fields.items()}),[])
