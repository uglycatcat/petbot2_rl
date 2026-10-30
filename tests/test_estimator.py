"""Sensor-only integration checks, run with the container's PyTorch."""
import math
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch tests run inside the dedicated container')
class EstimatorTests(unittest.TestCase):
    def estimator(self):
        from step_demo.estimator import InertialEstimator
        return InertialEstimator(2,'cpu',[dict(parent=-1,mass=1.,pos=[0.,0.,0.],
            com=[0.,0.,0.],axis=[0.,0.,0.],joint=None)],['hip_l','hip_r','wheel_l','wheel_r'])

    def test_rest_and_constant_acceleration(self):
        e=self.estimator(); joints=torch.zeros(2,4);gyro=torch.zeros(2,3)
        accel=torch.tensor([[0.,0.,9.81],[1.,0.,9.81]])
        for _ in range(500): e.update(accel,gyro,joints,.002)
        torch.testing.assert_close(e.com_position,torch.tensor([[0.,0.,0.],[.5,0.,0.]]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(e.com_velocity,torch.tensor([[0.,0.,0.],[1.,0.,0.]]),atol=1e-5,rtol=1e-5)

    def test_turn_updates_bearing_and_reset_is_selective(self):
        e=self.estimator();joints=torch.zeros(2,4)
        gyro=torch.tensor([[0.,0.,math.pi/2],[0.,0.,0.]])
        for _ in range(500): e.update(torch.tensor([[0.,0.,9.81]]).repeat(2,1),gyro,joints,.002)
        packet=torch.tensor([[.02,.3,math.pi/2]]).repeat(2,1)
        obs=e.observation(packet,joints,joints,[0,1],[2,3],gyro,torch.zeros(2,2))
        self.assertEqual(tuple(obs.shape),(2,26));self.assertAlmostEqual(float(obs[0,6]),1.,places=5)
        self.assertAlmostEqual(float(obs[0,4]),.3,places=5)
        e.reset(torch.tensor([0]));self.assertEqual(e.samples.tolist(),[0,500])
        self.assertEqual(e.q[0].tolist(),[0.,0.,0.,1.])

    def test_matches_scalar_observation_contract(self):
        from step_demo.observations import ObservationBuilder,TerrainPacket,MotionEstimate,Proprioception
        e=self.estimator();joints=torch.zeros(2,4)
        packet=torch.tensor([[.02,.3,math.pi/2]]).repeat(2,1)
        previous=torch.tensor([[0.,-1.]]).repeat(2,1)
        actual=e.observation(packet,joints,joints,[0,1],[2,3],torch.zeros(2,3),previous)[0]
        b=ObservationBuilder();b.receive_initial_terrain(TerrainPacket(.02,.3,math.pi/2))
        expected=b.build(Proprioception((0.,0.,-1.),(0.,0.,0.),(0.,0.),(0.,0.),(0.,0.)),
                         MotionEstimate(0.,0.,0.,0.,0.),(0.,-1.))
        torch.testing.assert_close(actual,torch.tensor(expected),atol=1e-6,rtol=1e-6)

    def test_checkpoint_origin_mismatch_is_rejected_before_loading(self):
        from step_demo.ppo import StepRunner
        import tempfile
        from pathlib import Path
        runner=StepRunner.__new__(StepRunner);runner.origin={'run_id':'new'};runner.device='cpu'
        with tempfile.TemporaryDirectory(prefix='step-origin-') as directory:
            p=Path(directory)/'foreign.pt';torch.save({'origin':{'run_id':'old'}},p)
            with self.assertRaises(ValueError): runner.restore(p,training=True)

if __name__=='__main__': unittest.main()
