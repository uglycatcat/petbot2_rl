"""Physical-time contracts for the 30ms policy experiment."""
import math
import unittest
import torch
from upright.config import load_config, ROOT
from upright.learning import runner_config
from upright.controller import Controller, Phase
from upright.runtime import Runtime
from test_contracts import MODEL, state


class PolicyPeriod(unittest.TestCase):
    def configs(self):
        return [load_config(ROOT/f'configs/round{n}.json') for n in (18,20)]

    def test_physical_rollout_and_return_horizons(self):
        old,new=self.configs()
        a,b=[runner_config(c,MODEL,'cpu')['algorithm'] for c in (old,new)]
        self.assertEqual(old.rollout_steps*old.decimation,new.rollout_steps*new.decimation)
        self.assertAlmostEqual(b['gamma']**2,a['gamma'])
        self.assertAlmostEqual(b['lam']**2,a['lam'])
        for k in ('num_learning_epochs','num_mini_batches','learning_rate'):
            self.assertEqual(a[k],b[k])
        self.assertEqual(new.history,5)
        self.assertEqual(new.decimation,15)
        self.assertEqual(new.balance_velocity_coarse_weight,0.)

    def test_reference_and_actuator_slew_match_elapsed_seconds(self):
        controls=[]
        for cfg,steps in zip(self.configs(),(1,2)):
            c=Controller(2,'cpu',.002*cfg.decimation,cfg,MODEL,.1)
            c.phase[:]=Phase.BALANCE
            c.set_command([[.75,0],[-.75,0]])
            for _ in range(steps):
                c.update_motion_reference()
                c.targets(torch.tensor([[1.,1.,0.],[-1.,-1.,0.]]),state())
            controls.append(c)
        torch.testing.assert_close(controls[0].motion_command(),controls[1].motion_command())
        torch.testing.assert_close(controls[0].executed,controls[1].executed)

    def test_velocity_error_filter_keeps_physical_time_constant(self):
        values=[]
        for cfg,steps in zip(self.configs(),(1,2)):
            alpha=1-math.exp(-.002*cfg.decimation/cfg.balance_velocity_error_tau_s)
            error=torch.zeros(2)
            for _ in range(steps):error+=(torch.tensor([.5,-.5])-error)*alpha
            values.append(error)
        torch.testing.assert_close(values[0],values[1])

    def test_seconds_based_contracts_and_source_constraints_preserved(self):
        old,new=self.configs()
        allowed={'round','decimation','rollout_steps','gae_lambda'}
        for k,v in old.to_dict().items():
            if k not in allowed:self.assertEqual(v,new.to_dict()[k],k)
        self.assertEqual(MODEL['physics_dt'],.002)

if __name__=='__main__':unittest.main()
