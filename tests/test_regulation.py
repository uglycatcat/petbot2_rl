"""Return-order and scope checks for continuing auxiliary stopping."""
import json
from pathlib import Path
import tempfile
import unittest
from step_demo.regulation import discounted_failure_cost
from step_demo.runtime import load_training
from step_demo.config import PROJECT_ROOT


class RegulationTests(unittest.TestCase):
    def test_early_failure_cannot_escape_discounted_running_costs(self):
        gamma=.99;cost=.02*6;penalty=discounted_failure_cost(6,.02,gamma)
        self.assertAlmostEqual(penalty,15.)
        for steps in (1,10,25,100,750,10000):
            delayed=-cost*(1-gamma**steps)/(1-gamma)-gamma**steps*penalty
            self.assertGreater(delayed,-penalty)
        self.assertGreater(-cost/(1-gamma),-penalty)

    def test_perfect_stationary_control_dominates_nonzero_errors(self):
        gamma=.99
        for cost_rate in (.01,.1,1.,5.7):
            self.assertGreater(0.,-.02*cost_rate/(1-gamma))

    def test_finite_horizon_full_configs_cannot_take_auxiliary_shortcuts(self):
        valid=load_training(PROJECT_ROOT/'configs/restart17_aux_seed70.json')
        self.assertEqual(valid['discount_gamma'],.99)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for changes in ({'task_variant':'full'},{'discount_gamma':1.},
                            {'terminate_auxiliary_flight':False},{'auxiliary_objective':'typo'},
                            {'_evaluation':True}):
                path.write_text(json.dumps(dict(valid,**changes)))
                with self.assertRaises(ValueError):load_training(path)
        self.assertEqual(load_training(PROJECT_ROOT/'configs/restart16_aux_seed70.json')['discount_gamma'],1.)


if __name__=='__main__':unittest.main()
