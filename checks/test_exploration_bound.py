"""Numerical PPO/ONNX contracts for a configurable exploration floor; no simulator training."""
import contextlib,io,tempfile,unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from upright.config import Config,load_config
from upright.distribution import UprightBetaDistribution,parameters
from upright.export import make_actor,export_actor

class ExplorationBound(unittest.TestCase):
    def test_same_initial_physical_distribution(self):
        obs=TensorDict({'policy':torch.zeros(64,195)},batch_size=[64]);models=[]
        for cap in (128.,512.):
            with contextlib.redirect_stdout(io.StringIO()): model=make_actor(Config(beta_shape_cap=cap))
            model(obs,stochastic_output=True);models.append(model)
        a,b=[m.distribution for m in models]
        self.assertTrue(torch.allclose(a.mean,b.mean,atol=2e-7,rtol=0))
        self.assertTrue(torch.allclose(a.std,b.std,atol=2e-7,rtol=0))
        self.assertTrue(torch.allclose(b.std,torch.tensor([.35,.45,.15]).expand(64,-1),atol=2e-7,rtol=0))
        self.assertTrue(torch.allclose(a.kl_divergence(a.params,b.params),torch.zeros(64),atol=2e-5,rtol=0))

    def test_exploration_floor_reduced_and_physical_bounds_preserved(self):
        ds=[]
        for cap in (128.,512.):
            d=UprightBetaDistribution(3,[0,0,0],[.35,.45,.15],cap)
            d.update(torch.full((128,2,3),1e6));ds.append(d)
            sample=d.sample();self.assertTrue(torch.isfinite(d.log_prob(sample)).all())
            self.assertLessEqual(float(sample.abs().max()),1.)
            self.assertTrue(torch.allclose(d.mean,torch.zeros(128,3),atol=1e-7))
        self.assertLess(float(ds[1].std.max()),float(ds[0].std.min())*.51)
        self.assertGreater(float(ds[1].std.min()),0.03)

    def test_extreme_head_and_native_gradients_finite(self):
        d=UprightBetaDistribution(3,[0,0,0],[.35,.45,.15],512.)
        x=torch.tensor([[[-100.,0.,100.],[100.,0.,-100.]]],requires_grad=True)
        d.update(x);sample=d.sample().detach();old=tuple(p.detach().clone() for p in d.params)
        loss=-d.log_prob(sample).mean()-.001*d.entropy.mean()+d.kl_divergence(old,d.params).mean()
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(d.deterministic_output(x)).all());self.assertLessEqual(float(d.mean.abs().max().detach()),1.)

    def test_export_retains_new_deterministic_transform(self):
        cfg=Config(beta_shape_cap=512.)
        with contextlib.redirect_stdout(io.StringIO()): model=make_actor(cfg)
        # Include non-initialized outputs so parity cannot pass on constant bias alone.
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name.startswith('mlp.'):p.add_(torch.randn_like(p)*.03)
        with tempfile.TemporaryDirectory() as folder:
            result=export_actor(model,cfg,Path(folder)/'export',{'synthetic':True})
        self.assertLessEqual(result['numeric_max_abs_error'],1e-5)
        self.assertFalse(result['simulation_validated'])

    def test_default_and_legacy_configs(self):
        self.assertEqual(load_config().round,23)
        self.assertEqual(load_config().beta_shape_cap,512.)
        self.assertEqual(Config().beta_shape_cap,128.)
        for cap in (0.,float('nan'),256.,True):
            with self.assertRaises(ValueError):Config(beta_shape_cap=cap).validate()

if __name__=='__main__':unittest.main()
