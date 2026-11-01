"""Privileged value inputs must not alter actor, physical control or reward."""
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from upright.config import Config
from upright.controller import Phase
from upright.export import make_actor, export_actor
from upright.learning import runner_config
from upright.runtime import Runtime, OBSERVATION_DIM, CRITIC_TASK_DIM, CRITIC_TASK_FIELDS
from test_contracts import MODEL, state, truth

CFG = Config(num_envs=32, critic_task_state=True, balance_observation_age_cap_s=2.,
             balance_velocity_cost=2., balance_velocity_cost_cap=1.5,
             balance_velocity_cost_tail='rational', balance_train_s=30., balance_eval_s=60.)

class CriticStateContracts(unittest.TestCase):
    def test_hidden_reward_history_and_timing_are_distinguished_only_by_critic(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);s=state()
        p.control.phase[:]=Phase.BALANCE;p.control.age[:]=torch.tensor([3.,20.])
        p.balance_duration[:]=torch.tensor([6.,30.]);p.balance_elapsed[:]=torch.tensor([3.,20.])
        p.balance_velocity_error[:]=torch.tensor([.2,-.2])
        p.peak_rise[:]=torch.tensor([.2,1.]);p.reversal_speed[:]=torch.tensor([.5,-.5])
        torch.testing.assert_close(p.observation(s)[0],p.observation(s)[1])
        v=p.critic_state();self.assertFalse(torch.equal(v[0],v[1]))
        for field in ['phase_age','balance_duration','velocity_error_mean','peak_rise','next_reversal_velocity']:
            self.assertNotEqual(float(v[0,CRITIC_TASK_FIELDS.index(field)]),float(v[1,CRITIC_TASK_FIELDS.index(field)]))

    def test_bounded_shape_current_state_and_pure_read(self):
        p=Runtime(6,'cpu',.06,CFG,MODEL);p.control.phase[:]=torch.arange(6)
        p.control.age[:]=1e5;p.balance_velocity_error[:]=-1e5;p.balance_elapsed[:]=1e5
        p.control.command[:]=torch.tensor([1e5,-1e5]);p.reversal_speed[:]=1e5
        rng=p.generator.get_state().clone();before={k:v.clone() for k,v in vars(p).items() if isinstance(v,torch.Tensor)}
        v=p.critic_state();self.assertEqual(v.shape,(6,24));self.assertEqual(CRITIC_TASK_DIM,24)
        self.assertTrue(torch.isfinite(v).all() and (v.abs()<=1).all())
        torch.testing.assert_close(v[:,CRITIC_TASK_FIELDS.index('velocity_error_mean')],-torch.ones(6))
        torch.testing.assert_close(rng,p.generator.get_state())
        for k,x in before.items():torch.testing.assert_close(x,getattr(p,k))

    def test_subset_reset_does_not_leak_previous_episode_memory(self):
        p=Runtime(2,'cpu',.06,CFG,MODEL);p.control.phase[:]=Phase.BALANCE;p.control.age[:]=4
        p.peak_rise[:]=1;p.peak_land[:]=1;p.balance_velocity_error[:]=.2;p.invoked[:]=True
        p.reversal_speed[:]=.5;p.reversal_episode[:]=True;p.short_cycle[:]=True
        before=p.critic_state()[1].clone();p.reset(torch.tensor([0]));after=p.critic_state()
        torch.testing.assert_close(before,after[1])
        for name in ['peak_rise','peak_land','velocity_error_mean','invoked','short_cycle','reversal_episode','next_reversal_velocity']:
            self.assertEqual(float(after[0,CRITIC_TASK_FIELDS.index(name)]),0.)

    def test_extra_read_cannot_change_scenario_reward_or_actor(self):
        old=Runtime(2,'cpu',.06,replace(CFG,critic_task_state=False),MODEL)
        new=Runtime(2,'cpu',.06,CFG,MODEL);s=state();s['gap'][:]=.1;s['motion'][:,0]=.1
        for p in (old,new):
            p.reset(torch.arange(2));p.control.phase[:]=Phase.BALANCE;p.invoked[:]=True;p.level[:]=2
        for _ in range(10):
            new.critic_state()
            a,b=old.advance(s,truth(raised=True)),new.advance(s,truth(raised=True))
            for k in a:torch.testing.assert_close(a[k],b[k])
            torch.testing.assert_close(old.observation(s),new.observation(s))
            torch.testing.assert_close(old.control.targets(torch.zeros(2,3),s),new.control.targets(torch.zeros(2,3),s))
            torch.testing.assert_close(old.generator.get_state(),new.generator.get_state())

    def test_native_critic_uses_extra_features_actor_ignores_them(self):
        config=runner_config(CFG,MODEL,'cpu');groups=config['obs_groups']
        self.assertEqual(groups,{'actor':['policy'],'critic':['policy','critic_task']})
        self.assertEqual(runner_config(Config(),MODEL,'cpu')['obs_groups']['critic'],['policy'])
        obs=TensorDict({'policy':torch.randn(4,OBSERVATION_DIM*CFG.history),
                        'critic_task':torch.randn(4,CRITIC_TASK_DIM,requires_grad=True)},batch_size=[4])
        kwargs=config['critic'].copy();kwargs.pop('class_name')
        with contextlib.redirect_stdout(io.StringIO()):
            actor=make_actor(CFG);critic=MLPModel(obs,groups,'critic',1,**kwargs)
        result=actor(obs);changed=obs.clone();changed['critic_task']+=.5
        torch.testing.assert_close(result,actor(changed))
        values=critic(obs);values.sum().backward()
        self.assertIsNotNone(obs['critic_task'].grad);self.assertTrue(obs['critic_task'].grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_actor_onnx_still_needs_only_195_sensor_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):actor=make_actor(CFG)
        with tempfile.TemporaryDirectory() as name:
            report=export_actor(actor,CFG,Path(name)/'numeric_only',{'synthetic_untrained_actor':True})
        self.assertEqual(report['frame_dim']*report['history'],195)
        self.assertLessEqual(report['numeric_max_abs_error'],1e-5)
        self.assertFalse(report['simulation_validated'])

    def test_opt_in_and_timer_validation(self):
        self.assertFalse(Config().critic_task_state)
        for value in (1,0,'true',None):
            with self.assertRaises(ValueError):replace(CFG,critic_task_state=value).validate()
        for name in ['return_confirm_s','ground_confirm_s','lifted_confirm_s','handoff_hold_s']:
            with self.assertRaises(ValueError):replace(CFG,**{name:0.}).validate()

if __name__=='__main__':unittest.main()
