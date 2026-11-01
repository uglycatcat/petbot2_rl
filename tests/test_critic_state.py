"""Reward-history observability and strict actor/critic separation."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from step_demo.config import PROJECT_ROOT
from step_demo.critic_state import CRITIC_STATE_FIELDS, reward_state
from step_demo.ppo import runner_config
from step_demo.runtime import load_training


class CriticStateTests(unittest.TestCase):
    def inputs(self):
        return dict(credit=torch.zeros(2,8), elapsed=torch.tensor([0.,15.]), timeout=15.,
            phase=torch.tensor([0,3]), supported_top=torch.zeros(2,4,dtype=torch.bool),
            front_landed=torch.zeros(2,2,dtype=torch.bool), forward=torch.tensor([0.,.8]),
            drive_start=torch.tensor([float('nan'),.5]), drive_distance=.15,
            support_gap=torch.tensor([0.,.08]), home_hold=torch.tensor([0.,.3]),
            stop_time=torch.tensor([0.,.6]), stop_hold=.5, near_top=torch.tensor([False,True]))

    def test_missing_drive_origin_is_finite_and_explicit(self):
        s=reward_state(**self.inputs())
        self.assertEqual(s.shape,(2,22));self.assertEqual(len(CRITIC_STATE_FIELDS),22)
        self.assertTrue(bool(torch.isfinite(s).all()))
        self.assertEqual(float(s[0,16]),0.);self.assertEqual(float(s[0,17]),0.)
        self.assertEqual(float(s[1,16]),1.);self.assertAlmostEqual(float(s[1,17]),2.)

    def test_reward_history_distinguishes_same_physical_state(self):
        a=self.inputs();b=self.inputs();b['credit'][0]=torch.tensor([6.,1.,1.,1.5,1.5,2.,2.,2.])
        x=reward_state(**a);y=reward_state(**b)
        torch.testing.assert_close(x[:,8:],y[:,8:]);torch.testing.assert_close(y[0,:8],torch.ones(8))
        self.assertFalse(torch.equal(x,y))

    def test_terminal_and_gap_features_are_bounded(self):
        s=reward_state(**self.inputs())
        self.assertTrue(bool((s>=0).all() and (s<=2).all()))
        self.assertEqual(float(s[1,8]),1.);self.assertEqual(float(s[1,19]),1.)
        self.assertEqual(float(s[1,20]),1.)

    def test_contacts_are_critic_features_not_extra_terrain_packets(self):
        x=self.inputs();x['supported_top'][0]=True;x['front_landed'][0]=True
        s=reward_state(**x);torch.testing.assert_close(s[0,10:16],torch.ones(6))

    def test_actor_works_without_critic_and_ignores_private_state(self):
        t=load_training(PROJECT_ROOT/'configs/round11_gate.json');cfg=runner_config(t)
        policy=torch.randn(2,26);state=reward_state(**self.inputs()).requires_grad_()
        td=TensorDict({'policy':policy,'critic':state},[2]);args=dict(cfg['actor']);args.pop('class_name')
        actor=MLPModel(td,cfg['obs_groups'],'actor',2,**args);actor.eval()
        a=actor.mlp(actor.get_latent(td));b=actor.mlp(actor.get_latent(TensorDict({'policy':policy},[2])))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        changed=actor.mlp(actor.get_latent(TensorDict({'policy':policy,'critic':torch.ones_like(state)*99},[2])))
        torch.testing.assert_close(a,changed,rtol=0,atol=0)
        a.sum().backward()
        self.assertIsNone(state.grad)
        c=dict(cfg['critic']);c.pop('class_name');critic=MLPModel(td,cfg['obs_groups'],'critic',1,**c);critic.eval()
        self.assertEqual(critic.get_latent(td).shape[-1],48)
        self.assertEqual(actor.get_latent(td).shape[-1],26)

    def test_legacy_critic_keeps_policy_only(self):
        cfg=runner_config(load_training(PROJECT_ROOT/'configs/round10_seed70.json'))
        self.assertEqual(cfg['obs_groups'],{'actor':['policy'],'critic':['policy']})

    def test_formal_configs_change_only_critic_information(self):
        for seed in [70,71]:
            old=load_training(PROJECT_ROOT/f'configs/round10_seed{seed}.json')
            new=load_training(PROJECT_ROOT/f'configs/round11_seed{seed}.json')
            self.assertEqual(new.pop('critic_observation'),'reward_state_v11');self.assertEqual(old,new)

    def test_unknown_critic_interface_is_rejected(self):
        d=json.loads((PROJECT_ROOT/'configs/round11_gate.json').read_text());d['critic_observation']='both_networks'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):load_training(p)


if __name__=='__main__':unittest.main()
