"""RSL-RL PPO configuration and same-run-only checkpoint persistence."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import BetaDistribution
from .distributions import CeilingBetaDistribution, std_ceiling_at
from .body_pitch import pitch_cost_scale_at


def runner_config(t):
    model = dict(class_name='MLPModel', hidden_dims=t['hidden_dims'], activation='elu', obs_normalization=True)
    distribution = (dict(class_name='BetaDistribution', action_range=(-1.,1.))
                    if t.get('policy_distribution','gaussian')=='beta' else
                    dict(class_name='GaussianDistribution', init_std=t['initial_std'], std_type='log'))
    if 'beta_std_ceiling' in t:
        name = ('ActionwiseCeilingBetaDistribution' if 'wheel_final_std' in t['beta_std_ceiling']
                else 'CeilingBetaDistribution')
        distribution['class_name']='step_demo.distributions:'+name
    return dict(num_steps_per_env=t['num_steps_per_env'], save_interval=t['save_interval'],
        logger='tensorboard', check_for_nan=True,
        obs_groups={'actor':['policy'], 'critic':(['policy','critic']
            if t.get('critic_observation', 'policy') == 'reward_state_v11' else ['policy'])},
        actor=dict(**model, distribution_cfg=distribution),
        critic=deepcopy(model), algorithm=dict(class_name='PPO', learning_rate=t['learning_rate'],
            num_learning_epochs=t['num_learning_epochs'], num_mini_batches=t['num_mini_batches'],
            clip_param=.2, gamma=t.get('discount_gamma',.99), lam=.95, entropy_coef=.01, desired_kl=.01,
            schedule=t.get('learning_rate_schedule','adaptive'), rnd_cfg=None, symmetry_cfg=None))


def distribution_metrics(distribution):
    mean=distribution.mean.detach();std=distribution.std.detach()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError('Nonfinite action distribution')
    kind=type(distribution).__name__
    if isinstance(distribution,BetaDistribution) and (mean.abs().max()>1.+1e-6 or std.max()>1.):
        raise RuntimeError('Bounded policy distribution violated its action range')
    result=dict(kind=kind,mean_abs_max=float(mean.abs().max()),std_min=float(std.min()),std_max=float(std.max()))
    if isinstance(distribution,CeilingBetaDistribution):
        result['std_ceiling']=distribution.std_ceiling.detach().cpu().tolist()
        if distribution.std_ceiling.ndim:
            result['std_min_per_action']=std.amin(dim=0).cpu().tolist()
            result['std_max_per_action']=std.amax(dim=0).cpu().tolist()
        if (std>distribution.std_ceiling+1e-6).any():raise RuntimeError('Action variance ceiling violated')
    return result


def state_digest(state):
    sha = hashlib.sha256()
    for key, value in sorted(state.items()):
        sha.update(key.encode()); sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


class StepRunner(OnPolicyRunner):
    def __init__(self, env, cfg, origin, log_dir=None, device='cuda:0'):
        self.origin = origin
        super().__init__(env, deepcopy(cfg), log_dir, device)
        self._set_std_ceiling(0)
        self._set_pitch_cost_scale(0, training=True)
        if log_dir:
            original_log=self.logger.log
            def log_progress(**values):
                losses={k:float(v) for k,v in values['loss_dict'].items()}
                if not all(math.isfinite(x) for x in losses.values()): raise RuntimeError('Nonfinite PPO loss')
                original_log(**values)
                record=dict(completed_updates=values['it']+1,losses=losses,
                    learning_rate=values['learning_rate'],collect_s=values['collect_time'],learn_s=values['learn_time'],
                    totals=env.unwrapped.totals.copy(),distribution=distribution_metrics(self.alg.actor.distribution),
                    body_pitch_cost_scale=env.unwrapped.body_pitch_cost_scale)
                write_json(Path(log_dir)/'progress.json',record)
                with (Path(log_dir)/'updates.jsonl').open('a') as file: file.write(json.dumps(record)+'\n')
                # Change only after PPO has consumed this rollout's log-probs.
                # Checkpoints now contain the ceiling for the next collection.
                self._set_std_ceiling(values['it']+1)
                self._set_pitch_cost_scale(values['it']+1, training=True)
            self.logger.log=log_progress

    def _set_std_ceiling(self, completed_updates):
        schedule=self.origin['training'].get('beta_std_ceiling')
        if schedule is not None:
            self.alg.actor.distribution.set_std_ceiling(std_ceiling_at(schedule,completed_updates))

    def _set_pitch_cost_scale(self, completed_updates, *, training):
        schedule = self.origin['training'].get('body_pitch_curriculum')
        self.env.unwrapped.body_pitch_cost_scale = pitch_cost_scale_at(schedule, completed_updates) if training else 1.

    def save(self, path, infos=None):
        data = self.alg.save()
        data.update(origin=self.origin, iter=self.current_learning_iteration,
            completed_updates=self.current_learning_iteration+1, infos=infos,
            learning_rate=self.alg.learning_rate,
            body_pitch_cost_scale=self.env.unwrapped.body_pitch_cost_scale,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                     python=random.getstate(),numpy=np.random.get_state()),
            totals=self.env.unwrapped.totals,
            logger=dict(tot_timesteps=self.logger.tot_timesteps,tot_time=self.logger.tot_time))
        path = Path(path); temp = path.with_suffix('.pt.tmp')
        torch.save(data, temp); temp.replace(path)
        write_json(path.parent/'latest.json', dict(checkpoint=path.name,
            completed_updates=data['completed_updates'],actor_sha256=state_digest(data['actor_state_dict'])))

    def restore(self, path, *, training):
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get('origin') != self.origin:
            raise ValueError('Checkpoint origin/config/source differs: transfer learning is forbidden')
        if data.get('completed_updates') != data['iter']+1:
            raise ValueError('Invalid checkpoint iteration accounting')
        pitch_schedule = self.origin['training'].get('body_pitch_curriculum')
        if pitch_schedule is not None:
            expected = pitch_cost_scale_at(pitch_schedule, data['completed_updates'])
            if not math.isclose(data.get('body_pitch_cost_scale', float('nan')), expected, abs_tol=1e-9):
                raise ValueError('Checkpoint body pitch curriculum is inconsistent')
        self.alg.load(data, None, strict=True)
        self.alg.learning_rate = data['learning_rate']
        self.current_learning_iteration = data['completed_updates']
        self._set_pitch_cost_scale(data['completed_updates'], training=training)
        schedule=self.origin['training'].get('beta_std_ceiling')
        if schedule is not None:
            actual=self.alg.actor.distribution.std_ceiling
            expected=torch.as_tensor(std_ceiling_at(schedule,data['completed_updates']),
                                     dtype=actual.dtype,device=actual.device)
            if actual.shape!=expected.shape or not torch.allclose(actual,expected,atol=1e-6,rtol=0):
                raise ValueError('Checkpoint action variance curriculum is inconsistent')
        if training:
            torch.set_rng_state(data['rng']['torch'].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in data['rng']['cuda']])
            random.setstate(data['rng']['python']); np.random.set_state(data['rng']['numpy'])
            self.env.unwrapped.totals = data['totals'].copy()
            for key,value in data['logger'].items(): setattr(self.logger,key,value)
        return dict(completed_updates=data['completed_updates'],
            actor_sha256=state_digest(self.alg.actor.state_dict()),
            critic_sha256=state_digest(self.alg.critic.state_dict()),
            optimizer_states=len(self.alg.optimizer.state),
            learning_rate=self.alg.learning_rate,
            body_pitch_cost_scale=self.env.unwrapped.body_pitch_cost_scale,
            physics_episodes_restarted=True)
