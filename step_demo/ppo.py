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


def runner_config(t):
    model = dict(class_name='MLPModel', hidden_dims=t['hidden_dims'], activation='elu', obs_normalization=True)
    distribution = (dict(class_name='BetaDistribution', action_range=(-1.,1.))
                    if t.get('policy_distribution','gaussian')=='beta' else
                    dict(class_name='GaussianDistribution', init_std=t['initial_std'], std_type='log'))
    return dict(num_steps_per_env=t['num_steps_per_env'], save_interval=t['save_interval'],
        logger='tensorboard', check_for_nan=True, obs_groups={'actor':['policy'],'critic':['policy']},
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
    if kind=='BetaDistribution' and (mean.abs().max()>1.+1e-6 or std.max()>1.):
        raise RuntimeError('Bounded policy distribution violated its action range')
    return dict(kind=kind,mean_abs_max=float(mean.abs().max()),std_min=float(std.min()),std_max=float(std.max()))


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
        if log_dir:
            original_log=self.logger.log
            def log_progress(**values):
                losses={k:float(v) for k,v in values['loss_dict'].items()}
                if not all(math.isfinite(x) for x in losses.values()): raise RuntimeError('Nonfinite PPO loss')
                original_log(**values)
                record=dict(completed_updates=values['it']+1,losses=losses,
                    learning_rate=values['learning_rate'],collect_s=values['collect_time'],learn_s=values['learn_time'],
                    totals=env.unwrapped.totals.copy(),distribution=distribution_metrics(self.alg.actor.distribution))
                write_json(Path(log_dir)/'progress.json',record)
                with (Path(log_dir)/'updates.jsonl').open('a') as file: file.write(json.dumps(record)+'\n')
            self.logger.log=log_progress

    def save(self, path, infos=None):
        data = self.alg.save()
        data.update(origin=self.origin, iter=self.current_learning_iteration,
            completed_updates=self.current_learning_iteration+1, infos=infos,
            learning_rate=self.alg.learning_rate,
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
        self.alg.load(data, None, strict=True)
        self.alg.learning_rate = data['learning_rate']
        self.current_learning_iteration = data['completed_updates']
        if training:
            torch.set_rng_state(data['rng']['torch'].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in data['rng']['cuda']])
            random.setstate(data['rng']['python']); np.random.set_state(data['rng']['numpy'])
            self.env.unwrapped.totals = data['totals'].copy()
            for key,value in data['logger'].items(): setattr(self.logger,key,value)
        return dict(completed_updates=data['completed_updates'],
            actor_sha256=state_digest(self.alg.actor.state_dict()),
            optimizer_states=len(self.alg.optimizer.state),
            learning_rate=self.alg.learning_rate,
            physics_episodes_restarted=True)
