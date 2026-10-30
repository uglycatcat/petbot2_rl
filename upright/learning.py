"""One native RSL-RL PPO model, with audited checkpoints and no warm-start path."""
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from .experiment import SCHEMA, atomic_json, check_identity, rng_state, restore_rng


def runner_config(cfg, model, device):
    lo, hi = model['joints']['left_frame']['range']
    home = (2*model['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    return dict(seed=cfg.seed, device=device, num_steps_per_env=cfg.rollout_steps,
        save_interval=cfg.checkpoint_interval, logger='tensorboard', experiment_name='upright_restart',
        run_name=f'round01_seed{cfg.seed}', obs_groups={'actor': ['policy'], 'critic': ['policy']},
        actor=dict(class_name='MLPModel', hidden_dims=[128, 128], activation='elu', obs_normalization=True,
            distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',
                                  initial_mean=[home, 0., 0.], initial_std=[.35, .45, .15], shape_cap=128.)),
        critic=dict(class_name='MLPModel', hidden_dims=[256, 128], activation='elu', obs_normalization=True),
        algorithm=dict(class_name='PPO', value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
            entropy_coef=.001, num_learning_epochs=5, num_mini_batches=4, learning_rate=cfg.learning_rate,
            schedule='fixed', gamma=.997**(cfg.decimation/10), lam=.95**(cfg.decimation/10),
            desired_kl=.01, max_grad_norm=1.))


class Adapter:
    def __init__(self, env):
        self.env, self.cfg, self.device, self.num_envs = env, env.cfg, env.device, env.num_envs
        self.num_actions = 3
        self.max_episode_length = env.max_episode_length

    @property
    def episode_length_buf(self): return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self, value): self.env.episode_length_buf = value

    def get_observations(self):
        # Do not append history a second time at a chunk boundary.
        return TensorDict(self.env.obs_buf, batch_size=[self.num_envs])

    def step(self, actions):
        obs, reward, terminated, timeout, extra = self.env.step(actions)
        extra['time_outs'] = timeout
        return TensorDict(obs, batch_size=[self.num_envs]), reward, (terminated | timeout).long(), extra


class Runner(OnPolicyRunner):
    def __init__(self, env, config, folder, manifest, device):
        self.manifest = manifest
        self.folder = Path(folder)
        self.runtime = env.env.upright_runtime
        self.checkpoints = self.folder/'checkpoints'
        self.checkpoints.mkdir(exist_ok=True)
        self.completed_updates = 0
        super().__init__(env, config, log_dir=str(folder/'tensorboard'), device=device)

    def save(self, path=None, infos=None):
        # Upstream assigns the zero-based iteration after the update. Save an
        # explicit completed-update count; never replay the last completed update.
        self._write_checkpoint(self.current_learning_iteration+1, infos)

    def _write_checkpoint(self, completed, infos=None):
        state = self.alg.save()
        state.update(iter=completed-1, infos=infos,
                     restart=dict(**self.manifest, updates_completed=completed,
                                  curriculum=self.runtime.course.state_dict(), rng=rng_state(),
                                  scenario_rng=self.runtime.generator.get_state()))
        target = self.checkpoints/f'update_{completed:06d}.pt'
        temporary = target.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(target)
        atomic_json(self.checkpoints/'latest.json', dict(file=target.name, updates_completed=completed))
        self.completed_updates = completed

    def load_same_run(self, checkpoint):
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        info = saved.get('restart', {})
        check_identity(info, self.manifest)
        updates = info['updates_completed']
        if updates <= 0 or updates > self.manifest['config']['iterations'] or saved['iter'] != updates-1:
            raise ValueError('Invalid completed-update counter')
        if not self.alg.load(saved, load_cfg=None, strict=True):
            raise RuntimeError('RSL-RL did not restore full training state')
        self.runtime.course.load_state_dict(info['curriculum'])
        self.current_learning_iteration = updates
        self.completed_updates = updates
        self.runtime.generator.set_state(info['scenario_rng'].cpu())
        restore_rng(info['rng'])
        # Physics solver/contact history is not serialized. Restart physical
        # episodes explicitly, using the restored course and random state.
        self.env.env.reset()
        return updates

    def load(self, *args, **kwargs):
        raise RuntimeError('Generic weight loading is forbidden. Use identity-checked same-run recovery.')

    def learn_chunk(self, count):
        self.current_learning_iteration = self.completed_updates
        super().learn(num_learning_iterations=count, init_at_random_ep_len=False)
        self.current_learning_iteration = self.completed_updates

    def save_boundary(self):
        self._write_checkpoint(self.completed_updates)
