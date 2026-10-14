"""Paired first-episode evaluation on identical states, plus fixed-action diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint')
parser.add_argument('--task', default='PetBot-Recovery-Sweep2-A-v0')
parser.add_argument('--mode', choices=['policy', 'default', 'zero', 'sequence'], default='policy')
parser.add_argument('--episodes', type=int, default=512)
parser.add_argument('--seed', type=int, default=20261015)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == 'policy' and not args.checkpoint:
    parser.error('policy evaluation requires --checkpoint')
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch
    import agile.rl_env.tasks.stand_up.petbot  # noqa: F401
    from agile.petbot.core import as_torch
    from agile.petbot.model import POSITION_JOINTS, JOINT_ORDER, load_model
    from agile.petbot.pre_learn import pre_learn
    from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner
    from isaaclab_tasks.utils import load_cfg_from_registry

    task = args.task
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint:
        saved = json.loads((checkpoint.parent / 'params/petbot.json').read_text())
        if saved['source_model']['source_sha256'] != load_model()['source_sha256']:
            raise ValueError('Source model fingerprint mismatch')
        task = saved['task']
    env_cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent_cfg = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    env_cfg.scene.num_envs = args.episodes
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or 'cuda:0'
    agent_cfg.seed = args.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.petbot_collection.split = 'eval'
    agent_cfg.petbot_collection.seed = args.seed
    agent_cfg.petbot_collection.num_states = args.episodes
    env = gym.make(task, cfg=env_cfg)
    base = env.unwrapped
    pre_learn(base, task, agent_cfg)
    # Hash actual tensors, not a cache path, so independent run copies can be compared.
    digest = hashlib.sha256()
    for key in ('root', 'joint_pos', 'joint_vel'):
        digest.update(base.petbot_dataset[key].contiguous().numpy().tobytes())
    dataset_hash = digest.hexdigest()
    base.petbot_eval_cursor = 0
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    base.petbot_episode_records = []
    obs = wrapped.get_observations()
    if checkpoint:
        # Evaluate only configurations actually used by this sweep, never silently drop saved fields.
        from isaaclab.utils.io import load_yaml
        recorded_env = load_yaml(str(checkpoint.parent / 'params/env.yaml'))
        recorded_agent = load_yaml(str(checkpoint.parent / 'params/agent.yaml'))
        for field in ('history_steps', 'shaping_gamma', 'episode_length_s', 'decimation'):
            if field in recorded_env and recorded_env[field] != getattr(env_cfg, field):
                raise ValueError(f'Evaluation configuration mismatch: {field}')
        if saved['actor_dim'] != 27 * getattr(env_cfg, 'history_steps', 1):
            raise ValueError('History/actor dimension mismatch')
        # Runner-only overrides (steps/lr/seeds) do not change policy architecture here.
        for field in ('actor_hidden_dims', 'critic_hidden_dims', 'activation', 'noise_std_type', 'noise_std_range'):
            if field in recorded_agent['policy']:
                setattr(agent_cfg.policy, field, recorded_agent['policy'][field])
        runner = make_rsl_rl_runner(wrapped, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_cfg={'actor': True, 'critic': True, 'optimizer': False, 'iteration': False})
        policy = runner.get_inference_policy(device=wrapped.device)
    robot = base.scene['robot']
    ids = robot.find_joints(list(POSITION_JOINTS), preserve_order=True)[0]
    q_start = as_torch(robot.data.joint_pos)[:, ids].clone()
    home = as_torch(robot.data.default_joint_pos)[:, ids].clone()
    spec = load_model()
    bounds = torch.tensor([spec['joints'][n]['range'] for n in POSITION_JOINTS], device=base.device)
    offset = bounds.mean(-1)
    scale = (bounds[:, 1] - bounds[:, 0]) / 2
    fixed = torch.zeros((args.episodes, 7), device=base.device)
    fixed[:, :5] = (home-offset) / scale
    initial_gravity = as_torch(robot.data.projected_gravity_b).clone()
    positive = initial_gravity[:, 0] >= 0  # sin(pitch)
    fallen = -initial_gravity[:, 2] < 2**-0.5
    first = {}
    traces = []
    action_stats = {'samples': 0, 'clipped': 0}
    for step in range(base.max_episode_length + 5):
        with torch.no_grad():
            if args.mode == 'policy':
                action = policy(obs)
            elif args.mode == 'default':
                action = fixed.clone()
            elif args.mode == 'zero':
                action = torch.zeros_like(fixed)
            else:
                # Same fixed-time reference as pose_recover_demo, via PhysX implicit drives.
                # This is a diagnostic of control/contacts, not a faithful external-PD transfer.
                t = step * base.step_dt
                def smooth(x):
                    x = min(max(x, 0.), 1.)
                    return x*x*(3-2*x)
                q = home.clone()
                wheels = torch.zeros(args.episodes, device=base.device)
                if t < 1:
                    q = q_start + (home-q_start) * smooth(t)
                else:
                    for is_positive in (False, True):
                        mask = fallen & (positive == is_positive)
                        local = t - (4 if is_positive else 0)
                        if is_positive and t < 4:
                            if t < 2:
                                q[mask, 3:] = -smooth(t-1)
                            elif t < 3:
                                q[mask, 3] = -1+2*smooth(t-2)
                                q[mask, 4] = -1
                            else:
                                q[mask, 3] = 1-1.4*smooth(t-3)
                                q[mask, 4] = -1+0.6*smooth(t-3)
                        elif local < 1:
                            q[mask, 3:] = -0.4*(1-smooth(local))
                        elif local < 2:
                            q[mask, 1:3] = smooth(local-1)
                            q[mask, 3:] = 1.2*smooth(local-1)
                        elif local < 3:
                            q[mask, 1:3] = 1-0.7*smooth(local-2)
                            q[mask, 3:] = 1.2
                        elif local < 4:
                            q[mask, 1:3] = 0.3
                            q[mask, 3:] = 1.2
                            wheels[mask] = 5 if local < 3.5 else -7
                        elif local < 6:
                            q[mask, 1:3] = 0.3*(1-smooth((local-4)/2))
                            q[mask, 3:] = 1.2*(1-smooth((local-4)/2))
                action = fixed.clone()
                action[:, :5] = (q-offset)/scale
                action[:, 5:] = wheels[:, None]/25
            action_stats['samples'] += action.numel()
            action_stats['clipped'] += int((action.abs() >= 1).sum())
            if step % 5 == 0:
                traces.append({'t': step*base.step_dt,
                               'root': as_torch(robot.data.root_state_w)[:8].cpu().tolist(),
                               'joint_names': robot.joint_names,
                               'q': as_torch(robot.data.joint_pos)[:8].cpu().tolist(),
                               'action': action[:8].cpu().tolist()})
            obs, _, _, _ = wrapped.step(action)
        for row in base.petbot_episode_records:
            first.setdefault(row['env_id'], row)
        base.petbot_episode_records.clear()
        if len(first) == args.episodes:
            break
    if len(first) != args.episodes:
        raise RuntimeError(f'Incomplete first episodes: {len(first)}/{args.episodes}')
    rows = [first[i] for i in range(args.episodes)]
    if sorted(r['initial_index'] for r in rows) != list(range(args.episodes)):
        raise RuntimeError('Evaluation did not cover each initial state exactly once')
    summary = {}
    for label in ('all', 'upright', 'inverted', 'side_or_tilted'):
        group = rows if label == 'all' else [r for r in rows if r['initial_class'] == label]
        summary[label] = {'count': len(group),
                          'success_rate': sum(r['success'] for r in group)/len(group) if group else None,
                          'stable_at_end_rate': sum(r['stable_at_end'] for r in group)/len(group) if group else None}
    result = {'mode': args.mode, 'checkpoint': str(checkpoint) if checkpoint else None,
              'dataset_sha256': dataset_hash, 'evaluation': base.petbot_provenance,
              'summary': summary, 'episodes': rows, 'action_stats': action_stats, 'traces_first8': traces}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary), flush=True)
    env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    app.close()
