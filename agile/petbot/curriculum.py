"""Training-only open-loop recoverability labels; no action trajectories or evaluation states."""
import hashlib
import json
import math
from dataclasses import asdict

import torch

from .collection import physics_step, write_state
from .core import HoldTracker, as_torch, gravity_from_xyzw, success_components
from .model import POSITION_JOINTS, PROJECT_ROOT, load_model, task_implementation_fingerprint


@torch.no_grad()
def classify_easy_states(env, collection_metadata):
    from .mdp import state, invalid_state, out_of_bounds
    if collection_metadata['collection']['split'] != 'train':
        raise ValueError('Curriculum labels must never use evaluation states')
    dataset = env.petbot_dataset
    digest = hashlib.sha256()
    for name in ('root', 'joint_pos', 'joint_vel'):
        digest.update(dataset[name].contiguous().numpy().tobytes())
    metadata = {'dataset_sha256': digest.hexdigest(), 'collection': collection_metadata,
                'thresholds': asdict(env.cfg.success), 'implementation': task_implementation_fingerprint(),
                'horizon_s': env.cfg.episode_length_s, 'classifier': 'hold_default_fallen_v1'}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    path = PROJECT_ROOT / 'artifacts/fallen_states' / f'easy_{key}.pt'
    if path.exists():
        cached = torch.load(path, weights_only=True)
        if cached['metadata'] != metadata or not len(cached['indices']):
            raise ValueError('Invalid easy-state cache')
        print(f'[PetBot] easy curriculum cache: {len(cached["indices"])}/{len(dataset["root"])}', flush=True)
        return cached['indices'], {**metadata, 'count': len(cached['indices'])}
    robot = env.scene['robot']; n = env.num_envs
    spec = load_model()
    bounds = torch.tensor([spec['joints'][name]['range'] for name in POSITION_JOINTS], device=env.device)
    home = torch.tensor([spec['joints'][name]['default'] for name in POSITION_JOINTS], device=env.device)
    action = torch.zeros(n, 7, device=env.device)
    action[:, :5] = (home - bounds.mean(-1)) / ((bounds[:, 1] - bounds[:, 0]) / 2)
    successes = []
    for start in range(0, len(dataset['root']), n):
        indices = torch.arange(start, start+n).clamp_max(len(dataset['root'])-1)
        root = dataset['root'][indices].to(env.device).clone()
        root[:, :3] += as_torch(env.scene.env_origins)
        write_state(env, root, dataset['joint_pos'][indices].to(env.device), dataset['joint_vel'][indices].to(env.device))
        env.action_manager.reset()
        tracker = HoldTracker(n, env.step_dt, env.cfg.success.hold_s, env.device)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        for step in range(env.max_episode_length):
            env.action_manager.process_action(action)
            for _ in range(env.cfg.decimation):
                env.action_manager.apply_action()
                physics_step(env)
            s = state(env)
            parts = success_components(**s, target_height=env.petbot_standing_height, cfg=env.cfg.success)
            pose = as_torch(robot.data.root_link_pose_w)
            alive &= ~invalid_state(env) & ~out_of_bounds(env)
            tracker.update(torch.stack(list(parts.values())).all(0) & alive, torch.full((n,), (step+1)*env.step_dt, device=env.device))
        # Only actual fallen states are eligible; do not turn this into upright stance training.
        fallen = -gravity_from_xyzw(dataset['root'][indices, 3:7])[:, 2] < math.cos(env.cfg.success.tilt_rad)
        count = min(n, len(dataset['root'])-start)
        successes.append((tracker.ever.cpu() & fallen)[:count])
        print(f'[PetBot] easy-state classification {start+count}/{len(dataset["root"])}', flush=True)
    labels = torch.cat(successes)
    pool = torch.where(labels)[0]
    if not len(pool):
        raise RuntimeError('No open-loop recoverable fallen training states; stop before curriculum training')
    temporary = path.with_suffix('.tmp')
    torch.save({'metadata': metadata, 'indices': pool}, temporary)
    temporary.replace(path)
    print(f'[PetBot] easy curriculum: {len(pool)}/{len(labels)}', flush=True)
    return pool, {**metadata, 'count': len(pool)}
