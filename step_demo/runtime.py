"""Container-only training/evaluation, with explicit provenance and failure artifacts."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import gzip
import json
import math
from pathlib import Path
import sys
import traceback
import uuid

from .config import PROJECT_ROOT, DEFAULT_CONFIG, load_config
from .model import source_fingerprint


def code_digest():
    sha = hashlib.sha256()
    for folder in ('step_demo','tools','configs'):
        for path in sorted((PROJECT_ROOT/folder).rglob('*')):
            if path.suffix in ('.py','.json'):
                sha.update(str(path.relative_to(PROJECT_ROOT)).encode()); sha.update(path.read_bytes())
    return sha.hexdigest()


def load_training(path):
    t = json.loads(Path(path).read_text())
    expected = {'schema','seed','num_envs','iterations','save_interval','decimation',
        'hip_kp','hip_kd','head_kp','head_kd','ear_kp','ear_kd','hip_target_rate_rad_s',
        'wheel_target_accel_rad_s2','num_steps_per_env','hidden_dims','memory_dim','learning_rate',
        'num_learning_epochs','num_mini_batches','initialization','pretrained_models','entropy_coef',
        'initial_mean','initial_std'}
    if set(t) - {'trace_interval_steps','compiled_geometry'} != expected or t['schema'] != 'petbot_step_ppo_v2':
        raise ValueError('Only the refactored petbot_step_ppo_v2 schema is accepted; historical configs are not runnable here')
    if t['initialization'] != 'random' or t['pretrained_models']:
        raise ValueError('Only random initialization is allowed')
    for name in ('seed','num_envs','iterations','save_interval','decimation','num_steps_per_env',
                 'num_learning_epochs','num_mini_batches','memory_dim'):
        if type(t[name]) is not int or t[name] < (0 if name == 'seed' else 1):
            raise ValueError(f'Invalid {name}')
    if t['decimation'] != 10:
        raise ValueError('Control is fixed at 50Hz; task checks run at 500Hz')
    if t['num_envs'] % t['num_mini_batches']:
        raise ValueError('Recurrent minibatches must divide the environment count')
    # Rollout boundaries are PPO storage boundaries, not episode boundaries.
    # RSL-RL bootstraps nonterminal returns and carries actor GRU state forward.
    # The environment's physical hold timer continues until a real reset.
    if t['num_steps_per_env'] < 16:
        raise ValueError('Use at least 16 control steps per PPO rollout')
    t.setdefault('trace_interval_steps', 10)
    t.setdefault('compiled_geometry', False)
    if type(t['compiled_geometry']) is not bool:
        raise ValueError('compiled_geometry must be boolean')
    if type(t['trace_interval_steps']) is not int or t['trace_interval_steps'] < 1:
        raise ValueError('trace_interval_steps must be a positive integer')
    for name in ('hip_kp','hip_kd','head_kp','head_kd','ear_kp','ear_kd','hip_target_rate_rad_s',
                 'wheel_target_accel_rad_s2','learning_rate'):
        if type(t[name]) not in (float,int) or not math.isfinite(t[name]) or t[name] <= 0:
            raise ValueError(f'Invalid {name}')
    if not t['hidden_dims'] or any(type(x) is not int or x <= 0 for x in t['hidden_dims']):
        raise ValueError('Invalid hidden_dims')
    if type(t['entropy_coef']) not in (float,int) or not 0 <= t['entropy_coef'] <= .01:
        raise ValueError('Invalid entropy coefficient')
    for key in ('initial_mean','initial_std'):
        if not isinstance(t[key],list) or len(t[key]) != 2 or not all(type(x) in (float,int) and math.isfinite(x) for x in t[key]):
            raise ValueError(f'Invalid {key}')
    if not -1<t['initial_mean'][0]<1 or not -1.1<t['initial_mean'][1]<1 or not all(0<x<.3 for x in t['initial_std']):
        raise ValueError('Initial exploration outside bounded action support')
    return t


def main(argv):
    if not Path('/.dockerenv').exists(): raise RuntimeError('Isaac/training must run inside the dedicated container')
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','evaluate'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--training-config',type=Path,default=PROJECT_ROOT/'configs/train.json')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--asset-dir',type=Path,default=PROJECT_ROOT/'artifacts/robot')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=str,help='Basename within this run; never a foreign checkpoint')
    parser.add_argument('--stop-after',type=int,help='Stop after this many new PPO updates, preserving the total budget')
    parser.add_argument('--output',type=Path,help='Fresh evaluation output directory')
    parser.add_argument('--eval-envs',type=int,default=32)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args(argv)
    from .ppo import write_json, state_digest, runner_config, StepRunner
    task=load_config(args.config,args.model_dir)
    if json.loads((args.asset_dir/'asset.json').read_text())['source_sha256']!=source_fingerprint(task.model_dir):
        raise ValueError('Derived asset does not match read-only source model')
    if args.command=='evaluate' and (args.resume or args.stop_after): parser.error('Evaluation cannot resume training')
    if args.command=='train' and args.checkpoint and not args.resume: parser.error('Fresh training cannot load checkpoints')
    if args.stop_after is not None and args.stop_after<=0: parser.error('stop-after must be positive')
    run=args.run_dir.resolve()
    if args.command=='train' and not args.resume:
        t=load_training(args.training_config)
        run.mkdir(parents=True,exist_ok=False)
        task_data=asdict(task); task_data.pop('model_dir')
        asset_digest=hashlib.sha256((args.asset_dir/'asset.json').read_bytes()+(args.asset_dir/'petbot.usd').read_bytes()).hexdigest()
        origin=dict(schema='step_random_run_v2',run_id=str(uuid.uuid4()),code_sha256=code_digest(),
            source_sha256=source_fingerprint(task.model_dir),asset_sha256=asset_digest,
            training=t,task=task_data,initialization='random',pretrained_models=[])
        write_json(run/'origin.json',origin)
    else:
        origin=json.loads((run/'origin.json').read_text())
        if origin.get('schema') != 'step_random_run_v2': raise ValueError('Historical runs require their archived fixed source')
        t=origin['training']
        task_data=asdict(task);task_data.pop('model_dir')
        if origin['code_sha256']!=code_digest() or origin['source_sha256']!=source_fingerprint(task.model_dir) or origin['task']!=task_data:
            raise ValueError('Code/model/task differs from the run; use its fixed source copy')
        asset_digest=hashlib.sha256((args.asset_dir/'asset.json').read_bytes()+(args.asset_dir/'petbot.usd').read_bytes()).hexdigest()
        if asset_digest!=origin['asset_sha256']: raise ValueError('Derived asset differs from run')
    lock=(run/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=run if args.command=='train' else args.output
    if output is None: parser.error('Evaluation requires --output')
    if args.command=='evaluate': output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='starting',command=args.command))
    app=None; env=None
    try:
        app=AppLauncher(args).app
        import torch
        import importlib.metadata
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from .isaac_env import StepEnv
        actual=t.copy()
        if args.command=='evaluate':
            actual['num_envs']=args.eval_envs
        env=StepEnv(task,actual,args.asset_dir,args.device)
        env.trace_interval_steps=t['trace_interval_steps'] if args.command=='train' else 1
        env.trace_ids=list(range(min(actual['num_envs'],8))) if args.command=='train' else list(range(actual['num_envs']))
        attempt=uuid.uuid4().hex[:12]
        trace_file=(output/(f'trace_resume_{attempt}.jsonl' if args.resume else 'trace.jsonl')).open('x')
        episode_file=(output/'episodes.jsonl').open('a')
        env.trace_writer=trace_file; env.episode_writer=episode_file
        env.attempt_id=attempt
        if args.command == 'evaluate':
            env.physics_trace_writer=gzip.open(output/'physics.jsonl.gz','xt')
        wrapped=RslRlVecEnvWrapper(env)
        runner=StepRunner(wrapped,runner_config(t),origin,str(run) if args.command=='train' else None,args.device)
        write_json(output/'layout.json',dict(joints=env.robot.joint_names,hips=env.hips,rear_wheels=env.wheels,
            trace_ids=env.trace_ids,step_dt=env.step_dt,source_code_sha256=origin['code_sha256'],
            actor_observation_dim=27, actor_memory='GRU', critic_observation_dim=env.critic_dim,
            critic_memory_fields=list(env.objective._memory_names), observation_normalization='fixed_units'))
        write_json(output/'runtime.json',{name:importlib.metadata.version(name) for name in ('torch','rsl-rl-lib','numpy')})
        if args.resume or args.command=='evaluate':
            basename=args.checkpoint or json.loads((run/'latest.json').read_text())['checkpoint']
            if Path(basename).name!=basename or not basename.endswith('.pt'): raise ValueError('Checkpoint must be a basename within this run')
            restored=runner.restore(run/basename,training=args.command=='train')
            write_json(output/f'restored_{attempt}.json',dict(checkpoint=basename,**restored))
        else:
            assert not runner.alg.optimizer.state
            assert all(p.requires_grad for p in runner.alg.actor.parameters())
            initial=runner.alg.save();initial['origin']=origin
            torch.save(initial,run/'initial.pt')
            write_json(run/'initialization.json',dict(initialization='random',pretrained_models=[],
                actor_sha256=state_digest(initial['actor_state_dict']),
                critic_sha256=state_digest(initial['critic_state_dict']),optimizer_states=0,frozen_parameters=0))
        if args.command=='train':
            remaining=t['iterations']-runner.current_learning_iteration
            if remaining<=0: raise ValueError('This run has exhausted its update budget')
            count=min(remaining,args.stop_after or remaining)
            write_json(output/'status.json',dict(status='running',completed_updates=runner.current_learning_iteration,
                total_updates=t['iterations'],attempt=attempt))
            runner.learn(count,init_at_random_ep_len=False)
            completed=runner.current_learning_iteration+1
            write_json(output/'status.json',dict(status='complete' if completed==t['iterations'] else 'interrupted',
                completed_updates=completed,total_updates=t['iterations'],totals=env.totals,
                captured_substeps=env.captured_substeps))
        else:
            policy=runner.get_inference_policy(args.device); obs=wrapped.get_observations()
            records={}
            with torch.inference_mode():
                for step in range(env.max_episode_length+1):
                    obs,_,dones,_=wrapped.step(policy(obs))
                    runner.alg.actor.reset(dones)
                    for record in env.terminal_records:
                        records.setdefault(record['env'],record)
                    env.terminal_records.clear()
                    if len(records)==actual['num_envs']: break
            if len(records)!=actual['num_envs']: raise RuntimeError('Evaluation did not terminate every initial episode')
            write_json(output/'result.json',dict(status='complete',episodes=list(records.values()),
                successes=sum(r['success'] for r in records.values()),num_envs=actual['num_envs'],
                completed_updates=runner.current_learning_iteration,scene='full',hold_after_success_s=env.objective.extra_hold_s,
                independent_randomization=False))
            write_json(output/'status.json',dict(status='complete',command='evaluate'))
        trace_file.close();episode_file.close()
        if env.physics_trace_writer is not None:
            env.physics_trace_writer.close(); env.physics_trace_writer=None
    except BaseException:
        write_json(output/'status.json',dict(status='failed',error=traceback.format_exc()))
        raise
    finally:
        try:
            if env is not None: env.close()
        finally:
            # Kit may exit during app.close; explicitly unlock before shutdown so
            # a forked native worker cannot retain this run's flock indefinitely.
            fcntl.flock(lock,fcntl.LOCK_UN)
            lock.close()
            if app is not None: app.close()
    return 0
