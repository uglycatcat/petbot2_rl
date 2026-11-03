"""Container-only training/evaluation, with explicit provenance and failure artifacts."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
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
    defaults=json.loads((PROJECT_ROOT/'configs/train_v1.json').read_text())
    t=json.loads(Path(path).read_text())
    optional={'policy_distribution','learning_rate_schedule','reward_version','discount_gamma','beta_std_ceiling','critic_observation','task_variant',
              'auxiliary_precision_cost','terminate_auxiliary_flight','entropy_coef','auxiliary_objective','full_objective'}
    if set(t)-optional!=set(defaults) or t['schema']!='petbot_step_ppo_v1': raise ValueError('Unknown training schema/fields')
    if t.get('policy_distribution','gaussian') not in ('gaussian','beta','stop_gaussian','full_step_gaussian'):raise ValueError('Unknown policy distribution')
    if t.get('critic_observation', 'policy') not in ('policy', 'reward_state_v11'):
        raise ValueError('Unknown critic observation')
    if t.get('learning_rate_schedule','adaptive') not in ('adaptive','fixed'):raise ValueError('Unknown learning rate schedule')
    if 'beta_std_ceiling' in t:
        s=t['beta_std_ceiling']
        if (t.get('policy_distribution')!='beta' or not isinstance(s,dict)
                or set(s) not in ({'start_update','end_update','final_std'},
                                  {'start_update','end_update','final_std','wheel_final_std'})
                or type(s['start_update']) is not int or type(s['end_update']) is not int
                or not 0<=s['start_update']<s['end_update']<=t['iterations']
                or not isinstance(s['final_std'],(int,float)) or not .005<=s['final_std']<=.577):
            raise ValueError('Invalid Beta variance curriculum')
        if 'wheel_final_std' in s and (not isinstance(s['wheel_final_std'],(int,float))
                or not .005<=s['wheel_final_std']<=.577):
            raise ValueError('Invalid wheel variance ceiling')
    if t.get('reward_version','legacy') not in ('legacy','edge_v3','progress_v4','stage_v5','settle_v6','touch_v8','brake_v10','brake_v13','restart14_staged'):raise ValueError('Unknown reward version')
    if not 0<t.get('discount_gamma',.99)<=1:raise ValueError('Invalid discount gamma')
    continuing = t.get('auxiliary_objective') == 'continuous_stop'
    full_discounted = t.get('full_objective') == 'discounted_staged'
    if 'full_objective' in t and (not full_discounted or t.get('task_variant') != 'full'
            or t.get('reward_version') != 'restart14_staged' or t.get('discount_gamma') != .99
            or t.get('policy_distribution') != 'full_step_gaussian' or t.get('entropy_coef') != 0.
            or 'auxiliary_objective' in t):
        raise ValueError('Discounted full task requires its explicit independent configuration')
    if t.get('policy_distribution') == 'full_step_gaussian' and (not full_discounted
            or not .05 <= t['initial_std'] <= .5):
        raise ValueError('Full-step Gaussian requires discounted full task and finite climbing exploration')
    if 'auxiliary_objective' in t and (not continuing or t.get('task_variant') != 'platform_stop_aux'
            or t.get('reward_version') != 'restart14_staged' or not 0<t.get('discount_gamma',1)<1
            or not t.get('terminate_auxiliary_flight')):
        raise ValueError('Continuing STOP is an auxiliary-only discounted objective with real quality failures')
    if t.get('policy_distribution') == 'stop_gaussian' and (not continuing
            or not .005 <= t['initial_std'] <= .05 or t.get('entropy_coef') != 0.):
        raise ValueError('STOP-centered Gaussian is restricted to continuing auxiliary STOP with local noise and zero entropy bonus')
    if t.get('reward_version') in ('progress_v4','stage_v5','settle_v6','touch_v8','brake_v10','brake_v13','restart14_staged') and t.get('discount_gamma')!=1. and not (continuing or full_discounted):
        raise ValueError('Progress rewards require undiscounted finite-episode returns')
    if t.get('task_variant','full') not in ('full','platform_stop_aux'):
        raise ValueError('Unknown task variant')
    if t.get('task_variant') == 'platform_stop_aux' and t.get('reward_version') != 'restart14_staged':
        raise ValueError('Auxiliary stopping requires the explicit restarted objective')
    for name in ('auxiliary_precision_cost', 'terminate_auxiliary_flight'):
        if name in t and (type(t[name]) is not bool or t.get('task_variant') != 'platform_stop_aux'):
            raise ValueError('Parking diagnostic options must not silently change the full task')
    if 'entropy_coef' in t and (type(t['entropy_coef']) not in (float,int)
                              or not math.isfinite(t['entropy_coef']) or not 0<=t['entropy_coef']<=.1):
        raise ValueError('Invalid entropy coefficient')
    if t['initialization']!='random' or t['pretrained_models']: raise ValueError('Only random initialization is allowed')
    for name in ('num_envs','iterations','save_interval','decimation','num_steps_per_env','num_learning_epochs','num_mini_batches'):
        if type(t[name]) is not int or t[name]<=0: raise ValueError(f'Invalid {name}')
    if t['num_envs']*t['num_steps_per_env']%t['num_mini_batches']: raise ValueError('Batch must divide into mini batches')
    if type(t['seed']) is not int or t['seed']<0: raise ValueError('Invalid seed')
    for name in ('hip_kp','hip_kd','head_kp','head_kd','ear_kp','ear_kd','hip_target_rate_rad_s',
                 'wheel_target_accel_rad_s2','learning_rate','initial_std'):
        if not isinstance(t[name],(float,int)) or not math.isfinite(t[name]) or t[name]<=0: raise ValueError(f'Invalid {name}')
    if not t['hidden_dims'] or any(type(x) is not int or x<=0 for x in t['hidden_dims']): raise ValueError('Invalid hidden_dims')
    return t


def main(argv):
    if not Path('/.dockerenv').exists(): raise RuntimeError('Isaac/training must run inside the dedicated container')
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','evaluate'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--training-config',type=Path,default=PROJECT_ROOT/'configs/round14_seed70.json')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--asset-dir',type=Path,default=PROJECT_ROOT/'artifacts/robot')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=str,help='Basename within this run; never a foreign checkpoint')
    parser.add_argument('--stop-after',type=int,help='Stop after this many new PPO updates, preserving the total budget')
    parser.add_argument('--output',type=Path,help='Fresh evaluation output directory')
    parser.add_argument('--eval-envs',type=int,default=32)
    parser.add_argument('--hold-after-success',type=float,default=0.)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args(argv)
    from .ppo import write_json, state_digest, runner_config, StepRunner
    task=load_config(args.config,args.model_dir)
    if json.loads((args.asset_dir/'asset.json').read_text())['source_sha256']!=source_fingerprint(task.model_dir):
        raise ValueError('Derived asset does not match read-only source model')
    if args.hold_after_success < 0 or (args.command != 'evaluate' and args.hold_after_success): parser.error('Extra hold is evaluation-only')
    if args.command=='evaluate' and (args.resume or args.stop_after): parser.error('Evaluation cannot resume training')
    if args.command=='train' and args.checkpoint and not args.resume: parser.error('Fresh training cannot load checkpoints')
    if args.stop_after is not None and args.stop_after<=0: parser.error('stop-after must be positive')
    run=args.run_dir.resolve()
    if args.command=='train' and not args.resume:
        t=load_training(args.training_config)
        run.mkdir(parents=True,exist_ok=False)
        task_data=asdict(task); task_data.pop('model_dir')
        asset_digest=hashlib.sha256((args.asset_dir/'asset.json').read_bytes()+(args.asset_dir/'petbot.usd').read_bytes()).hexdigest()
        origin=dict(schema='step_random_run_v1',run_id=str(uuid.uuid4()),code_sha256=code_digest(),
            source_sha256=source_fingerprint(task.model_dir),asset_sha256=asset_digest,
            training=t,task=task_data,initialization='random',pretrained_models=[])
        write_json(run/'origin.json',origin)
    else:
        origin=json.loads((run/'origin.json').read_text()); t=origin['training']
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
        from .critic_state import CRITIC_STATE_FIELDS
        actual=t.copy()
        if args.command=='evaluate':
            actual['num_envs']=args.eval_envs
            actual['hold_after_success_s']=args.hold_after_success
            actual['_evaluation']=True
        if args.hold_after_success and not (actual.get('task_variant') == 'platform_stop_aux'
                or actual.get('full_objective') == 'discounted_staged'):
            raise ValueError('Extra hold currently supported only by isolated platform STOP diagnostic')
        env=StepEnv(task,actual,args.asset_dir,args.device)
        env.trace_ids=list(range(min(actual['num_envs'],8))) if args.command=='train' else list(range(actual['num_envs']))
        attempt=uuid.uuid4().hex[:12]
        trace_file=(output/(f'trace_resume_{attempt}.jsonl' if args.resume else 'trace.jsonl')).open('x')
        episode_file=(output/'episodes.jsonl').open('a')
        env.trace_writer=trace_file; env.episode_writer=episode_file
        env.attempt_id=attempt
        wrapped=RslRlVecEnvWrapper(env)
        runner=StepRunner(wrapped,runner_config(t),origin,str(run) if args.command=='train' else None,args.device)
        write_json(output/'layout.json',dict(joints=env.robot.joint_names,hips=env.hips,rear_wheels=env.wheels,
            trace_ids=env.trace_ids,step_dt=env.step_dt,source_code_sha256=origin['code_sha256'],
            actor_observation_dim=26, critic_observation_dim=48 if env.reward_state_critic else 26,
            critic_extra_fields=list(CRITIC_STATE_FIELDS) if env.reward_state_critic else []))
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
                    obs,_,_,_=wrapped.step(policy(obs))
                    for record in env.terminal_records:
                        records.setdefault(record['env'],record)
                    env.terminal_records.clear()
                    if len(records)==actual['num_envs']: break
            if len(records)!=actual['num_envs']: raise RuntimeError('Evaluation did not terminate every initial episode')
            write_json(output/'result.json',dict(status='complete',episodes=list(records.values()),
                successes=sum(r['success'] for r in records.values()),num_envs=actual['num_envs'],
                completed_updates=runner.current_learning_iteration,scene=t.get('task_variant','full'),hold_after_success_s=args.hold_after_success,
                independent_randomization=False))
            write_json(output/'status.json',dict(status='complete',command='evaluate'))
        trace_file.close();episode_file.close()
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
