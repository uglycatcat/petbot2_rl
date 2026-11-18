"""Project entry point. Isaac Lab launcher must run before importing the environment."""
import argparse
from pathlib import Path
import json
import os
import time
import signal
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/step7cm.json')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--num-envs',type=int)
parser.add_argument('--iterations',type=int)
parser.add_argument('--seed',type=int)
parser.add_argument('--step-height',type=float)
parser.add_argument('--control-dt',type=float)
parser.add_argument('--asset-dir',type=Path,default=Path(__file__).parent/'assets')
parser.add_argument('--mode',choices=['train','evaluate','probe'],default='train')
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--resume',type=Path)
parser.add_argument('--stop-after',type=int,help='Engineering interruption after this many updates; original target stays unchanged')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
config=json.loads(args.config.read_text())
for k in ('num_envs','iterations','seed','step_height','control_dt'):
    if getattr(args,k) is not None:config[k]=getattr(args,k)
if not 0<config.get('step_height',0.05)<1:parser.error('step_height must be between0 and1 metre')
(args.output/'config.json').write_text(json.dumps(config,indent=2))
app=AppLauncher(args).app
stop_requested=False
class InterruptedRun(Exception): pass
def request_stop(signum,frame):
    global stop_requested
    stop_requested=True
signal.signal(signal.SIGUSR1,request_stop)
signal.signal(signal.SIGTERM,request_stop)
import torch
import numpy as np
from step_explore.env import ExploreEnv, ExploreCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

def write_json(name,value):
    p=args.output/name
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def runner_config():
    return {'seed':config['seed'],'device':args.device,'num_steps_per_env':config['num_steps_per_env'],
        'save_interval':100,'logger':'tensorboard','experiment_name':'step_explore',
        'obs_groups':{'policy':['policy'],'critic':['policy']},
        'policy':{'class_name':'ActorCritic','init_noise_std':config['initial_std'],'noise_std_type':'log',
            'actor_obs_normalization':True,'critic_obs_normalization':True,
            'actor_hidden_dims':[256,128,64],'critic_hidden_dims':[256,128,64],'activation':'elu'},
        'algorithm':{'class_name':'PPO','value_loss_coef':1.,'use_clipped_value_loss':True,
            'clip_param':.2,'entropy_coef':config['entropy_coef'],'num_learning_epochs':4,'num_mini_batches':8,
            'learning_rate':config['learning_rate'],'schedule':'adaptive','gamma':.995,'lam':.95,
            'desired_kl':.01,'max_grad_norm':1.}}

started=time.time()
try:
    cfg=ExploreCfg()
    cfg.seed=config['seed'];cfg.sim.device=args.device
    # Legacy5cm configurations remain reproducible; new runs record their height explicitly.
    cfg.height=config.get('step_height',0.05)
    cfg.top_reward_mode=config.get('top_reward_mode','distinct_wheels')
    control_dt=config.get('control_dt',0.02)
    cfg.decimation=round(control_dt/cfg.sim.dt)
    if cfg.decimation<1 or abs(cfg.decimation*cfg.sim.dt-control_dt)>1e-9:
        raise ValueError('control_dt must be a positive multiple of the2ms physics step')
    cfg.sim.render_interval=cfg.decimation
    cfg.scene.num_envs=config['num_envs'];cfg.episode_length_s=config['episode_seconds']
    cfg.asset_dir=str(args.asset_dir.resolve());cfg.exploration_bonus=config['exploration_bonus']
    env=ExploreEnv(cfg)
    wrapped=RslRlVecEnvWrapper(env)
    if args.mode=='probe':
        from step_explore.probe import run_probe
        run_probe(env,wrapped,args.output)
    else:
        class Runner(OnPolicyRunner):
            def save(self,path,infos=None):
                p=Path(path);tmp=p.with_suffix('.pending')
                super().save(str(tmp),infos={'config':config,'completed_updates':self.current_learning_iteration+1})
                tmp.replace(p)
            def log(self,locs,*a,**kw):
                super().log(locs,*a,**kw)
                write_json('status.json',{'status':'training','iteration':int(locs['it'])+1,
                    'target_iterations':config['iterations'],'elapsed_seconds':time.time()-started,
                    'episodes':env.total_episodes,'successes':env.total_success,'max_simultaneous_wheels_on_top':env.max_top,'max_distinct_wheels_reached':env.max_distinct_top,
                    'torch_peak_mib':torch.cuda.max_memory_reserved()/2**20,'updated_at':time.time()})
                if stop_requested:
                    self.save(str(args.output/'interrupted.pt'))
                    raise InterruptedRun()
        runner=Runner(wrapped,runner_config(),log_dir=str(args.output/'tensorboard'),device=args.device)
        if args.mode=='train':
            if args.checkpoint:
                raise ValueError('New training cannot load a policy; use --resume only for the same interrupted run')
            if args.resume:
                saved=torch.load(args.resume,map_location='cpu',weights_only=False)
                if saved['infos']['config']!=config:raise ValueError('Resume configuration differs from interrupted run')
                if args.resume.resolve().parent!=args.output.resolve() and args.resume.resolve().parent.parent!=args.output.resolve():
                    raise ValueError('Recovery must use a checkpoint from the same run directory')
                runner.load(str(args.resume))
                runner.current_learning_iteration=saved['infos']['completed_updates']
            remaining=config['iterations']-runner.current_learning_iteration
            runner.learn(num_learning_iterations=min(remaining,args.stop_after) if args.stop_after else remaining,
                         init_at_random_ep_len=False)
            if runner.writer is not None: runner.writer.flush()
            if args.stop_after and args.stop_after<remaining:
                runner.save(str(args.output/'interrupted.pt'))
                write_json('status.json',{'status':'engineering_interrupted','completed_updates':runner.current_learning_iteration+1,
                    'target_iterations':config['iterations'],'updated_at':time.time()})
                env.close();sys.stdout.flush();app.close(wait_for_replicator=False,skip_cleanup=True);sys.exit(0)
            runner.save(str(args.output/'final.pt'))
            write_json('status.json',{'status':'trained','iterations':config['iterations'],
                'elapsed_seconds':time.time()-started,'episodes':env.total_episodes,'successes':env.total_success,
                'max_simultaneous_wheels_on_top':env.max_top,'max_distinct_wheels_reached':env.max_distinct_top,'torch_peak_mib':torch.cuda.max_memory_reserved()/2**20,'updated_at':time.time()})
        else:
            runner.load(str(args.checkpoint),load_optimizer=False)
        if args.mode=='evaluate':
            from step_explore.evaluate import evaluate
            evaluate(env,wrapped,runner.get_inference_policy(device=args.device),args.output,
                     config['eval_episodes'])
    env.close()
except InterruptedRun:
    if runner.writer is not None: runner.writer.flush()
    write_json('status.json',{'status':'interrupted','completed_updates':runner.current_learning_iteration+1,
        'target_iterations':config['iterations'],'checkpoint':'interrupted.pt','updated_at':time.time()})
    env.close()
except BaseException as exc:
    import traceback
    traceback.print_exc()
    if 'env' in globals(): env.close()
    sys.stdout.flush();sys.stderr.flush()
    write_json('error.json',{'error':repr(exc),'elapsed_seconds':time.time()-started})
    raise
finally:
    sys.stdout.flush();sys.stderr.flush()
    app.close(wait_for_replicator=False,skip_cleanup=True)
