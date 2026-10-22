"""Versioned full upright cycle; independent observation and evaluation contract."""
import argparse,copy,hashlib,json,pathlib,subprocess
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--seed',type=int,default=42);p.add_argument('--log_dir',default='logs/cycle')
p.add_argument('--speed-limit',type=float,default=.1);p.add_argument('--yaw-limit',type=float,default=.3)
p.add_argument('--frame-min',type=float,default=-.3);p.add_argument('--frame-max',type=float,default=.3)
p.add_argument('--wheel-slew',type=float,choices=[120.,240.],default=120.)
p.add_argument('--frame-slew',type=float,default=3.)
p.add_argument('--learning-rate',type=float,default=.0001);p.add_argument('--save-interval',type=int,default=100)
p.add_argument('--initialize-from');p.add_argument('--checkpoint');p.add_argument('--evaluate',action='store_true')
p.add_argument('--allow-dirty',action='store_true');p.add_argument('--probe',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from upright.cycle_task import CycleEnvCfg,SPEC,ensure
from upright.transfer import expand_actor

class Adapter:
    def __init__(self,env):
        self.env=env;self.cfg=env.cfg;self.device=env.device;self.num_envs=env.num_envs
        self.num_actions=3;self.max_episode_length=env.max_episode_length;env.reset()
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,v):self.env.episode_length_buf=v
    def get_observations(self):return TensorDict(self.env.observation_manager.compute(),batch_size=[self.num_envs])
    def step(self,a):
        obs,r,term,timeout,extra=self.env.step(a)
        extra['time_outs']=timeout & ~term
        return TensorDict(obs,batch_size=[self.num_envs]),r,(term|timeout).long(),extra

try:
    assert 0<args.speed_limit<=.3 and 0<args.yaw_limit<=1
    assert -1<=args.frame_min<=args.frame_max<=1.2 and 0<args.frame_slew<=6
    assert not (args.checkpoint and args.initialize_from)
    cfg=CycleEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    cfg.speed_limit=args.speed_limit;cfg.yaw_limit=args.yaw_limit;cfg.frame_range=(args.frame_min,args.frame_max)
    cfg.wheel_slew=args.wheel_slew;cfg.frame_slew=args.frame_slew;cfg.wheel_mode='common_diff'
    cfg.feedback_mode='pose_feedback_tight'
    torch.manual_seed(args.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty:raise RuntimeError('Commit source before formal execution')
    names=['train_cycle.py']+['upright/'+n+'.py' for n in ['cycle','cycle_task','moving','task','distribution','model','asset_cfg','prepare','transfer']]
    fingerprint=hashlib.sha256(''.join(hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()
    contract=dict(task='upright_cycle_v1',observation_schema='cycle_49x5_v1',fingerprint=fingerprint,
        action_dim=3,wheel_mode='common_diff',wheel_slew=args.wheel_slew,frame_slew=args.frame_slew,
        control_dt=env.step_dt,source_sha256=SPEC['source_sha256'],rear_duty_threshold=.9,
        transition_reference='reanchor_once_on_balance',mode_request='landing_latched_until_handoff')
    if args.checkpoint:
        prior=json.loads((pathlib.Path(args.checkpoint).parent/'config.json').read_text())
        if prior['contract']!=contract:raise ValueError('Checkpoint contract mismatch')
    if args.evaluate and not args.checkpoint:raise ValueError('Evaluation requires a cycle checkpoint')
    lo,hi=SPEC['joints']['left_frame']['range'];home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=args.save_interval,logger='tensorboard',experiment_name='upright_cycle',run_name=f'seed{args.seed}',
        obs_groups={'actor':['policy'],'critic':['policy']},
        actor=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.)),
        critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
        algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=args.learning_rate,schedule='fixed',gamma=.997**3,lam=.95**3,desired_kl=.01,max_grad_norm=1.))
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    metadata=dict(contract=contract,args=vars(args),runner=config,working_tree_dirty=dirty,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if args.initialize_from:
        initial=pathlib.Path(args.initialize_from);prior=json.loads((initial.parent/'config.json').read_text())
        if prior['commit']!='5bb0c71a28c9b2b9733926d9768d7ae44625a224' or prior['working_tree_dirty']:raise ValueError('Only clean rebuilt batch2 actor allowed')
        for k,v in dict(source_sha256=SPEC['source_sha256'],observation_schema='moving_41x5_raw_support_v1',action_dim=3,wheel_mode='common_diff',control_dt=env.step_dt).items():
            if prior['contract'].get(k)!=v:raise ValueError('Source mismatch '+k)
        loaded=torch.load(initial,map_location='cpu',weights_only=False)
        actor=runner.alg.get_policy();actor.load_state_dict(expand_actor(loaded['actor_state_dict'],actor.state_dict()),strict=True)
        assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state
        metadata['initialization']=dict(source_commit=prior['commit'],checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),actor_only=True,critic_fresh=True,optimizer_loaded=False,extra_features_zero_weight=True)
        del loaded
    if args.checkpoint:runner.load(args.checkpoint)
    (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    if not args.evaluate and not args.probe:runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    else:
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations();n=args.num_envs
        alive=torch.ones(n,device=args.device,dtype=torch.bool)
        flags={k:torch.zeros_like(alive) for k in ['takeover','balance','landing','handoff','complete','qualified','failure']}
        trace=[];max_frame=0.;max_wheel=0.;peak_force=torch.zeros(n,device=args.device);track_sum=torch.zeros(n,2,device=args.device);track_count=torch.zeros(n,device=args.device)
        balance_good=torch.zeros(n,device=args.device);balance_count=torch.zeros(n,device=args.device)
        offsets=torch.zeros(n,device=args.device);first_balance=torch.full((n,),-1.,device=args.device)
        with torch.inference_mode():
            for tick in range(env.max_episode_length+4):
                a=env.action_manager.get_term('moving');previous=a.applied.clone()
                action=policy(obs)
                obs,r,done,extra=wrapped.step(action)
                s,m,raw,command,applied,since=env.moving_snapshot;valid=alive.clone()
                for key,event in [('takeover',m['switched']),('balance',m['entered_balance']),('landing',m['entered_landing']),('handoff',m['entered_handoff']),('complete',m['completed']),('qualified',m['qualified']),('failure',m['failure'])]:flags[key] |= valid & event
                entering=valid&m['entered_balance'];first_balance[entering]=since[entering]
                # Measured pre-reset snapshot, never post-reset progress buffers.
                offsets[entering]=m['transition_offset'][entering]
                b=valid&(m['phase']==2)&(m['phase_time']>.5)
                track_sum+=m['raw_error'].abs()*b[:,None];track_count+=b
                balance_good+=b&m['balanced'];balance_count+=b
                peak_force=torch.maximum(peak_force,torch.where(valid&(m['phase']==3),s['front_force_peak'],0.))
                delta=(applied-previous).abs();max_frame=max(max_frame,float(delta[valid,0].max())*(hi-lo)/2 if valid.any() else 0.)
                max_wheel=max(max_wheel,float(delta[valid,1:].max())*a.max_speed if valid.any() else 0.)
                trace.append(dict(t=(tick+1)*env.step_dt,valid=valid.tolist(),phase=m['phase'].tolist(),phase_time=m['phase_time'].tolist(),since=since.tolist(),raw=raw.tolist(),command=command.tolist(),pos=s['pos'].tolist(),forward=s['forward'].tolist(),heading_rate=s['heading_rate'].tolist(),front_z=s['front_z'].tolist(),front_vertical_speed=s['front_vertical_speed'].tolist(),rear_fraction=s['rear_fraction'].tolist(),four_fraction=s['four_fraction'].tolist(),applied=applied.tolist(),failure=m['failure'].tolist(),balanced=m['balanced'].tolist()))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
                alive &= ~done.bool()
                if not alive.any():break
        result=dict(count=n,**{k:int(v.sum()) for k,v in flags.items()},outcomes={k:v.tolist() for k,v in flags.items()},unfinished=int(alive.sum()),contract=contract,
            first_balance_seconds=first_balance.tolist(),transition_offset_m=offsets.tolist(),balanced_fraction=(balance_good/balance_count.clamp_min(1)).tolist(),balance_samples=balance_count.tolist(),tracking_mae=(track_sum/track_count.clamp_min(1)[:,None]).tolist(),landing_peak_front_force=peak_force.tolist(),max_frame_target_step=max_frame,max_wheel_target_step=max_wheel)
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'trace.json').write_text(json.dumps(trace))
        assert max_frame<=args.frame_slew*env.step_dt+1e-5 and max_wheel<=args.wheel_slew*env.step_dt+1e-4
        assert not alive.any()
        print({k:v for k,v in result.items() if isinstance(v,(int,float))})
    env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
