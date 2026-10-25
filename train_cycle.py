"""Versioned full upright cycle; independent observation and evaluation contract."""
import argparse,copy,hashlib,json,pathlib,subprocess
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--seed',type=int,default=42);p.add_argument('--log_dir',default='logs/cycle')
p.add_argument('--speed-limit',type=float,default=.1);p.add_argument('--yaw-limit',type=float,default=.3)
p.add_argument('--frame-min',type=float,default=-.3);p.add_argument('--frame-max',type=float,default=.3)
p.add_argument('--wheel-slew',type=float,choices=[120.,240.,480.],default=120.)
p.add_argument('--frame-slew',type=float,default=3.)
p.add_argument('--learning-rate',type=float,default=.0001);p.add_argument('--save-interval',type=int,default=100)
p.add_argument('--exploration-profile',choices=['moderate','fine','precision'],default='fine')
p.add_argument('--yaw-filter-tau',type=float,choices=[0.,.12],default=0.)
p.add_argument('--tracking-shape',choices=['gaussian','cauchy'],default='gaussian')
p.add_argument('--tracking-weight',type=float,choices=[6.,12.],default=12.)
p.add_argument('--instant-tracking-weight',type=float,choices=[.3,.7],default=.3)
p.add_argument('--tracking-objective',choices=['mixed','instant_tolerance'],default='mixed')
p.add_argument('--balance-noise-factor',type=float,choices=[1.,16.],default=1.)
p.add_argument('--actor-normalization',choices=['inherited','rebase_recent'],default='inherited')
p.add_argument('--yaw-feedback-design',choices=['legacy60','fast20','fast2'],default='legacy60')
p.add_argument('--steering-design',choices=['feedback'],default='feedback')
p.add_argument('--hold-min',type=float,default=8.);p.add_argument('--hold-max',type=float,default=12.)
p.add_argument('--initialize-cycle-from');p.add_argument('--initialize-from');p.add_argument('--checkpoint');p.add_argument('--evaluate',action='store_true')
p.add_argument('--allow-dirty',action='store_true');p.add_argument('--probe',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from upright.feedback_task import FeedbackEnvCfg as CycleEnvCfg,SPEC,ensure
from upright.feedback import CONTROLLER
from upright.fast_feedback import FAST_CONTROLLER

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
    assert sum(bool(x) for x in (args.checkpoint,args.initialize_from,args.initialize_cycle_from))<=1
    assert 2.<=args.hold_min<=args.hold_max<=30.
    if args.initialize_from:raise ValueError('V14 requires explicit cycle-v13 actor transfer')
    assert args.actor_normalization=='inherited' and args.exploration_profile=='fine' and args.yaw_feedback_design in ('fast20','fast2')
    cfg=CycleEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    cfg.speed_limit=args.speed_limit;cfg.yaw_limit=args.yaw_limit;cfg.frame_range=(args.frame_min,args.frame_max)
    cfg.wheel_slew=args.wheel_slew;cfg.frame_slew=args.frame_slew;cfg.wheel_mode='common_diff'
    cfg.feedback_mode='pose_feedback_tight';cfg.rewards.tracking.weight=args.tracking_weight;cfg.tracking_shape=args.tracking_shape;cfg.yaw_filter_tau=args.yaw_filter_tau
    cfg.steering_design=args.steering_design;cfg.hold_range=(args.hold_min,args.hold_max)
    cfg.tracking_objective=args.tracking_objective
    cfg.instant_tracking_weight=args.instant_tracking_weight;cfg.yaw_feedback_design=args.yaw_feedback_design
    cfg.active_seconds=args.hold_max+12.;cfg.episode_length_s=cfg.active_seconds+8.
    torch.manual_seed(args.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty:raise RuntimeError('Commit source before formal execution')
    names=['train_cycle.py','upright/balance_adapter.py']+['upright/'+n+'.py' for n in ['normalization','fast_feedback','ground_motion','feedback','feedback_task','continuous','continuous_task','cycle','cycle_task','moving','task','distribution','model','asset_cfg','prepare','transfer']]
    fingerprint=hashlib.sha256(''.join(hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()
    scales={'moderate':[1.,4.,4.],'fine':[4.,16.,16.],'precision':[16.,64.,64.]}[args.exploration_profile]
    contract=dict(task='upright_cycle_v14',tracking_objective=args.tracking_objective,actor_design='frozen_base_balance_yaw_adapter_v1',balance_noise_factor=args.balance_noise_factor,adapter_limits=[.03,.08,.8],adapter_gate=[1.2,1.8],adapter_observation_stats='frozen_parent_critic',actor_normalization=args.actor_normalization,normalization_prior_count=10000000 if args.actor_normalization=='rebase_recent' else None,yaw_feedback_design=args.yaw_feedback_design,fast_yaw_controller=dict(FAST_CONTROLLER,period=.002 if args.yaw_feedback_design=='fast2' else .02),heading_reference='wheel_lateral',contact_sampling='eager_physics_step_history30_v1',instant_tracking_weight=args.instant_tracking_weight,yaw_controller=CONTROLLER,steering_design=args.steering_design,yaw_filter_tau=args.yaw_filter_tau,tracking_shape=args.tracking_shape,command_sampling='mixed_six_sequence_v2',reward_design=('instant_tolerance_soft_landing_v1' if args.tracking_objective=='instant_tolerance' else 'joint_ground_motion_soft_landing_v2'),motion_feedback='pose_delta_ema_120ms',concentration_scale=scales,tracking_weight=args.tracking_weight,observation_schema='cycle_58x5_wheel_lateral_v2',fingerprint=fingerprint,
        action_dim=3,wheel_mode='common_diff',wheel_slew=args.wheel_slew,frame_slew=args.frame_slew,
        control_dt=env.step_dt,source_sha256=SPEC['source_sha256'],rear_duty_threshold=.9,
        transition_reference='reanchor_once_on_balance',mode_request='landing_latched_until_handoff',landing_force_limit_n=20.,touchdown_speed_limit=.35)
    if args.checkpoint:
        prior=json.loads((pathlib.Path(args.checkpoint).parent/'config.json').read_text())
        if prior['contract']!=contract:raise ValueError('Checkpoint contract mismatch')
    if args.evaluate and not args.checkpoint:raise ValueError('Evaluation requires a cycle checkpoint')
    lo,hi=SPEC['joints']['left_frame']['range'];home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=args.save_interval,logger='tensorboard',experiment_name='upright_cycle',run_name=f'seed{args.seed}',
        obs_groups={'actor':['policy'],'critic':['policy']},
        actor=dict(class_name='upright.balance_adapter:BalanceAdapterModel',balance_noise_factor=args.balance_noise_factor,hidden_dims=[128,128],activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.,concentration_scale=scales)),
        critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
        algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=args.learning_rate,schedule='fixed',gamma=.997**3,lam=.95**3,desired_kl=.01,max_grad_norm=1.))
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    metadata=dict(contract=contract,args=vars(args),runner=config,working_tree_dirty=dirty,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if args.initialize_cycle_from:
        initial=pathlib.Path(args.initialize_cycle_from);prior=json.loads((initial.parent/'config.json').read_text())
        if prior['contract'].get('task')!='upright_cycle_v13' or prior['working_tree_dirty']:raise ValueError('Only clean cycle-v13 source allowed')
        if prior['commit']!='2b41c8fdceeb1afbb52cb9d84483ac4dcd89284f':raise ValueError('Only validated cycle-v13 source allowed')
        if prior['contract']['observation_schema']!='cycle_58x5_wheel_lateral_v2' or prior['contract']['steering_design']!='feedback':raise ValueError('Source observation schema mismatch')
        for k in ('action_dim','wheel_mode','wheel_slew','frame_slew','control_dt','source_sha256','concentration_scale','instant_tracking_weight','tracking_weight','balance_noise_factor'):
            if prior['contract'][k]!=contract[k]:raise ValueError('Transfer mismatch '+k)
        if prior['contract']['balance_noise_factor']!=16. or prior['contract']['tracking_objective']!='instant_tolerance' or prior['contract']['yaw_feedback_design']!='fast20':raise ValueError('V14 uses cycle13 B parents')
        if args.tracking_objective!='instant_tolerance':raise ValueError('V14 retains the cycle13 B reward')
        loaded=torch.load(initial,map_location='cpu',weights_only=False)
        actor=runner.alg.get_policy()
        actor.load_state_dict(loaded['actor_state_dict'],strict=True)
        runner.alg.critic.load_state_dict(loaded['critic_state_dict'],strict=True)
        assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state
        metadata['initialization']=dict(source_commit=prior['commit'],checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),actor_base_frozen=True,critic_loaded=True,optimizer_loaded=False,prior_adapter_loaded=True,yaw_adapter_loaded=True,reward_changed=False,feedback_period_changed=args.yaw_feedback_design=='fast2',observation_width=58)
        del loaded
    if args.checkpoint:runner.load(args.checkpoint)
    (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    if not args.evaluate and not args.probe:runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    else:
        from upright.balance_adapter import balance_gate
        from upright.distribution import BoundedBetaMean
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations();n=args.num_envs
        alive=torch.ones(n,device=args.device,dtype=torch.bool)
        flags={k:torch.zeros_like(alive) for k in ['takeover','balance','landing','handoff','complete','qualified','failure']}
        coverage={stage:{name:torch.zeros(n,device=args.device,dtype=torch.bool) for name in ['forward','reverse','left','right','curve','stop']} for stage in ['active','balance']}
        trace=[];max_frame=0.;max_wheel=0.;peak_force=torch.zeros(n,device=args.device);peak_touch_speed=peak_force.clone();track_sum=torch.zeros(n,2,device=args.device);track_count=torch.zeros(n,device=args.device)
        balance_good=torch.zeros(n,device=args.device);balance_count=torch.zeros(n,device=args.device)
        offsets=torch.zeros(n,device=args.device);first_balance=torch.full((n,),-1.,device=args.device)
        takeover_motion=torch.zeros(n,2,device=args.device);last_category=torch.full((n,),-1,device=args.device,dtype=torch.long)
        switches=torch.zeros(n,device=args.device,dtype=torch.long);transition_matrix=torch.zeros(6,6,device=args.device,dtype=torch.long)
        balance_duration=torch.zeros(n,device=args.device)
        legacy_error_sum=torch.zeros(n,2,device=args.device);legacy_good=torch.zeros(n,device=args.device)
        wheel_axis_error=0.
        from upright.ground_motion import lateral_axis
        from upright.cycle_task import tensor
        rear_body_ids=env.scene['robot'].find_bodies(['left_rear_wheel','right_rear_wheel'],preserve_order=True)[0]
        with torch.inference_mode():
            for tick in range(env.max_episode_length+4):
                a=env.action_manager.get_term('moving');previous=a.applied.clone()
                decision_raw=ensure(env).raw.clone()
                action=policy(obs)
                actor=runner.alg.get_policy();gate=balance_gate(obs['policy'])
                progress=ensure(env)
                expected_gate=(progress.phase==2)*((progress.phase_time-1.2)/.6).clamp(0.,1.)
                assert torch.allclose(gate[alive],expected_gate[alive],atol=1e-6), 'Adapter phase/history mismatch'
                base_action=BoundedBetaMean(127.)(actor.mlp(actor.obs_normalizer(obs['policy'])))
                correction=action-base_action
                assert (correction.abs()<=correction.new_tensor([.03,.08,.8])+1e-6).all()
                assert not correction[gate==0].any()
                obs,r,done,extra=wrapped.step(action)
                s,m,raw,command,applied,since=env.moving_snapshot;valid=alive.clone()
                assert torch.equal(raw[valid],decision_raw[valid]), 'Command changed after policy decision'
                assert torch.equal(m['failure'],m['failure_code']!=0), 'Incomplete failure cause classification'
                for key,event in [('takeover',m['switched']),('balance',m['entered_balance']),('landing',m['entered_landing']),('handoff',m['entered_handoff']),('complete',m['completed']),('qualified',m['qualified']),('failure',m['failure'])]:flags[key] |= valid & event
                rv,rw=raw.unbind(-1)
                categories=dict(forward=(rv>0)&(rw==0),reverse=(rv<0)&(rw==0),left=(rv==0)&(rw>0),right=(rv==0)&(rw<0),curve=(rv!=0)&(rw!=0),stop=(rv==0)&(rw==0))
                category=torch.full((n,),-1,device=args.device,dtype=torch.long)
                for index,mask in enumerate(categories.values()):category[mask]=index
                in_balance=valid&(m['phase']==2);changed=in_balance&(last_category>=0)&(category!=last_category)
                switches+=changed
                if changed.any():transition_matrix+=torch.bincount(last_category[changed]*6+category[changed],minlength=36).reshape(6,6)
                last_category[in_balance]=category[in_balance];balance_duration+=in_balance*env.step_dt
                switched=valid&m['switched'];takeover_motion[switched]=m['motion_feedback'][switched]
                for stage,mask in [('active',valid&(m['phase']>=1)),('balance',valid&(m['phase']==2))]:
                    for name,category in categories.items():coverage[stage][name] |= mask&category
                entering=valid&m['entered_balance'];first_balance[entering]=since[entering]
                # Measured pre-reset snapshot, never post-reset progress buffers.
                offsets[entering]=m['transition_offset'][entering]
                b=valid&(m['phase']==2)&(m['phase_time']>.5)
                track_sum+=m['raw_error'].abs()*b[:,None];track_count+=b
                legacy_error=torch.stack((s['legacy_forward'],s['legacy_heading_rate']),-1)-raw
                legacy_error_sum+=legacy_error.abs()*b[:,None]
                legacy_good+=b&(legacy_error.abs()<raw.new_tensor([.005,.015])+.25*raw.abs()).all(-1)
                wheel_axis=lateral_axis(tensor(env.scene['robot'].data.body_quat_w)[:,rear_body_ids])
                root_axis=lateral_axis(s['quat'])[:,None]
                axis_valid=valid&~done.bool()
                if axis_valid.any():wheel_axis_error=max(wheel_axis_error,float((wheel_axis-root_axis)[axis_valid].norm(dim=-1).max()))
                balance_good+=b&m['balanced'];balance_count+=b
                peak_force=torch.maximum(peak_force,torch.where(valid&(m['phase']==3),s['front_force_peak'],0.))
                peak_touch_speed=torch.maximum(peak_touch_speed,torch.where(valid,m['max_touchdown_speed'],0.))
                delta=(applied-previous).abs();max_frame=max(max_frame,float(delta[valid,0].max())*(hi-lo)/2 if valid.any() else 0.)
                max_wheel=max(max_wheel,float(delta[valid,1:].max())*a.max_speed if valid.any() else 0.)
                trace.append(dict(adapter_gate=gate.tolist(),adapter_correction=correction.tolist(),legacy_yaw=s['legacy_yaw'].tolist(),legacy_forward=s['legacy_forward'].tolist(),legacy_heading_rate=s['legacy_heading_rate'].tolist(),yaw_feedback_integral=m['yaw_feedback_integral'].tolist(),yaw_feedback_target=m['yaw_feedback_target'].tolist(),gravity=s['g'].tolist(),body_velocity=s['vel'].tolist(),failure_code=m['failure_code'].tolist(),t=(tick+1)*env.step_dt,valid=valid.tolist(),phase=m['phase'].tolist(),phase_time=m['phase_time'].tolist(),since=since.tolist(),raw=raw.tolist(),command=command.tolist(),pos=s['pos'].tolist(),heading=s['yaw'].tolist(),forward=s['forward'].tolist(),heading_rate=s['heading_rate'].tolist(),motion_feedback=m['motion_feedback'].tolist(),interval_motion=m['interval_motion'].tolist(),wheel_measured=s['wheel'].tolist(),front_z=s['front_z'].tolist(),front_vertical_speed=s['front_vertical_speed'].tolist(),rear_fraction=s['rear_fraction'].tolist(),four_fraction=s['four_fraction'].tolist(),applied=applied.tolist(),failure=m['failure'].tolist(),balanced=m['balanced'].tolist()))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
                alive &= ~done.bool()
                if not alive.any():break
        result=dict(max_wheel_axis_alignment_error=wheel_axis_error,
            legacy_tracking_mae=(legacy_error_sum/track_count.clamp_min(1)[:,None]).tolist(),
            legacy_joint_tracking_pass=int(((track_count>=20)&(legacy_good/track_count.clamp_min(1)>=.8)).sum()),count=n,**{k:int(v.sum()) for k,v in flags.items()},outcomes={k:v.tolist() for k,v in flags.items()},unfinished=int(alive.sum()),contract=contract,
            takeover_motion=takeover_motion.tolist(),balance_duration=balance_duration.tolist(),command_switches=switches.tolist(),command_transition_matrix=transition_matrix.tolist(),
            command_coverage={stage:{name:int(mask.sum()) for name,mask in groups.items()} for stage,groups in coverage.items()},
            first_balance_seconds=first_balance.tolist(),transition_offset_m=offsets.tolist(),balanced_fraction=(balance_good/balance_count.clamp_min(1)).tolist(),balance_samples=balance_count.tolist(),tracking_mae=(track_sum/track_count.clamp_min(1)[:,None]).tolist(),landing_peak_front_force=peak_force.tolist(),landing_peak_touchdown_speed=peak_touch_speed.tolist(),max_frame_target_step=max_frame,max_wheel_target_step=max_wheel)
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'trace.json').write_text(json.dumps(trace))
        assert max_frame<=args.frame_slew*env.step_dt+1e-5 and max_wheel<=args.wheel_slew*env.step_dt+1e-4
        assert not alive.any()
        print({k:v for k,v in result.items() if isinstance(v,(int,float))})
    env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
