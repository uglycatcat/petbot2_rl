"""Versioned full upright cycle; independent observation and evaluation contract."""
import argparse,copy,hashlib,json,pathlib,subprocess,os
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
p.add_argument('--tracking-weight',type=float,choices=[4.,6.,12.],default=12.)
p.add_argument('--instant-tracking-weight',type=float,choices=[.3,.7],default=.3)
p.add_argument('--tracking-objective',choices=['mixed','instant_tolerance','stability_first'],default='mixed')
p.add_argument('--observation-frame',choices=['world','local','balance_local'],default='world')
p.add_argument('--handoff-alignment',choices=['before','after'],default='before')
p.add_argument('--refinement-target',choices=['balance','rise','all'],default='balance')
p.add_argument('--transition-contract',choices=['legacy','linear_heading'],default='legacy')
p.add_argument('--task-profile',choices=['legacy','full_range'],default='legacy')
p.add_argument('--training-mixture',choices=['off','stratified_v1'],default='off')
p.add_argument('--distributed',action='store_true')
p.add_argument('--balance-reflex',choices=['off','frozen20'],default='off')
p.add_argument('--balance-noise-factor',type=float,choices=[1.,4.,16.],default=1.)
p.add_argument('--actor-normalization',choices=['inherited','rebase_recent'],default='inherited')
p.add_argument('--yaw-feedback-design',choices=['legacy60','fast20','fast2'],default='legacy60')
p.add_argument('--yaw-feedback-gain',type=float,choices=[8.,16.],default=8.)
p.add_argument('--steering-design',choices=['feedback'],default='feedback')
p.add_argument('--hold-min',type=float,default=8.);p.add_argument('--hold-max',type=float,default=12.)
p.add_argument('--initialize-cycle-from');p.add_argument('--initialize-from');p.add_argument('--checkpoint');p.add_argument('--evaluate',action='store_true')
p.add_argument('--allow-dirty',action='store_true');p.add_argument('--probe',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
rank=int(os.getenv('RANK','0'));world_size=int(os.getenv('WORLD_SIZE','1'))
if world_size>1:
    if not args.distributed:raise ValueError('Multi-GPU launch requires --distributed')
    args.device='cuda:'+os.getenv('LOCAL_RANK','0')
app=AppLauncher(args).app
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
    def get_observations(self):
        self.last_observation=TensorDict(self.env.observation_manager.compute(),batch_size=[self.num_envs])
        return self.last_observation
    def step(self,a):
        self.env.action_manager.get_term('moving').policy_observation=self.last_observation['policy']
        obs,r,term,timeout,extra=self.env.step(a)
        extra['time_outs']=timeout & ~term
        self.last_observation=TensorDict(obs,batch_size=[self.num_envs])
        return self.last_observation,r,(term|timeout).long(),extra

try:
    assert 0<args.speed_limit<=.75 and 0<args.yaw_limit<=15.
    assert 1<=args.num_envs<=4096
    assert -1<=args.frame_min<=args.frame_max<=1.2 and 0<args.frame_slew<=6
    assert sum(bool(x) for x in (args.checkpoint,args.initialize_from,args.initialize_cycle_from))<=1
    assert 2.<=args.hold_min<=args.hold_max<=180.
    if args.initialize_from:raise ValueError('V19 requires explicit validated cycle actor transfer')
    if args.refinement_target in ('rise','all'):
        assert args.observation_frame=='balance_local' and args.handoff_alignment=='after'
    assert args.exploration_profile=='fine' and args.yaw_feedback_design=='fast2' and args.yaw_feedback_gain==16.
    if args.task_profile=='full_range':
        assert args.refinement_target=='all' and args.transition_contract=='linear_heading'
    else:assert args.actor_normalization=='inherited'
    if args.training_mixture!='off':assert args.task_profile=='full_range'
    cfg=CycleEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    cfg.speed_limit=args.speed_limit;cfg.yaw_limit=args.yaw_limit;cfg.frame_range=(args.frame_min,args.frame_max)
    cfg.wheel_slew=args.wheel_slew;cfg.frame_slew=args.frame_slew;cfg.wheel_mode='common_diff'
    cfg.feedback_mode='pose_feedback_tight';cfg.rewards.tracking.weight=args.tracking_weight;cfg.tracking_shape=args.tracking_shape;cfg.yaw_filter_tau=args.yaw_filter_tau
    cfg.steering_design=args.steering_design;cfg.hold_range=(args.hold_min,args.hold_max)
    cfg.tracking_objective=args.tracking_objective;cfg.observation_frame=args.observation_frame
    cfg.handoff_alignment=args.handoff_alignment
    cfg.transition_contract=args.transition_contract;cfg.task_profile=args.task_profile
    cfg.training_mixture='off' if args.evaluate else args.training_mixture
    if args.transition_contract=='linear_heading':cfg.rewards.transition.weight=2.
    cfg.instant_tracking_weight=args.instant_tracking_weight;cfg.yaw_feedback_design=args.yaw_feedback_design;cfg.yaw_feedback_gain=args.yaw_feedback_gain
    cfg.active_seconds=args.hold_max+(18. if args.task_profile=='full_range' else 12.)
    cfg.episode_length_s=cfg.active_seconds+(26. if args.task_profile=='full_range' else 8.)
    cfg.seed=args.seed+rank;torch.manual_seed(cfg.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty:raise RuntimeError('Commit source before formal execution')
    names=['train_cycle.py','upright/rise_adapter.py','upright/balance_adapter.py','upright/balance_reflex.py','upright/stability.py','upright/functional.py','upright/local_frame.py']+['upright/'+n+'.py' for n in ['normalization','fast_feedback','ground_motion','feedback','feedback_task','continuous','continuous_task','cycle','cycle_task','moving','task','distribution','model','asset_cfg','prepare','transfer']]
    names+=['upright/kinematics.py','upright/full_range.py','upright/behavior.py']
    fingerprint=hashlib.sha256(''.join(hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in names).encode()).hexdigest()
    scales={'moderate':[1.,4.,4.],'fine':[4.,16.,16.],'precision':[16.,64.,64.]}[args.exploration_profile]
    contract=dict(task='upright_cycle_v19',balance_reflex=args.balance_reflex,reflex_contract=dict(parent='cycle15B',period=.02,gate_after=1.8,frozen=True,clock='outer60ms',physical_features_only=True),tracking_objective=args.tracking_objective,actor_design='frozen_base_balance_yaw_adapter_v1',balance_noise_factor=args.balance_noise_factor,adapter_limits=[.03,.08,.8],adapter_gate=[1.2,1.8],adapter_observation_stats='frozen_parent_critic',actor_normalization=args.actor_normalization,normalization_prior_count=10000000 if args.actor_normalization=='rebase_recent' else None,yaw_feedback_design=args.yaw_feedback_design,fast_yaw_controller=dict(FAST_CONTROLLER,period=.002 if args.yaw_feedback_design=='fast2' else .02,kp=args.yaw_feedback_gain),heading_reference='wheel_lateral',contact_sampling='eager_physics_step_history30_v1',instant_tracking_weight=args.instant_tracking_weight,yaw_controller=CONTROLLER,steering_design=args.steering_design,yaw_filter_tau=args.yaw_filter_tau,tracking_shape=args.tracking_shape,command_sampling='all_directed_six_sequence_v3',reward_design=('stability_first_coarse_motion_v1' if args.tracking_objective=='stability_first' else 'instant_tolerance_soft_landing_v1' if args.tracking_objective=='instant_tolerance' else 'joint_ground_motion_soft_landing_v2'),motion_feedback='pose_delta_ema_120ms',concentration_scale=scales,tracking_weight=args.tracking_weight,observation_schema='cycle_58x5_wheel_lateral_v2',fingerprint=fingerprint,
        action_dim=3,wheel_mode='common_diff',wheel_slew=args.wheel_slew,frame_slew=args.frame_slew,
        control_dt=env.step_dt,source_sha256=SPEC['source_sha256'],rear_duty_threshold=.9,
        transition_reference='reanchor_once_on_balance',mode_request='landing_latched_until_handoff',landing_force_limit_n=20.,touchdown_speed_limit=.35)
    from upright.stability import STABILITY_CONTRACT
    from upright.functional import FUNCTIONAL_CONTRACT
    from upright.local_frame import LOCAL_FRAME_CONTRACT
    from upright.rise_adapter import RISE_CONTRACT, rise_gate
    contract['task']='upright_cycle_v20'
    from upright.kinematics import CONTRACT as KINEMATIC_CONTRACT
    contract['transition_contract']=args.transition_contract
    contract['kinematic_landing']=KINEMATIC_CONTRACT if args.transition_contract=='linear_heading' else None
    contract['refinement_target']=args.refinement_target
    contract['rise_adapter']=RISE_CONTRACT if args.refinement_target in ('rise','all') else None
    if args.refinement_target in ('rise','all'):contract['actor_design']='full_cycle_finetune_v1' if args.refinement_target=='all' else 'frozen_cycle_rise_adapter_v1'
    contract.update(observation_frame=args.observation_frame,local_frame=LOCAL_FRAME_CONTRACT)
    contract['handoff_alignment']=dict(mode=args.handoff_alignment,physical_ground_hold_s=.30,completion_aligned_hold_s=1.,frame_tolerance_rad=.1,ordinary_action_slew_preserved=True)
    if args.transition_contract=='linear_heading':
        contract['handoff_alignment'].update(physical_ground_hold_s=None,kinematic_ground_hold_s=.12)
        contract['transition_reference']='capture_measured_forward_speed_and_heading_v1'
        contract['local_frame']=dict(LOCAL_FRAME_CONTRACT,initial_heading='captured_takeover_heading',
            transition_interpolation='shortest_angle',original_rise=False)
        contract['fast_yaw_controller'].update(active_after=0.,active_phases=[1,2,3],half_difference_limit=25.,transition_policy_residual=0.)
    if args.observation_frame=='local':contract['observation_schema']='cycle_58x5_wheel_ground_local_v3'
    if args.observation_frame=='balance_local':contract['observation_schema']='cycle_58x5_balance_local_v4'
    contract.update(stability=STABILITY_CONTRACT,functional=FUNCTIONAL_CONTRACT,workspace='3m_plus_speed_times_hold_and_transition_v1')
    if args.transition_contract=='linear_heading':
        contract['stability']=dict(STABILITY_CONTRACT,detector='imu_frame_geometry_v1',
            reasons=['front_low','excessive_rear_axle_height_difference','overturned','persistent_overspeed'])
    if args.task_profile=='full_range':
        from upright.full_range import CONTRACT as FULL_RANGE_CONTRACT
        contract.update(task='upright_cycle_v21',task_profile='full_range',independent_call=FULL_RANGE_CONTRACT,command_sampling='feasible_six_sequence_v4',reward_design='independent_call_stability_v1',observation_request='phase_effective_v1')
        contract['stability']=dict(contract['stability'],persistent_overspeed_termination=False,early_return='armed_lift_then_return_in_rise_or_balance')
        contract['fast_yaw_controller'].update(anti_windup=True,transition_heading_rate_limit=15.)
        from upright.full_range import TRAINING_MIXTURE
        contract['task']='upright_cycle_v22'
        contract['training_distribution']=TRAINING_MIXTURE if args.training_mixture!='off' else dict(version='off')
    if args.checkpoint:
        prior=json.loads((pathlib.Path(args.checkpoint).parent/'config.json').read_text())
        if prior['contract']!=contract:raise ValueError('Checkpoint contract mismatch')
    if args.evaluate and not args.checkpoint:raise ValueError('Evaluation requires a cycle checkpoint')
    lo,hi=SPEC['joints']['left_frame']['range'];home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=args.save_interval,logger='tensorboard',experiment_name='upright_cycle',run_name=f'seed{args.seed}',
        obs_groups={'actor':['policy'],'critic':['policy']},
        actor=dict(class_name='upright.balance_reflex:ReflexAdapterModel',balance_noise_factor=args.balance_noise_factor,hidden_dims=[128,128],activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.,concentration_scale=scales)),
        critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
        algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=args.learning_rate,schedule='fixed',gamma=.997**3,lam=.95**3,desired_kl=.01,max_grad_norm=1.))
    if args.refinement_target in ('rise','all'):
        config['actor']['class_name']='upright.rise_adapter:RiseAdapterModel'
        config['algorithm']['lam']=RISE_CONTRACT['gae_lambda']
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    if args.refinement_target=='all':
        actor=runner.alg.get_policy()
        for name,parameter in actor.named_parameters():parameter.requires_grad_(not name.startswith('balance_reflex.'))
        optimized={id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
        assert all(id(p) in optimized for p in actor.parameters() if p.requires_grad)
    metadata=dict(contract=contract,args=vars(args),runner=config,working_tree_dirty=dirty,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if args.initialize_cycle_from:
        initial=pathlib.Path(args.initialize_cycle_from);prior=json.loads((initial.parent/'config.json').read_text())
        allowed_sources={'upright_cycle_v16':'03deaca07644e244040319df07ebfc1347fb5f1e','upright_cycle_v17':'a64c75569fb50b6f24a122a3affd16fc0d25e2ca'}
        if args.refinement_target=='all' and args.transition_contract=='linear_heading':
            allowed_sources['upright_cycle_v19']='001a54e98ff18f1274e2d7e961404325de4e50f7'
        if args.task_profile=='full_range':
            allowed_sources['upright_cycle_v20']='b269d49aacb50d5bf5d8a3d9cd618696e087734e'
            allowed_sources['upright_cycle_v21']='c30b2c209d7c212e442c7614b17aeaf69205f19d'
        if prior['working_tree_dirty'] or allowed_sources.get(prior['contract'].get('task'))!=prior['commit']:raise ValueError('Only validated clean cycle-v16/v17 source allowed')
        if prior['contract']['observation_schema'] not in ('cycle_58x5_wheel_lateral_v2','cycle_58x5_balance_local_v4') or prior['contract']['steering_design']!='feedback':raise ValueError('Source observation schema mismatch')
        for k in ('action_dim','wheel_mode','wheel_slew','frame_slew','control_dt','source_sha256','concentration_scale','instant_tracking_weight','balance_noise_factor'):
            if k=='balance_noise_factor' and args.task_profile=='full_range':continue
            if prior['contract'][k]!=contract[k]:raise ValueError('Transfer mismatch '+k)
        if prior['contract']['task']!='upright_cycle_v21' and prior['contract']['balance_noise_factor']!=16.:raise ValueError('Legacy parents require validated exploration')
        if prior['contract']['tracking_objective']!='stability_first' or prior['contract']['yaw_feedback_design']!='fast2' or prior['contract']['fast_yaw_controller']['kp']!=16.:raise ValueError('Preserve validated stability-first parents')
        if prior['contract']['balance_reflex']!='frozen20' or args.balance_reflex!='frozen20':raise ValueError('V19 preserves validated frozen20 feedback')
        loaded=torch.load(initial,map_location='cpu',weights_only=False)
        actor=runner.alg.get_policy()
        if args.refinement_target=='rise':actor.initialize_rise(loaded['actor_state_dict'],loaded['critic_state_dict'])
        else:actor.load_state_dict(loaded['actor_state_dict'],strict=True)
        runner.alg.critic.load_state_dict(loaded['critic_state_dict'],strict=True)
        normalization_review=None
        if args.actor_normalization=='rebase_recent':
            from upright.normalization import rebase_all_branches
            normalization_review=rebase_all_branches(actor,loaded['critic_state_dict'])
        assert runner.current_learning_iteration==0 and not runner.alg.optimizer.state
        metadata['initialization']=dict(source_commit=prior['commit'],checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),actor_base_frozen=True,critic_loaded=True,optimizer_loaded=False,prior_adapter_loaded=True,yaw_adapter_loaded=True,reward_changed=args.tracking_objective!=prior['contract']['tracking_objective'] or args.tracking_weight!=prior['contract']['tracking_weight'],observation_frame_changed=args.observation_frame!=prior['contract'].get('observation_frame','world'),task_changed=True,handoff_alignment_changed=args.handoff_alignment!='before',feedback_period_changed=False,feedback_gain_changed=False,frozen_balance_reflex=args.balance_reflex,observation_width=58)
        metadata['initialization']['normalization_review']=normalization_review
        metadata['initialization']['exploration_changed']=prior['contract']['balance_noise_factor']!=args.balance_noise_factor
        metadata['initialization']['trainable_actor_scope']=args.refinement_target
        metadata['initialization']['actor_base_frozen']=args.refinement_target!='all'
        del loaded
    if args.checkpoint:runner.load(args.checkpoint)
    from upright.balance_reflex import attach_balance_reflex
    attach_balance_reflex(env,runner.alg.get_policy(),args.balance_reflex=='frozen20')
    metadata['distributed']=dict(world_size=world_size,rank=rank,local_envs=args.num_envs,total_envs=world_size*args.num_envs)
    metadata['effective_training_mixture']=cfg.training_mixture
    if rank==0:(folder/'config.json').write_text(json.dumps(metadata,indent=2))
    if not args.evaluate and not args.probe:runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    else:
        from upright.balance_adapter import balance_gate
        from upright.distribution import BoundedBetaMean
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations();n=args.num_envs
        alive=torch.ones(n,device=args.device,dtype=torch.bool)
        flags={k:torch.zeros_like(alive) for k in ['takeover','balance','landing','handoff','complete','qualified','failure','interrupted']}
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
                if args.refinement_target in ('rise','all'):
                    from upright.balance_reflex import ReflexAdapterModel
                    parent_action=ReflexAdapterModel.forward(actor,obs,stochastic_output=False)
                    lift_gate=rise_gate(obs['policy']);lift_correction=action-parent_action
                    expected_lift=(progress.phase==1)*(progress.phase_time/.12).clamp(0.,1.)+(progress.phase==2)*(1.-progress.phase_time/.6).clamp(0.,1.)
                    assert torch.allclose(lift_gate[alive],expected_lift[alive],atol=1e-6), 'Rise phase/history mismatch'
                    assert (lift_correction.abs()<=lift_correction.new_tensor([.15,.25,0.])+1e-6).all()
                    assert not lift_correction[lift_gate==0].any()
                else:
                    parent_action=action;lift_gate=torch.zeros_like(gate);lift_correction=torch.zeros_like(action)
                correction=parent_action-base_action
                assert (correction.abs()<=correction.new_tensor([.03,.08,.8])+1e-6).all()
                assert not correction[gate==0].any()
                obs,r,done,extra=wrapped.step(action)
                s,m,raw,command,applied,since=env.moving_snapshot;valid=alive.clone()
                assert torch.equal(raw[valid],decision_raw[valid]), 'Command changed after policy decision'
                assert torch.equal(m['failure'],m['failure_code']!=0), 'Incomplete failure cause classification'
                for key,event in [('takeover',m['switched']),('balance',m['entered_balance']),('landing',m['entered_landing']),('handoff',m['entered_handoff']),('complete',m['completed']),('qualified',m['qualified']),('failure',m['failure'])]:flags[key] |= valid & event
                if 'interrupted' in m:flags['interrupted'] |= valid&m['interrupted']
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
                trace.append(dict(rise_adapter_gate=lift_gate.tolist(),rise_adapter_correction=lift_correction.tolist(),instability=m['instability'].tolist(),instability_dwell=m['instability_dwell'].tolist(),body_contact=s['body_history'].tolist(),adapter_gate=gate.tolist(),adapter_correction=correction.tolist(),legacy_yaw=s['legacy_yaw'].tolist(),legacy_forward=s['legacy_forward'].tolist(),legacy_heading_rate=s['legacy_heading_rate'].tolist(),yaw_feedback_integral=m['yaw_feedback_integral'].tolist(),yaw_feedback_target=m['yaw_feedback_target'].tolist(),gravity=s['g'].tolist(),body_velocity=s['vel'].tolist(),failure_code=m['failure_code'].tolist(),t=(tick+1)*env.step_dt,valid=valid.tolist(),phase=m['phase'].tolist(),phase_time=m['phase_time'].tolist(),since=since.tolist(),raw=raw.tolist(),command=command.tolist(),pos=s['pos'].tolist(),heading=s['yaw'].tolist(),forward=s['forward'].tolist(),heading_rate=s['heading_rate'].tolist(),motion_feedback=m['motion_feedback'].tolist(),interval_motion=m['interval_motion'].tolist(),wheel_measured=s['wheel'].tolist(),front_z=s['front_z'].tolist(),front_vertical_speed=s['front_vertical_speed'].tolist(),rear_fraction=s['rear_fraction'].tolist(),four_fraction=s['four_fraction'].tolist(),applied=applied.tolist(),failure=m['failure'].tolist(),balanced=m['balanced'].tolist()))
                trace[-1].update(joint_position=s['q'].tolist(),joint_velocity=s['qd'].tolist(),position_target=m['position_target'].tolist(),wheel_target=m['wheel_target'].tolist(),actuator_effort=m['effort'].tolist(),transition_heading_error=m.get('transition_heading_error',torch.zeros(n,device=args.device)).tolist(),geometry_ground=m.get('geometry_ground',s['four']).tolist(),interrupted=m.get('interrupted',torch.zeros_like(valid)).tolist(),exit_phase=m.get('exit_phase',torch.zeros_like(category)).tolist(),requested=m.get('requested_command',raw).tolist(),switched=m['switched'].tolist())
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
                alive &= ~done.bool()
                if not alive.any():break
        result=dict(max_wheel_axis_alignment_error=wheel_axis_error,
            legacy_tracking_mae=(legacy_error_sum/track_count.clamp_min(1)[:,None]).tolist(),
            legacy_joint_tracking_pass=int(((track_count>=20)&(legacy_good/track_count.clamp_min(1)>=.8)).sum()),count=n,**{k:int(v.sum()) for k,v in flags.items()},outcomes={k:v.tolist() for k,v in flags.items()},unfinished=int(alive.sum()),contract=contract,
            takeover_motion=takeover_motion.tolist(),balance_duration=balance_duration.tolist(),command_switches=switches.tolist(),command_transition_matrix=transition_matrix.tolist(),
            command_coverage={stage:{name:int(mask.sum()) for name,mask in groups.items()} for stage,groups in coverage.items()},
            first_balance_seconds=first_balance.tolist(),transition_offset_m=offsets.tolist(),balanced_fraction=(balance_good/balance_count.clamp_min(1)).tolist(),balance_samples=balance_count.tolist(),tracking_mae=(track_sum/track_count.clamp_min(1)[:,None]).tolist(),landing_peak_front_force=peak_force.tolist(),landing_peak_touchdown_speed=peak_touch_speed.tolist(),max_frame_target_step=max_frame,max_wheel_target_step=max_wheel)
        from upright.functional import score_trajectories
        result['functional']=score_trajectories(trace,result['outcomes'])
        from upright.behavior import review_trajectories
        result['behavior']=review_trajectories(trace,result['outcomes'])
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'trace.json').write_text(json.dumps(trace))
        assert max_frame<=args.frame_slew*env.step_dt+1e-5 and max_wheel<=args.wheel_slew*env.step_dt+1e-4
        assert not alive.any()
        print({k:v for k,v in result.items() if isinstance(v,(int,float))})
    env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:app.close()
