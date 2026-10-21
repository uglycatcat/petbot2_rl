"""Moving takeover task. Separate checkpoint contract from the static v5 baseline."""
import argparse,copy,hashlib,json,pathlib,subprocess
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=1500)
p.add_argument('--seed',type=int,default=42);p.add_argument('--log_dir',default='logs/moving')
p.add_argument('--speed-limit',type=float,default=.3);p.add_argument('--yaw-limit',type=float,default=1.)
p.add_argument('--frame-min',type=float,default=-1.);p.add_argument('--frame-max',type=float,default=1.2)
p.add_argument('--initialize-from')
p.add_argument('--feedback-mode',choices=['command','pose_feedback','heading_feedback','pose_feedback_tight'],default='command')
p.add_argument('--learning-rate',type=float,default=.0001)
p.add_argument('--contact-audit',action='store_true')
p.add_argument('--physical-contact-weight',type=float,default=0.)
p.add_argument('--save-interval',type=int,default=250)
p.add_argument('--wheel-slew',type=float,choices=[120.,900.],default=120.)
p.add_argument('--wheel-mode',choices=['independent','common_diff'],default='independent')
p.add_argument('--active-seconds',type=float,default=12.);p.add_argument('--checkpoint')
p.add_argument('--evaluate',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--allow-dirty',action='store_true')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner
from upright.moving_task import MovingEnvCfg,SPEC,ensure

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
        obs,r,term,timeout,extra=self.env.step(a);extra['time_outs']=timeout
        return TensorDict(obs,batch_size=[self.num_envs]),r,(term|timeout).long(),extra

try:
    if not (0<args.speed_limit<=.6 and 0<args.yaw_limit<=2.):raise ValueError('Unvalidated command envelope; initial implementation v<=.6m/s, yaw<=2rad/s')
    lo,hi=SPEC['joints']['left_frame']['range']
    if not lo<=args.frame_min<=args.frame_max<=hi:raise ValueError('Frame bounds outside source limits')
    if args.active_seconds<4:raise ValueError('At least four active seconds required')
    if args.checkpoint and args.initialize_from:raise ValueError('Use resume OR explicit initialization')
    if not 0<args.learning_rate<=.0003:raise ValueError('Learning rate outside validated bound')
    if not 0<=args.physical_contact_weight<=10:raise ValueError('Contact penalty magnitude must be in [0,10]')
    torch.manual_seed(args.seed)
    if args.contact_audit and not args.evaluate:raise ValueError('Contact audit is evaluation-only')
    cfg=MovingEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    cfg.speed_limit=args.speed_limit;cfg.yaw_limit=args.yaw_limit;cfg.frame_range=(args.frame_min,args.frame_max)
    cfg.feedback_mode=args.feedback_mode;cfg.rewards.path.weight=1. if args.feedback_mode=='command' else 4.
    cfg.wheel_slew=args.wheel_slew;cfg.wheel_mode=args.wheel_mode;cfg.active_seconds=args.active_seconds;cfg.episode_length_s=args.active_seconds+6.
    # Identical history sampling for treatment and zero-weight control.
    cfg.scene.contact_forces.history_length=cfg.decimation
    cfg.rewards.physical_contact.weight=-args.physical_contact_weight
    env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty:raise RuntimeError('Commit source before formal execution')
    files=['train_moving.py',*['upright/'+x for x in ['moving.py','moving_task.py','task.py','distribution.py','model.py','asset_cfg.py','prepare.py']]]
    fingerprint=hashlib.sha256(''.join(hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in files).encode()).hexdigest()
    contract={'task':'moving_takeover_audit_v1','observation_schema':'moving_41x5_raw_support_v1','physical_contact_weight':args.physical_contact_weight,'contact_history_steps':cfg.decimation,'feedback_mode':args.feedback_mode,'wheel_mode':args.wheel_mode,'wheel_slew':args.wheel_slew,'fingerprint':fingerprint,'action_dim':3,'control_dt':env.step_dt,'beta_shape_cap':128.,'source_sha256':SPEC['source_sha256']}
    if args.checkpoint:
        old=json.loads((pathlib.Path(args.checkpoint).parent/'config.json').read_text())
        if old.get('contract')!=contract:raise ValueError('Incompatible checkpoint: task, implementation or action/observation contract changed')
    if args.evaluate and not (args.checkpoint or args.initialize_from):raise ValueError('Evaluation requires a checkpoint')
    home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=args.save_interval,logger='tensorboard',experiment_name='upright_moving',run_name=f'seed{args.seed}',
        obs_groups={'actor':['policy'],'critic':['policy']},
        actor=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.)),
        critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
        algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=args.learning_rate,schedule='fixed',gamma=.997**3,lam=.95**3,desired_kl=.01,max_grad_norm=1.))
    metadata=dict(contract=contract,args=vars(args),runner=config,working_tree_dirty=dirty,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    if args.checkpoint:runner.load(args.checkpoint)
    if args.initialize_from:
        initial=pathlib.Path(args.initialize_from)
        prior=json.loads((initial.parent/'config.json').read_text());previous=prior['contract']
        sources={'34b0effd40c853a3e7fac0ebc1afa22a2e5943fe':'moving_takeover_v3', '4c1bd237c02181c2869fe11dd39a33a5b5798b67':'moving_takeover_v4', '22477c50c42c9ac76fb39fe98c045d8d0391c047':'moving_takeover_audit_v1'}
        if prior.get('commit') not in sources or prior.get('working_tree_dirty'):
            raise ValueError('Initialization is restricted to clean audited batch3/batch4/rebuild1 sources')
        for key in ['wheel_mode','wheel_slew','action_dim','control_dt','beta_shape_cap','source_sha256']:
            if previous.get(key)!=contract[key]:raise ValueError(f'Initialization contract mismatch: {key}')
        if previous.get('task')!=sources[prior['commit']] or previous.get('feedback_mode') not in ('pose_feedback','heading_feedback','pose_feedback_tight'):
            raise ValueError('Expected audited moving source')
        if previous.get('task')=='moving_takeover_audit_v1' and previous.get('observation_schema')!=contract['observation_schema']:raise ValueError('Observation schema mismatch')
        if prior['runner']['actor']!=config['actor'] or prior['runner']['critic']!=config['critic']:
            raise ValueError('Network or normalization interface changed')
        runner.load(str(initial),load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        assert runner.current_learning_iteration==0
        assert not runner.alg.optimizer.state, 'Initialization must not load optimizer moments'
        metadata['initialization']={'checkpoint':str(initial),'sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'source_commit':prior['commit'],'optimizer_loaded':False,'iteration_loaded':False}
        (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    if not (args.evaluate or args.probe):runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    else:
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations()
        n=args.num_envs;device=args.device
        alive=torch.ones(n,device=device,dtype=torch.bool);taken=~alive;success=~alive;final=~alive
        extra=alive.clone();extra_seen=~alive;counts=torch.zeros(n,device=device);sum_error=torch.zeros(n,2,device=device);sum_path=counts.clone()
        low_speed=torch.zeros(n,device=device);moving_samples=low_speed.clone();max_acceleration=0.;last_forward=None
        lifted=~alive;support_count=counts.clone();max_height=counts.clone();first_lift=torch.full((n,),-1.,device=device)
        premature=~alive;warmup_failed=~alive;max_frame=0.;max_wheel=0.;fixed_error=0.;frame_symmetry=0.;trace=[];takeovers=[]
        criterion_names=['support','forward','yaw_rate','path','heading','fixed','roll','tilt_rate']
        criterion_counts=torch.zeros(n,8,device=device);tail_criterion_counts=criterion_counts.clone();tail_counts=torch.zeros(n,device=device)
        heading_rate_error=counts.clone();physical_ended=~alive;path_ended=~alive
        rear_missing=counts.clone();contact_trace=[]
        with torch.inference_mode():
            for tick in range(env.max_episode_length+3):
                action=env.action_manager.get_term('moving')
                a=action.applied.clone() if args.probe else policy(obs)
                if args.probe and args.wheel_mode=='common_diff':
                    a[:,1]=action.applied[:,1:].mean(-1)
                    a[:,2]=(action.speed[:,1]-action.speed[:,0])/10.
                previous=action.applied.clone();before=alive.clone()
                obs,reward,done,info=wrapped.step(a)
                s,m,raw,command,applied,since=env.moving_snapshot
                if args.contact_audit:
                    from upright.task import tensor
                    h=tensor(env.scene['contact_forces'].data.net_forces_w_history)
                    assert h.ndim==4 and h.shape[1]==env.cfg.decimation and h.shape[-1]==3
                    fraction=(h[:,:,env.upright_ids[3],2]>.05).all(-1).float().mean(-1)
                    contact_trace.append(dict(t=tick*env.step_dt,valid=(before&m['active']&~done.bool()).tolist(),sampled_rear=s['rear'].tolist(),physics_rear_fraction=fraction.tolist(),front_z=s['front_z'].tolist(),other=s['other'].tolist()))
                active=before&m['active'];switched=before&m['switched'];taken |= switched
                event=active&m['lift_event'];first_lift=torch.where(event & ~lifted,since,first_lift);lifted |= event
                support_count+=active&m['support'];max_height=torch.maximum(max_height,torch.where(active,s['front_z'],0.))
                success |= active&m['held'];final=torch.where(before,m['held'],final)
                tail=active&(since>=args.active_seconds-2);extra_seen |= tail;extra &= ~tail | m['held']
                criteria=torch.stack((m['support'],m['raw_error'][:,0].abs()<.12,m['raw_error'][:,1].abs()<.3,m['raw_distance']<.15,m['heading_error'].abs()<.25,m['fixed_error']<.05,s['g'][:,1].abs()<.15,m['tilt_rate']<.8),-1)
                criterion_counts+=criteria*active[:,None];tail_criterion_counts+=criteria*tail[:,None];tail_counts+=tail
                requested_motion=active&(raw[:,0].abs()>.05)
                moving_samples+=requested_motion;low_speed+=requested_motion&(s['forward'].abs()<.025)
                if last_forward is not None and active.any():max_acceleration=max(max_acceleration,float(((s['forward']-last_forward).abs()/env.step_dt)[active].max()))
                last_forward=s['forward'].clone()
                counts+=active;sum_error+=m['raw_error'].abs()*active[:,None];sum_path+=m['raw_distance']*active
                heading_rate_error+=(s['heading_rate']-raw[:,1]).abs()*active
                rear_missing+=active&~s['rear']
                physical_ended |= before&m['physical_failure']
                path_ended |= active&(m['raw_distance']>1.5)
                premature |= before&done.bool()&~info['time_outs'].bool();warmup_failed |= before&m['warmup_failed']
                delta=(applied-previous).abs()
                max_frame=max(max_frame,float(delta[:,0].max())*(hi-lo)/2);max_wheel=max(max_wheel,float(delta[:,1:].max())*action.max_speed)
                fixed_error=max(fixed_error,float(m['fixed_error'][before].max()) if before.any() else 0.)
                frame_symmetry=max(frame_symmetry,float((action.target[:,3]-action.target[:,4]).abs().max()))
                for idx in torch.where(switched)[0].tolist():
                    takeovers.append(dict(env=idx,mode=int(ensure(env).mode[idx]),four=bool(s['four'][idx]),forward=float(s['forward'][idx]),yaw_rate=float(s['vel'][idx,5]),frame=s['q'][idx,3:].tolist(),raw=raw[idx].tolist()))
                trace.append(dict(t=(tick+1)*env.step_dt,active=active[:4].tolist(),pos=s['pos'][:4].tolist(),g=s['g'][:4].tolist(),raw=raw[:4].tolist(),command=command[:4].tolist(),forward=s['forward'][:4].tolist(),yaw_rate=s['vel'][:4,5].tolist(),front_z=s['front_z'][:4].tolist(),held=m['held'][:4].tolist(),applied=applied[:4].tolist()))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                alive &= ~done.bool()
                if not alive.any():break
        result=dict(count=n,taken_over=int(taken.sum()),success=int(success.sum()),maintained_final_2s=int((success&extra&extra_seen&~premature).sum()),final_success=int((final&~premature).sum()),premature=int(premature.sum()),warmup_failed=int(warmup_failed.sum()),unfinished=int(alive.sum()),
            criterion_names=criterion_names,criterion_fraction_per_env=(criterion_counts/counts.clamp_min(1)[:,None]).tolist(),tail_criterion_fraction_per_env=(tail_criterion_counts/tail_counts.clamp_min(1)[:,None]).tolist(),tail_samples=tail_counts.tolist(),
            success_per_env=success.tolist(),maintained_per_env=(success&extra&extra_seen&~premature).tolist(),premature_per_env=premature.tolist(),
            heading_rate_mae_per_env=(heading_rate_error/counts.clamp_min(1)).tolist(),
            rear_missing_fraction_per_env=(rear_missing/counts.clamp_min(1)).tolist(),
            physical_failure_per_env=physical_ended.tolist(),path_failure_per_env=path_ended.tolist(),
            lifted=int(lifted.sum()),lifted_per_env=lifted.tolist(),first_lift_seconds=first_lift.tolist(),support_fraction_per_env=(support_count/counts.clamp_min(1)).tolist(),max_front_height_per_env=max_height.tolist(),
            raw_velocity_mae_per_env=(sum_error/counts.clamp_min(1)[:,None]).tolist(),raw_path_mae_per_env=(sum_path/counts.clamp_min(1)).tolist(),active_samples=counts.tolist(),nonzero_command_low_speed_fraction=(low_speed/moving_samples.clamp_min(1)).tolist(),max_forward_acceleration=max_acceleration,
            max_frame_target_step=max_frame,max_wheel_target_step=max_wheel,max_fixed_error=fixed_error,frame_symmetry_target_error=frame_symmetry,takeovers=takeovers,contract=contract)
        if args.contact_audit:(folder/'contact_trace.json').write_text(json.dumps(contact_trace))
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'trace.json').write_text(json.dumps(trace))
        print({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
        assert frame_symmetry==0
        assert max_frame<=6*env.step_dt+1e-5 and max_wheel<=args.wheel_slew*env.step_dt+1e-4
        assert all(x['four'] for x in takeovers)
    env.close()
except BaseException:
    # SimulationApp.close may terminate the interpreter before an exception is
    # printed. Emit the traceback and preserve failure status for the supervisor.
    import traceback,sys,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    os._exit(1)
else:
    app.close()
