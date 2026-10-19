"""Moving takeover task. Separate checkpoint contract from the static v5 baseline."""
import argparse,copy,hashlib,json,pathlib,subprocess
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=2048);p.add_argument('--iterations',type=int,default=1500)
p.add_argument('--seed',type=int,default=42);p.add_argument('--log_dir',default='logs/moving')
p.add_argument('--speed-limit',type=float,default=.3);p.add_argument('--yaw-limit',type=float,default=1.)
p.add_argument('--frame-min',type=float,default=-1.);p.add_argument('--frame-max',type=float,default=1.2)
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
    torch.manual_seed(args.seed)
    cfg=MovingEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    cfg.speed_limit=args.speed_limit;cfg.yaw_limit=args.yaw_limit;cfg.frame_range=(args.frame_min,args.frame_max)
    cfg.active_seconds=args.active_seconds;cfg.episode_length_s=args.active_seconds+6.
    env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty:raise RuntimeError('Commit source before formal execution')
    files=['train_moving.py',*['upright/'+x for x in ['moving.py','moving_task.py','task.py','distribution.py','model.py','asset_cfg.py','prepare.py']]]
    fingerprint=hashlib.sha256(''.join(hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in files).encode()).hexdigest()
    contract={'task':'moving_takeover_v1','fingerprint':fingerprint,'action_dim':3,'control_dt':env.step_dt,'beta_shape_cap':128.,'source_sha256':SPEC['source_sha256']}
    if args.checkpoint:
        old=json.loads((pathlib.Path(args.checkpoint).parent/'config.json').read_text())
        if old.get('contract')!=contract:raise ValueError('Incompatible checkpoint: task, implementation or action/observation contract changed')
    if args.evaluate and not args.checkpoint:raise ValueError('Evaluation requires a checkpoint')
    home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
    config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=250,logger='tensorboard',experiment_name='upright_moving',run_name=f'seed{args.seed}',
        obs_groups={'actor':['policy'],'critic':['policy']},
        actor=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.)),
        critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
        algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=.0003,schedule='fixed',gamma=.997**3,lam=.95**3,desired_kl=.01,max_grad_norm=1.))
    metadata=dict(contract=contract,args=vars(args),runner=config,working_tree_dirty=dirty,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    if args.checkpoint:runner.load(args.checkpoint)
    if not (args.evaluate or args.probe):runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    else:
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations()
        n=args.num_envs;device=args.device
        alive=torch.ones(n,device=device,dtype=torch.bool);taken=~alive;success=~alive;final=~alive
        extra=alive.clone();extra_seen=~alive;counts=torch.zeros(n,device=device);sum_error=torch.zeros(n,2,device=device);sum_path=counts.clone()
        low_speed=torch.zeros(n,device=device);moving_samples=low_speed.clone();max_acceleration=0.;last_forward=None
        premature=~alive;warmup_failed=~alive;max_frame=0.;max_wheel=0.;fixed_error=0.;frame_symmetry=0.;trace=[];takeovers=[]
        with torch.inference_mode():
            for tick in range(env.max_episode_length+3):
                action=env.action_manager.get_term('moving')
                a=action.applied.clone() if args.probe else policy(obs)
                previous=action.applied.clone();before=alive.clone()
                obs,reward,done,info=wrapped.step(a)
                s,m,raw,command,applied,since=env.moving_snapshot
                active=before&m['active'];switched=before&m['switched'];taken |= switched
                success |= active&m['held'];final=torch.where(before,m['held'],final)
                tail=active&(since>=args.active_seconds-2);extra_seen |= tail;extra &= ~tail | m['held']
                requested_motion=active&(raw[:,0].abs()>.05)
                moving_samples+=requested_motion;low_speed+=requested_motion&(s['forward'].abs()<.025)
                if last_forward is not None and active.any():max_acceleration=max(max_acceleration,float(((s['forward']-last_forward).abs()/env.step_dt)[active].max()))
                last_forward=s['forward'].clone()
                counts+=active;sum_error+=m['raw_error'].abs()*active[:,None];sum_path+=m['raw_distance']*active
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
            raw_velocity_mae_per_env=(sum_error/counts.clamp_min(1)[:,None]).tolist(),raw_path_mae_per_env=(sum_path/counts.clamp_min(1)).tolist(),active_samples=counts.tolist(),nonzero_command_low_speed_fraction=(low_speed/moving_samples.clamp_min(1)).tolist(),max_forward_acceleration=max_acceleration,
            max_frame_target_step=max_frame,max_wheel_target_step=max_wheel,max_fixed_error=fixed_error,frame_symmetry_target_error=frame_symmetry,takeovers=takeovers,contract=contract)
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'trace.json').write_text(json.dumps(trace))
        print({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
        assert frame_symmetry==0
        assert max_frame<=6*env.step_dt+1e-5 and max_wheel<=120*env.step_dt+1e-4
        assert all(x['four'] for x in takeovers)
    env.close()
finally:app.close()
