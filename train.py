"""Isaac Lab 3 / RSL-RL 5 PPO entry point. Run only inside the dedicated container."""
import argparse, copy, hashlib, json, pathlib, subprocess
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=2048)
parser.add_argument('--iterations',type=int,default=1500)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--log_dir',default='logs/upright')
parser.add_argument('--checkpoint')
parser.add_argument('--evaluate',action='store_true')
parser.add_argument('--probe',action='store_true')
parser.add_argument('--allow-dirty',action='store_true',help='Development gates only; formal runs require a clean checkout')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from upright.task import UprightEnvCfg, SPEC, state, metrics

class Adapter:
    def __init__(self,env):
        self.env=env;self.cfg=env.cfg;self.device=env.device;self.num_envs=env.num_envs
        self.num_actions=2;self.max_episode_length=env.max_episode_length
        env.reset()
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,v):self.env.episode_length_buf=v
    def get_observations(self):return TensorDict(self.env.observation_manager.compute(),batch_size=[self.num_envs])
    def step(self,a):
        obs,r,term,timeout,extra=self.env.step(a)
        extra['time_outs']=timeout
        return TensorDict(obs,batch_size=[self.num_envs]),r,(term|timeout).long(),extra

lo,hi=SPEC['joints']['left_frame']['range']
home=(2*SPEC['joints']['left_frame']['default']-lo-hi)/(hi-lo)
config=dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=250,
    logger='tensorboard',experiment_name='upright_rl_demo',run_name=f'seed{args.seed}',
    obs_groups={'actor':['policy'],'critic':['policy']},
    actor=dict(class_name='MLPModel',hidden_dims=[128,128],activation='elu',obs_normalization=True,
        distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[home,0.],initial_std=[.25,.15])),
    critic=dict(class_name='MLPModel',hidden_dims=[256,128],activation='elu',obs_normalization=True),
    algorithm=dict(class_name='PPO',value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
        entropy_coef=.001,num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,
        schedule='adaptive',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.))
try:
    cfg=UprightEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
    torch.manual_seed(args.seed)
    env=ManagerBasedRLEnv(cfg=cfg)
    wrapped=Adapter(env)
    folder=pathlib.Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty and not args.allow_dirty: raise RuntimeError('Commit changes before a formal run')
    from upright.model import task_implementation_fingerprint
    metadata={'working_tree_dirty':dirty,'implementation_sha256':task_implementation_fingerprint(),
        'entrypoint_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'args':vars(args),'runner':config,'source_sha256':SPEC['source_sha256'],
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (folder/'config.json').write_text(json.dumps(metadata,indent=2))
    runner=OnPolicyRunner(wrapped,copy.deepcopy(config),log_dir=str(folder),device=args.device)
    if args.checkpoint:runner.load(args.checkpoint)
    if args.probe or args.evaluate:
        policy=runner.get_inference_policy(device=args.device)
        obs=wrapped.get_observations()
        alive=torch.ones(args.num_envs,device=args.device,dtype=torch.bool)
        success=torch.zeros_like(alive);max_height=torch.zeros(args.num_envs,device=args.device)
        max_asym=0.;max_fixed_error=0.
        with torch.inference_mode():
            for _ in range(env.max_episode_length):
                a=policy(obs) if args.evaluate else obs['policy'].new_tensor([home,0.]).expand(args.num_envs,-1)
                # Record before stepping: done environments reset in step, don't count their next episode.
                s,support,good,held,fresh=metrics(env)
                success |= alive & held
                max_height=torch.maximum(max_height,torch.where(alive,s['front_z'],0.))
                action=env.action_manager.get_term('symmetric')
                max_asym=max(max_asym,(action.target[:,3]-action.target[:,4]).abs().max().item(),(action.speed[:,0]-action.speed[:,1]).abs().max().item())
                max_fixed_error=max(max_fixed_error,(action.target[:,:3]-action.target.new_tensor([SPEC['joints'][n]['default'] for n in ['head','left_ear','right_ear']])).abs().max().item())
                obs,r,done,extra=wrapped.step(a)
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(r).all()
                alive &= ~done.bool()
        result=dict(success=int(success.sum()),count=args.num_envs,max_front_height=float(max_height.max()),
                    symmetry_target_error=max_asym,fixed_joint_target_error=max_fixed_error)
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2));print(result)
        assert max_asym==0 and max_fixed_error==0
    else:
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    env.close()
finally:
    app.close()
