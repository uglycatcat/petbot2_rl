"""Scratch grouped PPO. One independent policy per fixed initial pose."""
import argparse
import json
import os
from pathlib import Path
import random
import time

parser=argparse.ArgumentParser()
parser.add_argument('--num-envs',type=int,default=8192)
parser.add_argument('--iterations',type=int,default=1000)
parser.add_argument('--steps',type=int,default=64)
parser.add_argument('--seed',type=int,default=501)
parser.add_argument('--run-dir',required=True)
parser.add_argument('--learning-rate',type=float,default=3e-4)
parser.add_argument('--entropy',type=float,default=.02)
parser.add_argument('--width',type=int,default=128)
parser.add_argument('--save-every',type=int,default=100)
parser.add_argument('--resume')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from recover.env import RecoverEnv,EnvCfg
    from recover.policy import Policy
    from recover.contract import OBS_DIM,ACTION_DIM,GAMMA
    from recover.model import task_implementation_fingerprint,source_fingerprint
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    cfg=EnvCfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed
    env=RecoverEnv(cfg); env.render_enabled=False
    model=Policy(width=args.width).to(env.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,eps=1e-5)
    n=args.num_envs//4; T=args.steps; device=env.device
    shapes={'obs':(OBS_DIM,), 'raw':(ACTION_DIM,), 'logp':(), 'value':(), 'reward':(), 'done':()}
    buf={k:torch.empty((T,4,n,*shape),device=device) for k,shape in shapes.items()}
    advantage=torch.empty(T,4,n,device=device)
    code_hash=task_implementation_fingerprint(); source_hash=source_fingerprint()
    start_update=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device,weights_only=False)
        if checkpoint['code_hash']!=code_hash or checkpoint['source_hash']!=source_hash:
            raise ValueError('Interrupted-run resume requires identical code and source model')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start_update=checkpoint['iteration']
        torch.set_rng_state(checkpoint['rng_cpu']); torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    (run/'config.json').write_text(json.dumps(dict(vars(args),code_hash=code_hash,source_hash=source_hash,
        policy_count=4,independent_policies=True,init='resume_same_run' if args.resume else 'random_scratch'),indent=2)+'\n')
    writer=SummaryWriter(str(run/'tensorboard'))
    obs=env.reset()[0]['policy'].reshape(4,n,OBS_DIM)
    start=time.monotonic(); samples=0
    def save(iteration,name):
        path=run/name; temp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,
                        args=vars(args),code_hash=code_hash,source_hash=source_hash,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
                        physics_summary=env.summary()),temp)
        temp.replace(path)
    for update in range(start_update+1,args.iterations+1):
        began=time.monotonic()
        with torch.no_grad():
            for t in range(T):
                raw,logp,value=model.sample(obs)
                out,reward,terminated,truncated,_=env.step(torch.tanh(raw).reshape(-1,ACTION_DIM))
                buf['obs'][t]=obs; buf['raw'][t]=raw; buf['logp'][t]=logp; buf['value'][t]=value
                buf['reward'][t]=reward.reshape(4,n); buf['done'][t]=(terminated|truncated).reshape(4,n)
                obs=out['policy'].reshape(4,n,OBS_DIM)
            bootstrap=model.value(obs); gae=torch.zeros_like(bootstrap)
            for t in reversed(range(T)):
                live=1-buf['done'][t]
                delta=buf['reward'][t]+GAMMA*bootstrap*live-buf['value'][t]
                gae=delta+GAMMA*.97*live*gae
                advantage[t]=gae; bootstrap=buf['value'][t]
            returns=advantage+buf['value']
            adv=advantage.permute(1,0,2).reshape(4,-1)
            adv=(adv-adv.mean(-1,keepdim=True))/(adv.std(-1,keepdim=True)+1e-8)
            ret=returns.permute(1,0,2).reshape(4,-1)
            flat={k:v.permute(1,0,2,*range(3,v.ndim)).reshape(4,T*n,*v.shape[3:]) for k,v in buf.items()}
        losses=[]; kls=[]; grad_peak=0.
        for epoch in range(4):
            order=torch.randperm(T*n,device=device)
            for idx in order.chunk(8):
                lp,entropy,value=model.evaluate(flat['obs'][:,idx],flat['raw'][:,idx])
                ratio=(lp-flat['logp'][:,idx]).exp()
                pg=-torch.minimum(adv[:,idx]*ratio,adv[:,idx]*ratio.clamp(.8,1.2)).mean()
                vf=.5*(value-ret[:,idx]).square().mean()
                loss=pg+vf-args.entropy*entropy.mean()
                if not torch.isfinite(loss): raise RuntimeError('Non-finite PPO loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad): raise RuntimeError('Non-finite gradient')
                optimizer.step(); grad_peak=max(grad_peak,float(grad))
                with torch.no_grad():
                    kl=((ratio-1)-(lp-flat['logp'][:,idx])).mean()
                losses.append(float(loss.detach())); kls.append(float(kl))
            if np.mean(kls[-8:])>.03: break
        samples+=T*args.num_envs
        report=dict(iteration=update,total_iterations=args.iterations,seconds=time.monotonic()-start,
                    update_seconds=time.monotonic()-began,samples_per_second=samples/(time.monotonic()-start),
                    reward=buf['reward'].mean((0,2)).cpu().tolist(),loss=float(np.mean(losses)),
                    kl=float(np.mean(kls)),gradient_peak=grad_peak,
                    torch_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,**env.summary())
        for i in range(4):
            writer.add_scalar(f'pose_{i+1:02d}/success_rate',report['success_rate'][i],update)
            writer.add_scalar(f'pose_{i+1:02d}/max_stable_seconds',report['max_stable_seconds'][i],update)
            writer.add_scalar(f'pose_{i+1:02d}/reward',report['reward'][i],update)
        for k in ('loss','kl','gradient_peak','samples_per_second','update_seconds'):
            writer.add_scalar(f'train/{k}',report[k],update)
        with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
        tmp=run/'status.tmp'; tmp.write_text(json.dumps(dict(status='running',pid=os.getpid(),**report),indent=2)+'\n')
        tmp.replace(run/'status.json')
        print(json.dumps(report),flush=True)
        if update%args.save_every==0 or update==args.iterations:
            save(update,f'update_{update:06d}.pt'); save(update,'latest.pt'); writer.flush()
    status=json.loads((run/'status.json').read_text()); status['status']='completed'
    (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    writer.close(); env.close()
except BaseException:
    import traceback,sys,os
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
