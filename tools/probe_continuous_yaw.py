"""Fixed-policy yaw feedback intervention; diagnostic only, no training changes."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--probe-mode',choices=['policy','feedback'],default='policy')
p.add_argument('--probe-kp',type=float,default=2.)
p.add_argument('--probe-ki',type=float,default=1.)
p.add_argument('--probe-diff-rate',type=float,default=1.)
options,remaining=p.parse_known_args()
assert '--evaluate' in remaining
assert 0<=options.probe_kp<=8 and 0<=options.probe_ki<=8 and 0<options.probe_diff_rate<=8
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_probe(env,folder):
    import torch
    from upright.continuous_task import ensure
    from upright.continuous import yaw_feedforward
    a=env.action_manager.get_term('moving');original=a.process_actions
    probe={'integral':torch.zeros(env.num_envs,device=env.device),'target':torch.zeros(env.num_envs,device=env.device)}
    def process(actions):
        previous=a.speed.clone();original(actions)
        if options.probe_mode=='policy':return
        progress=ensure(env);phase=progress.phase;clock=progress.phase_time
        active=(phase==2)&(clock>=.6)
        probe['integral'][phase<=1]=0.
        error=progress.command[:,1]-progress.motion[:,1]
        probe['integral'][active]=(probe['integral']+options.probe_ki*env.step_dt*error)[active].clamp(-1.5,1.5)
        desired=(yaw_feedforward(progress.command[:,1])+options.probe_kp*error+probe['integral']).clamp(-2.,2.)
        old=(previous[:,1]-previous[:,0])*.5
        target=old+(desired-old).clamp(-options.probe_diff_rate*env.step_dt,options.probe_diff_rate*env.step_dt)
        blend=torch.where(phase==2,((clock-.6)/.6).clamp(0,1),torch.where(phase==3,(1-clock/.6).clamp(0,1),0.))
        policy_half=(a.speed[:,1]-a.speed[:,0])*.5
        difference=policy_half*(1-blend)+target*blend
        common=a.speed.mean(-1);margin=(a.max_speed-common.abs()).clamp_min(0)
        difference=torch.maximum(torch.minimum(difference,margin),-margin)
        a.speed.copy_(torch.stack((common-difference,common+difference),-1))
        a.applied[:,1:]=a.speed/a.max_speed
        probe['target'].copy_(target)
    a.process_actions=process
    (folder/'probe.json').write_text(json.dumps(dict(diagnostic_only=True,options=vars(options),
        unchanged='fixed policy frame/common; unchanged task commands and duration',
        feedback='pose-difference EMA120ms; integral clamp1.5, halfdiff clamp2',
        script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))
    return probe


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    if source.count(old)!=1:raise ValueError('Evaluator changed; inspect probe injection')
    source=source.replace(old,new)
replace('        alive=torch.ones(n,device=args.device,dtype=torch.bool)',
        '        probe=install_probe(env,folder)\n        alive=torch.ones(n,device=args.device,dtype=torch.bool)')
replace('trace.append(dict(gravity=',"trace.append(dict(probe_integral=probe['integral'].tolist(),probe_half_target=probe['target'].tolist(),gravity=")
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
print('CONTINUOUS YAW DIAGNOSTIC ONLY',vars(options),flush=True)
exec(compile(source,'train_cycle.py','exec'),globals())
