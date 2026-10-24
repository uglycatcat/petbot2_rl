"""Fixed-v10 diagnostic of slower frame-position targets during stable balance."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--balance-frame-slew',type=float,choices=[.1,.3,3.],default=3.)
options,remaining=p.parse_known_args();assert '--evaluate' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_frame_probe(env,folder):
    from upright.continuous_task import ensure
    a=env.action_manager.get_term('moving');original=a.process_actions
    def process(actions):
        previous=a.target[:,3].clone();original(actions)
        p=ensure(env);mask=(p.phase==2)&(p.phase_time>=1.2)
        limit=options.balance_frame_slew*env.step_dt
        target=previous+(a.target[:,3]-previous).clamp(-limit,limit)
        a.target[mask,3:]=target[mask,None]
        a.applied[:,0]=2*(a.target[:,3]-a.lo)/(a.hi-a.lo)-1
    a.process_actions=process
    (folder/'frame_probe.json').write_text(json.dumps(dict(diagnostic_only=True,balance_frame_slew=options.balance_frame_slew,
        active='phase2 clock>=1.2',source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Fixed evaluator changed'
    source=source.replace(old,new)
replace('obs=wrapped.get_observations();n=args.num_envs','install_frame_probe(env,folder);obs=wrapped.get_observations();n=args.num_envs')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
