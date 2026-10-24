"""Fixed-v8 policy diagnostic of body-forward versus wheel-lateral heading."""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--heading-probe',choices=['baseline','lateral'],default='baseline')
options,remaining=p.parse_known_args();assert '--evaluate' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_heading_probe(env,folder):
    import torch
    from upright import continuous_task,cycle_task
    from upright.moving import projected_heading_rate
    original=cycle_task.state
    def state(e):
        s=original(e)
        s['legacy_yaw']=s['yaw'].clone();s['legacy_forward']=s['forward'].clone();s['legacy_heading_rate']=s['heading_rate'].clone()
        x,y,z,w=s['quat'].unbind(-1)
        lateral=torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
        projected=torch.stack((lateral[:,1],-lateral[:,0]),-1)
        unit=projected/projected.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        s['axle_yaw']=torch.atan2(unit[:,1],unit[:,0])
        s['axle_forward']=(s['vel'][:,:2]*unit).sum(-1)
        s['axle_heading_rate']=projected_heading_rate(lateral,s['vel'][:,3:])
        if options.heading_probe=='lateral':
            s['heading']=unit*s['heading'].norm(dim=-1,keepdim=True)
            s['yaw']=s['axle_yaw'];s['forward']=s['axle_forward'];s['heading_rate']=s['axle_heading_rate']
        return s
    cycle_task.state=state;continuous_task.state=state
    (folder/'heading_probe.json').write_text(json.dumps(dict(diagnostic_only=True,mode=options.heading_probe,
        definition='ground travel heading perpendicular to horizontal wheel-lateral axis; common symmetric pitch joints preserve lateral direction',
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Evaluator changed'
    source=source.replace(old,new)
replace('obs=wrapped.get_observations();n=args.num_envs','install_heading_probe(env,folder);obs=wrapped.get_observations();n=args.num_envs')
replace('trace.append(dict(yaw_feedback_integral=',"trace.append(dict(legacy_yaw=s['legacy_yaw'].tolist(),legacy_forward=s['legacy_forward'].tolist(),legacy_heading_rate=s['legacy_heading_rate'].tolist(),axle_yaw=s['axle_yaw'].tolist(),axle_forward=s['axle_forward'].tolist(),axle_heading_rate=s['axle_heading_rate'].tolist(),yaw_feedback_integral=")
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
