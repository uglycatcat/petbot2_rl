"""Fixed-policy diagnostic: express world-frame observation vectors locally.

Does not change the model, commands, reward, failure rules, or action bounds.
Artifacts are explicitly diagnostic and not formal checkpoint evaluations.
"""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--observation-frame-probe',choices=['world','local'],required=True)
options,remaining=p.parse_known_args()
assert '--evaluate' in remaining and '--checkpoint' in remaining
sys.argv=[sys.argv[0],*remaining]
sys.path.insert(0,str(pathlib.Path.cwd()))

def install_probe(env,folder):
    import torch
    from upright import cycle_task
    original=cycle_task.observations
    def observations(e):
        result=original(e)
        if options.observation_frame_probe=='world':return result
        h=result[:,3:5];unit=h/h.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        c,s=unit.unbind(-1)
        # Heading amplitude retains pitch information. Linear and angular
        # horizontal world velocities and both path errors share this rotation.
        for first in (3,5,8,27,29):
            x,y=result[:,first:first+2].clone().unbind(-1)
            result[:,first]=c*x+s*y
            result[:,first+1]=-s*x+c*y
        sy,cy=result[:,31:33].clone().unbind(-1)
        result[:,31]=sy*c-cy*s
        result[:,32]=cy*c+sy*s
        return result
    cycle_task.observations=observations
    (folder/'local_frame_probe.json').write_text(json.dumps(dict(diagnostic_only=True,
        frame=options.observation_frame_probe,source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        semantics='Per-observation wheel-ground travel frame, before observation history and feature bounds; local references, same full-cycle controls and evaluation.')))

source=pathlib.Path('train_cycle.py').read_text()
old='torch.manual_seed(args.seed);env=ManagerBasedRLEnv(cfg=cfg);wrapped=Adapter(env)'
assert source.count(old)==1
source=source.replace(old,"probe_folder=pathlib.Path(args.log_dir);probe_folder.mkdir(parents=True,exist_ok=True);install_probe(None,probe_folder)\n    "+old)
source=source.replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
source=source.replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
