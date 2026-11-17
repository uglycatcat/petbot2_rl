"""Choose each independent pose policy by actual original-MuJoCo outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); args=p.parse_args()
run=Path(args.run_dir); reports=sorted((run/'evaluations').glob('update_*.json'))
if not reports: raise RuntimeError('No MuJoCo evaluations to select from')
entries=[]
for path in reports:
    r=json.loads(path.read_text()); trajectory=np.load(path.with_suffix('.npz'))
    # Success first. For failures, preserve the most upright actually visited state for review.
    scores=[(bool(r['success'][i]),bool(r['final_stable'][i]),r['max_stable_s'][i],
             float(np.where(trajectory['eligible'][:,i],trajectory['up'][:,i],-1).max()),
             -r['first_success_s'][i] if r['success'][i] else 0)
            for i in range(4)]
    entries.append((r,scores))
selected=[max(range(len(entries)),key=lambda j:entries[j][1][i]) for i in range(4)]
loaded={}
for index in set(selected):
    loaded[index]=torch.load(entries[index][0]['checkpoint'],map_location='cpu',weights_only=False)
first=loaded[selected[0]]
model={name:value.clone() for name,value in first['model'].items()}
for pose,index in enumerate(selected):
    for name,value in model.items(): value[pose].copy_(loaded[index]['model'][name][pose])
selection=[dict(pose_id=f'pose_{i+1:02d}',checkpoint=entries[j][0]['checkpoint'],
                score=entries[j][1][i],success=entries[j][0]['success'][i]) for i,j in enumerate(selected)]
# This is an evaluation bundle, never an initialization/resume source for training.
bundle=dict(model=model,args=first['args'],iteration=[loaded[j]['iteration'] for j in selected],
            source_hash=first['source_hash'],code_hash=first['code_hash'],selection=selection,
            kind='evaluation_bundle_independent_policies')
torch.save(bundle,run/'best_per_pose.pt')
(run/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps(selection),flush=True)
