"""Read TensorBoard first, then summarize actual evaluation trajectories."""
import argparse
from pathlib import Path
import json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run
acc=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});acc.Reload()
curves={}
for tag in acc.Tags().get('scalars',[]):
    events=acc.Scalars(tag)
    if not events:continue
    vals=np.asarray([x.value for x in events],dtype=float)
    curves[tag]={'points':len(vals),'first':float(vals[0]),'last':float(vals[-1]),
        'last100_mean':float(vals[-100:].mean()),'min':float(vals.min()),'max':float(vals.max())}
(run/'tensorboard_review.json').write_text(json.dumps(curves,indent=2))
result=json.loads((run/'eval/evaluation.json').read_text())
t=np.load(run/'eval/evaluation_traces.npz')
valid=t['alive'];success=(t['success']&valid).any(0)
count=t['top'].sum(-1)
selected=[]
if success.any():selected.append(int(np.flatnonzero(success)[0]))
if (~success).any():
    failures=np.flatnonzero(~success)
    selected.append(int(failures[np.argmax(np.where(valid[:,failures],count[:,failures],0).max(0))]))
samples=[]
for i in selected:
    frames=np.flatnonzero(valid[:,i]);last=int(frames[-1])
    first_success=np.flatnonzero(t['success'][:,i]&valid[:,i])
    event=int(first_success[0]) if len(first_success) else last
    samples.append({'episode':i,'success':bool(success[i]),'duration_s':float((last+1)*result['trace_dt']),
        'event_time_s':float((event+1)*result['trace_dt']),
        'root_first':t['root_pose'][0,i].tolist(),'root_event':t['root_pose'][event,i].tolist(),
        'wheel_positions_event':t['wheel_pos'][event,i].tolist(),
        'wheel_bottoms_event':t['bottom'][event,i].tolist(),
        'wheel_support_forces_event':t['forces'][event,i,:,2].tolist(),
        'target_min':t['targets'][frames,i].min(0).tolist(),'target_max':t['targets'][frames,i].max(0).tolist(),
        'joint_position_min':t['joint_pos'][frames,i].min(0).tolist(),
        'joint_position_max':t['joint_pos'][frames,i].max(0).tolist(),
        'joint_velocity_min':t['joint_vel'][frames,i].min(0).tolist(),
        'joint_velocity_max':t['joint_vel'][frames,i].max(0).tolist(),
        'max_simultaneous_top':int(count[frames,i].max()),'outside':bool(t['outside'][event,i])})
review={'result':result,'representative_samples':samples,'all_finite':all(np.isfinite(t[k]).all() for k in t.files),
        'scope':'Observed trajectories; no posture, smoothness or parking acceptance added.',
        'tensorboard_review':'tensorboard_review.json'}
(run/'trajectory_review.json').write_text(json.dumps(review,indent=2))
if not review['all_finite']:raise RuntimeError('Non-finite physical evaluation trace')
print(json.dumps({'successes':int(success.sum()),'episodes':len(success),'samples':selected,'curve_tags':len(curves)}))
