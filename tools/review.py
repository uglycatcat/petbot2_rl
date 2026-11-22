"""Read TensorBoard first, then summarize actual evaluation trajectories."""
import argparse
from pathlib import Path
import json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
p.add_argument('--evaluation',type=Path)
p.add_argument('--output',type=Path)
p.add_argument('--max-step',type=int,help='Only read training curves up to this checkpoint index')
p.add_argument('--training-only',action='store_true',help='Read curves before dispatching physical evaluations')
a=p.parse_args()
run=a.run;evaluation=a.evaluation or run/'eval';output=a.output or run
output.mkdir(parents=True,exist_ok=True)
acc=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});acc.Reload()
curves={}
for tag in acc.Tags().get('scalars',[]):
    events=acc.Scalars(tag)
    if a.max_step is not None:events=[x for x in events if x.step<=a.max_step]
    if not events:continue
    vals=np.asarray([x.value for x in events],dtype=float)
    curves[tag]={'points':len(vals),'first':float(vals[0]),'last':float(vals[-1]),
        'last100_mean':float(vals[-100:].mean()),'min':float(vals.min()),'max':float(vals.max())}
(output/'tensorboard_review.json').write_text(json.dumps(curves,indent=2))
if a.training_only:
    print(json.dumps({'curve_tags':len(curves),'max_step':a.max_step}))
    raise SystemExit(0)
result=json.loads((evaluation/'evaluation.json').read_text())
t=np.load(evaluation/'evaluation_traces.npz')
valid=t['alive'];success=(t['success']&valid).any(0)
count=t['top'].sum(-1)
selected=[]
if success.any():
    ids=np.flatnonzero(success)
    ends=np.argmax(t['success'][:,ids]&valid[:,ids],axis=0)
    selected.extend([int(ids[0]),int(ids[-1]),int(ids[ends.argmin()]),int(ids[ends.argmax()])])
if (~success).any():
    failures=np.flatnonzero(~success)
    selected.append(int(failures[np.argmax(np.where(valid[:,failures],count[:,failures],0).max(0))]))
selected=list(dict.fromkeys(selected))
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
        'tensorboard_review':'tensorboard_review.json','training_curve_max_step':a.max_step}
if success.any():
    event=np.argmax(t['success']&valid,axis=0)[success];ids=np.flatnonzero(success)
    times=(event+1)*result['trace_dt']
    review['success_statistics']={'min_time_s':float(times.min()),'mean_time_s':float(times.mean()),
        'max_time_s':float(times.max()),'minimum_wheel_support_N':float(t['forces'][event,ids,:,2].min()),
        'maximum_wheel_bottom_error_m':float(np.abs(t['bottom'][event,ids]-result['step_height']).max())}
(output/'trajectory_review.json').write_text(json.dumps(review,indent=2))
if not review['all_finite']:raise RuntimeError('Non-finite physical evaluation trace')
print(json.dumps({'successes':int(success.sum()),'episodes':len(success),'samples':selected,'curve_tags':len(curves)}))
