"""Read TensorBoard scalar evidence; run with container training dependencies."""
import argparse,json,math,pathlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('log_dirs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();out={}
for name in a.log_dirs:
 e=EventAccumulator(name,size_guidance={'scalars':0});e.Reload();scalars={}
 for tag in e.Tags()['scalars']:
  xs=e.Scalars(tag);values=[x.value for x in xs]
  if not values:continue
  scalars[tag]={'last_step':xs[-1].step,'first':values[0],'last':values[-1],'min':min(values),'max':max(values),'mean_last100':sum(values[-100:])/len(values[-100:]),'finite':all(math.isfinite(x) for x in values)}
 out[name]=scalars
pathlib.Path(a.output).write_text(json.dumps(out,indent=2))
for name,d in out.items():
 print(name,'finite',all(v['finite'] for v in d.values()))
 for key in ['Upright/phase2','Upright/episode_lifted','Upright/episode_success','Upright/front_height','Policy/mean_std','Train/mean_reward']:
  if key in d:print(key,d[key])
