"""Read actual stage/motion/joint trajectories before selecting another batch."""
import argparse,collections,json,math,pathlib,statistics
p=argparse.ArgumentParser();p.add_argument('evaluations',nargs='+',type=pathlib.Path);a=p.parse_args()
for folder in a.evaluations:
 result=json.loads((folder/'evaluation.json').read_text());rows=json.loads((folder/'trace.json').read_text());episodes=result['behavior']['episodes'];n=result['count']
 causes=collections.Counter();heading={1:[],3:[]};wheel_max=0.;wheel_exceeded=wheel_samples=0;frame_min=math.inf;frame_max=-math.inf
 for i,e in enumerate(episodes):
  causes[str(e['last_failure_code'])]+=1
  for stage,label in ((1,'rise'),(3,'landing')):
   if e['stages'][label]['samples']:heading[stage].append(e['stages'][label]['maximum_heading_deviation_deg'])
 for r in rows:
  for i,valid in enumerate(r['valid']):
   if not valid:continue
   wheel_max=max(wheel_max,*map(abs,r['wheel_measured'][i]));wheel_exceeded+=any(abs(v)>25. for v in r['wheel_measured'][i]);wheel_samples+=1
   frame_min=min(frame_min,*r['joint_position'][i][3:]);frame_max=max(frame_max,*r['joint_position'][i][3:])
 selected=[]
 successes=[i for i,e in enumerate(episodes) if e['complete']]
 successes.sort(key=lambda i:max(episodes[i]['stages'][s].get('maximum_heading_deviation_deg',0.) for s in ('rise','landing')))
 if successes:selected += [successes[len(successes)//2],successes[-1]]
 groups={}
 for i,e in enumerate(episodes):
  if not e['complete']:
   stage=next((s for s in ('handoff','landing','balance','rise','preparation') if e['stages'][s]['samples']), 'none')
   groups.setdefault((stage,e['last_failure_code']),i)
 selected+=list(groups.values())
 samples=[]
 keys=('t','phase','phase_time','raw','command','motion_feedback','gravity','joint_position','joint_velocity','position_target','wheel_target','wheel_measured','actuator_effort','transition_heading_error','geometry_ground','four_fraction','failure_code','interrupted')
 for i in dict.fromkeys(selected):
  active=[r for r in rows if r['valid'][i]];points={0,len(active)-1}
  for j in range(1,len(active)):
   if active[j]['phase'][i]!=active[j-1]['phase'][i]:points.update((max(0,j-1),j))
  points.update(range(max(0,len(active)-8),len(active),2))
  samples.append(dict(environment=i,outcome=episodes[i],trace=[{k:(active[j][k] if k=='t' else active[j][k][i]) for k in keys} for j in sorted(points)]))
 summary=dict(count=n,takeover=result['takeover'],preparation_failures=n-result['takeover'],complete=result['complete'],interrupted=result['interrupted'],functional_pass=result['functional']['functional_pass'],failure_codes=dict(causes),transition_heading_deg={str(k):dict(count=len(v),median=statistics.median(v) if v else None,max=max(v) if v else None) for k,v in heading.items()},wheel_measured_abs_max=wheel_max,wheel_measured_above_target_limit_fraction=wheel_exceeded/max(1,wheel_samples),frame_measured_range=[frame_min,frame_max],command_coverage=result['command_coverage'],selected_samples=samples,note='Target cap is 25rad/s; actual overshoot is reported separately. Sampled traces do not replace all-episode aggregates or operator acceptance.')
 (folder/'behavior_review.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(str(folder),json.dumps({k:v for k,v in summary.items() if k not in ('selected_samples','command_coverage','note')}))
