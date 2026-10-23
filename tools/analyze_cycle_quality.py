"""Offline re-scoring only; never changes trajectories or original qualification."""
import argparse,json,pathlib,collections
p=argparse.ArgumentParser();p.add_argument('batch_dir',type=pathlib.Path);a=p.parse_args()
state=json.loads((a.batch_dir/'eval_status.json').read_text());groups={};runs={}
for key,job in state.items():
    if job['status']!='complete':continue
    r=job['result'];rows=json.loads((a.batch_dir/('eval_'+key)/'trace.json').read_text());n=r['count']
    count=[0]*n;good=[0]*n;balanced=[0]*n;steady_count=[0]*n;steady_good=[0]*n;last=[None]*n;age=[0.]*n
    speed_good=[0]*n;yaw_good=[0]*n;speed_error=[0.]*n;yaw_error=[0.]*n
    previous_time=0.
    for row in rows:
        dt=row['t']-previous_time;previous_time=row['t']
        for i in range(n):
            raw=row['raw'][i]
            age[i]=age[i]+dt if last[i]==raw else 0.;last[i]=raw
            if not(row['valid'][i] and row['phase'][i]==2 and row['phase_time'][i]>.5):continue
            v,w=raw;ok=abs(row['forward'][i]-v)<.005+.25*abs(v) and abs(row['heading_rate'][i]-w)<.015+.25*abs(w)
            speed_good[i]+=abs(row['forward'][i]-v)<.005+.25*abs(v);yaw_good[i]+=abs(row['heading_rate'][i]-w)<.015+.25*abs(w)
            speed_error[i]+=abs(row['forward'][i]-v);yaw_error[i]+=abs(row['heading_rate'][i]-w)
            count[i]+=1;good[i]+=ok;balanced[i]+=row['balanced'][i]
            if age[i]>=.5:steady_count[i]+=1;steady_good[i]+=ok
    track=[c>=20 and g/c>=.8 for c,g in zip(count,good)]
    stable=[c>=20 and b/c>=.9 for c,b in zip(count,balanced)]
    # New requirement retains tracking and balance thresholds, removes only the
    # former touchdown speed/force gates; handoff includes safe completed cycle.
    qualified=[h and t and b for h,t,b in zip(r['outcomes']['complete'],track,stable)]
    d=dict(count=n,handoff=r['handoff'],original_qualified=r['qualified'],tracking_pass=sum(track),balance_pass=sum(stable),revised_qualified=sum(qualified),steady_tracking_pass=sum(c>=20 and g/c>=.8 for c,g in zip(steady_count,steady_good)))
    d.update(speed_tracking_pass=sum(c>=20 and g/c>=.8 for c,g in zip(count,speed_good)),yaw_tracking_pass=sum(c>=20 and g/c>=.8 for c,g in zip(count,yaw_good)),
             tracking_episode_count=sum(c>0 for c in count),speed_mae_sum=sum(e/c for e,c in zip(speed_error,count) if c),yaw_mae_sum=sum(e/c for e,c in zip(yaw_error,count) if c))
    runs[key]=d;parts=key.split('_');group='_'.join(parts[-2:])+'_'+parts[-3][0];g=groups.setdefault(group,collections.Counter());g.update(d)
result=dict(definition='Offline phase2 after0.5s; revised qualification keeps80% tracking and90% balance plus completed handoff; excludes touchdown force/speed only. Steady diagnostic excludes0.5s after each raw-command change and is NOT the acceptance criterion.',groups=groups,runs=runs)
(a.batch_dir/'quality_review.json').write_text(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))
