"""Measured capability evidence, distinct from the legacy functional score."""
import math

def review_capability(trace,outcomes):
    n=len(outcomes['complete']);episodes=[]
    for i in range(n):
        rows=[r for r in trace if r['valid'][i]]
        active=[r for r in rows if r['phase'][i] in (1,2,3)]
        balance=[r for r in rows if r['phase'][i]==2 and r['phase_time'][i]>=2.]
        groups={name:[] for name in ('forward','reverse','aw','dw','sa','sd','left','right','stop')}
        for r in balance:
            v,w=r['command'][i]
            name=('aw' if w>0 else 'dw') if v>0 and abs(w)>.015 else ('sa' if w>0 else 'sd') if v<0 and abs(w)>.015 else ('forward' if v>0 else 'reverse') if abs(v)>.005 else ('left' if w>0 else 'right') if abs(w)>.015 else 'stop'
            groups[name].append(r)
        summary={}
        for name,rs in groups.items():
            if rs:
                summary[name]=dict(seconds=len(rs)*.06,requested=[sum(r['command'][i][j] for r in rs)/len(rs) for j in range(2)],
                    measured=[sum(r['interval_motion'][i][j] for r in rs)/len(rs) for j in range(2)])
        target_speed={}
        for name,sign in (('forward',1),('reverse',-1)):
            rs=[r for r in balance if sign*r['command'][i][0]>=.749 and abs(r['command'][i][1])<.001]
            target_speed[name]=dict(samples=len(rs),seconds=len(rs)*.06,
                mean_speed=sum(sign*r['interval_motion'][i][0] for r in rs)/max(1,len(rs)),
                fraction_within_005=sum(abs(r['motion_feedback'][i][0]-sign*.75)<=.05 for r in rs)/max(1,len(rs)),
                tested=bool(rs))
        last=rows[-1]
        episodes.append(dict(index=i,complete=outcomes['complete'][i],failure=outcomes['failure'][i],
            settled_rise=any(r['rise_settled'][i] for r in rows),initial_motion_frame=last['initial_motion_frame'][i],
            preparation_seconds=last['preparation_elapsed'][i],preparation_speed_change=last['preparation_speed_change'][i],
            preparation_hip_change=last['preparation_hip_change'][i],groups=summary,target_speed=target_speed,
            wheel_measured_peak=max((abs(x) for r in active for x in r['wheel_measured'][i]),default=0.)))
    return dict(version=1,independent_of_legacy_precision=True,settled_rise=sum(e['settled_rise'] for e in episodes),
        target_speed_required=[-.75,.75],episodes=episodes,
        mirror_note='Aggregate classes are not matched trials; use paired initial-state/command probes for a symmetry verdict.')
