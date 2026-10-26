"""Offline functional driving criteria; kept separate from legacy precision scores."""
import numpy as np

FUNCTIONAL_CONTRACT=dict(version='basic_remote_v1',settle_seconds=.6,
    minimum_segment_seconds=.6,command_deadband=[.005,.015],
    minimum_response_ratio=.2,maximum_response_ratio=2.5,
    required_response_fraction=.9,stop_excursion_m=.03,stop_heading_excursion_rad=.10)


def score_trajectories(rows,outcomes):
    a=lambda key:np.asarray([row[key] for row in rows])
    raw=a('raw');valid=a('valid')&(a('phase')==2)
    t=a('t');dt=float(t[1]-t[0]);pos=a('pos')[:,:,:2];heading=a('heading')
    motion=a('interval_motion');n=raw.shape[1];episodes=[]
    for i in range(n):
        change=np.ones(len(t),dtype=bool)
        change[1:]=(raw[1:,i]!=raw[:-1,i]).any(-1)|(valid[1:,i]!=valid[:-1,i])
        starts=np.flatnonzero(change);ends=np.r_[starts[1:],len(t)]
        checks=[];stop_checks=[];stop_distance=[];stop_heading=[];ratios=[]
        categories=set()
        for start,end in zip(starts,ends):
            if not valid[start,i]:continue
            g=np.arange(start,end);g=g[(t[g]-t[start])>=.6-1e-6]
            if len(g)*dt<.6-1e-6:continue
            command=raw[start,i];actual=motion[g,i].mean(0)
            for axis,deadband in enumerate((.005,.015)):
                if abs(command[axis])>deadband:
                    ratio=float(actual[axis]/command[axis]);ratios.append(ratio)
                    checks.append(.2<=ratio<=2.5)
                    categories.add((axis,int(np.sign(command[axis]))))
            if (command==0).all():
                distance=float(np.linalg.norm(pos[g,i]-pos[g[0],i],axis=-1).max())
                angle=heading[g,i]-heading[g[0],i]
                angle=float(np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max())
                stop_distance.append(distance);stop_heading.append(angle)
                stop_checks.append(distance<=.03 and angle<=.10)
        complete=bool(outcomes['complete'][i] and not outcomes['failure'][i])
        response_fraction=float(np.mean(checks)) if checks else None
        # Short/native episodes may not cover every category; expose coverage, never fill missing tests with passes.
        covered=all(x in categories for x in ((0,1),(0,-1),(1,1),(1,-1))) and bool(stop_checks)
        functional=complete and covered and response_fraction>=.9 and all(stop_checks)
        episodes.append(dict(complete=complete,coverage_complete=covered,
            response_checks=len(checks),response_fraction=response_fraction,
            response_ratio_min=min(ratios) if ratios else None,
            response_ratio_max=max(ratios) if ratios else None,
            stop_segments=len(stop_checks),stop_segments_passed=sum(stop_checks),
            max_stop_excursion_m=max(stop_distance) if stop_distance else None,
            max_stop_heading_excursion_rad=max(stop_heading) if stop_heading else None,
            functional_pass=bool(functional)))
    return dict(definition=FUNCTIONAL_CONTRACT,count=n,
        stable_completed=sum(e['complete'] for e in episodes),
        coverage_complete=sum(e['coverage_complete'] for e in episodes),
        functional_pass=sum(e['functional_pass'] for e in episodes),episodes=episodes)
