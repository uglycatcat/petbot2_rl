"""Pure scheduling and selection rules for the independent three-track search."""
import copy
import math


def can_admit(free, reservations, own_used, candidate_peak, reserve):
    # Active jobs may still grow from their currently observed allocation to their reserved peak.
    growth=sum(max(0,peak-own_used.get(name,0)) for name,peak in reservations.items())
    return free-growth >= candidate_peak+reserve


def next_config(initial, history, lane_number, height_ceiling=.30):
    cfg=copy.deepcopy(initial);trial=len(history)
    cfg['seed']=initial['seed']+1009*trial
    track=cfg['research_track']
    if not history:return cfg
    last=history[-1];cfg['iterations']=1500 if last.get('success_rate',0)<.5 else 1000
    if track=='diverse':
        cfg['control_dt']=[.04,.05,.06][(trial+lane_number)%3]
        cfg['top_reward_mode']=['distinct_wheels','simultaneous_peak'][(trial+lane_number)%2]
        cfg['initial_std']=[.65,.85,.5][trial%3]
        cfg['entropy_coef']=[.01,.015,.005][trial%3]
    elif track=='quality':
        # Compare from-scratch solutions; never use a successful policy as a teacher/initial state.
        factor=.5 if last.get('success_rate',0)<.9 else [1.,1.4,.8][trial%3]
        for k in ('head_default_weight','approach_frame_weight','action_rate_weight','joint_speed_weight'):
            cfg[k]=initial.get(k,0.)*factor
        cfg['entropy_coef']=.005 if last.get('success_rate',0)>=.9 else .01
    elif track=='height':
        previous=last['step_height'];rate=last.get('success_rate',0)
        if rate>=.9:
            cfg['step_height']=min(height_ceiling,round(previous+.02,3))
        else:
            # Repeat an unsuccessful height with another seed before refining the bracket.
            consecutive=sum(r['step_height']==previous and r.get('success_rate',0)<.9 for r in history[-2:])
            passed=[r['step_height'] for r in history if r.get('success_rate',0)>=.9]
            floor=max(passed,default=.07)
            cfg['step_height']=previous
            if consecutive>=2 and previous-floor>.0051:
                cfg['step_height']=round(round(((previous+floor)/2)/.005)*.005,3)
        cfg['initial_std']=[.65,.85,.5][trial%3]
        cfg['entropy_coef']=[.01,.015,.005][trial%3]
        cfg['top_reward_mode']=['distinct_wheels','simultaneous_peak'][(trial+lane_number)%2]
    return cfg


def descriptor_distance(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))


def quality_score(metrics, baseline):
    current=metrics['successful_episodes'];reference=baseline['successful_episodes']
    keys=('head_abs_mean_rad','far_frame_abs_mean_rad','bounded_target_rate_rms_per_s')
    ratios=[current[k]['mean']/max(reference[k]['mean'],1e-6)
            for k in keys if current.get(k) and reference.get(k)]
    return sum(ratios)/len(ratios) if ratios else float('inf')
