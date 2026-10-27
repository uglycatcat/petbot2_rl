"""Independent per-phase physical review, with coverage separate from success."""
import numpy as np


def review_trajectories(rows,outcomes):
    cache={}
    def a(key):
        if key not in cache:cache[key]=np.asarray([row[key] for row in rows])
        return cache[key]
    valid=a('valid');phase=a('phase');n=valid.shape[1]
    raw=a('raw');command=a('command');actual=a('motion_feedback')
    yaw=np.abs(a('transition_heading_error'))
    effort=np.abs(a('actuator_effort'))
    wheels=np.abs(a('wheel_target'))
    episodes=[]
    for i in range(n):
        stages={}
        for number,name in ((0,'preparation'),(1,'rise'),(2,'balance'),(3,'landing'),(4,'handoff')):
            mask=valid[:,i]&(phase[:,i]==number)
            if not mask.any():stages[name]=dict(samples=0);continue
            stages[name]=dict(samples=int(mask.sum()),
                requested_abs_max=np.abs(raw[mask,i]).max(0).tolist(),
                command_abs_max=np.abs(command[mask,i]).max(0).tolist(),
                measured_abs_max=np.abs(actual[mask,i]).max(0).tolist(),
                command_projection_mae=np.abs(raw[mask,i]-command[mask,i]).mean(0).tolist(),
                wheel_target_abs_max=float(wheels[mask,i].max()),
                wheel_saturation_fraction=float((wheels[mask,i]>=24.99).any(-1).mean()),
                actuator_effort_abs_max=effort[mask,i].max(0).tolist())
            if number in (1,3):
                stages[name]['maximum_heading_deviation_deg']=float(np.rad2deg(yaw[mask,i].max()))
                stages[name]['net_heading_deviation_deg']=float(np.rad2deg(a('transition_heading_error')[mask,i][-1]))
        mask=valid[:,i]
        physical=a('four_fraction')[:,i]>=.9
        geometry=a('geometry_ground')[:,i]
        episodes.append(dict(complete=bool(outcomes['complete'][i]),
            interrupted=bool(outcomes.get('interrupted',[False]*n)[i]),
            last_failure_code=int(a('failure_code')[mask,i][-1]),stages=stages,
            geometry_true_physics_false_samples=int((mask&geometry&~physical).sum()),
            geometry_false_physics_true_samples=int((mask&~geometry&physical).sum())))
    return dict(definition='phase_physical_review_v1',count=n,episodes=episodes,
        note='Geometry/contact disagreement is diagnostic, not a claim of detector accuracy; inspect dwell and dynamic samples.')
