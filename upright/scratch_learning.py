"""Bounded rise shaping for a random policy; never changes success or control."""
import torch

CONTRACT=dict(version='rise_progress_v2',progress_weight=4.,entry_reward=5.,
    preparation_potential_cap=.25,lift_height=.075,front_radius=.045,
    preparation_axis='negative_body_gravity_x',progress='new_peak_above_takeover_potential',
    rising_transition_and_smooth='multiply_by_actual_front_lift_fraction',
    repeat_posture_credit=False,physical_state_assignment=False,success_conditions_changed=False)

def rise_rewards(s,m,peak,dt):
    lifting=(m['phase']==1)|m['entered_balance']
    valid=lifting&~m['failure']
    rear=s['rear_fraction']
    lift=((s['front_z']-.045)/.075).clamp(0.,1.)*rear
    prepare=.25*(-s['g'][:,0]/.8).clamp(0.,1.)*rear
    potential=(prepare+lift)*(s['g'][:,2]<-.25)
    # An operator's pre-existing head/body angle is not policy progress.
    baseline=torch.where(m['switched'],potential,peak)
    new_peak=torch.where(valid,torch.maximum(baseline,potential),baseline)
    progress=torch.where(valid,(new_peak-baseline)/dt,0.)
    rising=m['phase']==1
    return new_peak,dict(rise_progress_reward=progress,rise_lift_fraction=lift,
        transition_track=torch.where(rising,m['transition_track']*lift,m['transition_track']),
        smooth_reward=torch.where(rising,m['smooth_reward']*lift,m['smooth_reward']))
