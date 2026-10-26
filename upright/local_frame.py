"""Yaw-invariant travel-frame encoding for the existing observation layout."""
import torch

LOCAL_FRAME_CONTRACT=dict(version='wheel_ground_local_v1',
    horizontal_vector_offsets=[3,5,8,27,29], reference_yaw_sin_cos=[31,32],
    history='encode_each_sample_before_history_and_bounds',
    unchanged='body gravity, joints, commands, vertical velocities and task clocks',
    balance_local_transition=dict(start_phase2_seconds=1.2,end_phase2_seconds=1.8,
                                  retain_through_landing=True,original_rise=True))


def ground_frame_features(features: torch.Tensor, heading: torch.Tensor, frame: str) -> torch.Tensor:
    if frame=='world':return features
    if frame=='balance_local':
        blend=(features[:,43]>.5)*((features[:,48]*4.-1.2)/.6).clamp(0.,1.)
        blend=torch.where((features[:,44]>.5)|(features[:,45]>.5),1.,blend)
        heading=heading*blend
    elif frame!='local':raise ValueError('Unknown observation frame')
    result=features.clone()
    c,s=heading.cos(),heading.sin()
    for first in (3,5,8,27,29):
        x,y=features[:,first],features[:,first+1]
        result[:,first]=c*x+s*y
        result[:,first+1]=-s*x+c*y
    sy,cy=features[:,31],features[:,32]
    result[:,31]=sy*c-cy*s
    result[:,32]=cy*c+sy*s
    return result
