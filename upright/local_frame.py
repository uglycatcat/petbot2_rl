"""Yaw-invariant travel-frame encoding for the existing observation layout."""
import torch

LOCAL_FRAME_CONTRACT=dict(version='wheel_ground_local_v1',
    horizontal_vector_offsets=[3,5,8,27,29], reference_yaw_sin_cos=[31,32],
    history='encode_each_sample_before_history_and_bounds',
    unchanged='body gravity, joints, commands, vertical velocities and task clocks')


def ground_frame_features(features: torch.Tensor, heading: torch.Tensor, frame: str) -> torch.Tensor:
    if frame=='world':return features
    if frame!='local':raise ValueError('Unknown observation frame')
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
