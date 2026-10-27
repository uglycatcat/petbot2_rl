"""Ground travel frame from the wheel-lateral axis, independent of frame pitch."""
import torch
from .moving import projected_heading_rate


def lateral_axis(quaternion):
    x,y,z,w=quaternion.unbind(-1)
    return torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)


def wheel_heading(quaternion, world_velocity):
    """Return unit ground forward, heading and its exact projected angular rate.

    PetBot's frame and wheel hinges are parallel to body Y. Their rotations do
    not change that lateral axis. Quaternion order follows the local xyzw API.
    Position and linear velocity reference points are deliberately unchanged.
    """
    lateral=lateral_axis(quaternion)
    ground=torch.stack((lateral[:,1],-lateral[:,0]),-1)
    ground=ground/ground.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    return ground,torch.atan2(ground[:,1],ground[:,0]),projected_heading_rate(lateral,world_velocity[:,3:])
