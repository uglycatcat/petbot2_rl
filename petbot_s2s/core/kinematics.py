"""Flat-ground wheel geometry from body gravity and actual frame encoders.

The rear wheels define the supporting plane for this upright task. No contact
force, world position, or simulated body height is used by this calculation.
Dimensions are the immutable MJCF joint offsets; wheel radii are the existing
drive/model reference (front45mm, rear30mm). This is geometry, not a force sensor.
"""
import torch

CONTRACT = dict(version='body_imu_frame_fk_v1', front_radius=.045, rear_radius=.030,
    axle_offsets=[[.18000,.01204,.01372],[.18000,-.01204,.01372]],
    ground_gap_tolerance=.008, rear_height_tolerance=.010,
    grounded_dwell=.12, support_assumption='rear_wheels_on_flat_ground')


def wheel_geometry(gravity, frame):
    c,s=frame.cos(),frame.sin()
    delta=torch.stack((.18*c+.01372*s,
        frame.new_tensor([.01204,-.01204]).expand_as(frame),
        -.18*s+.01372*c),-1)
    up=-gravity
    gap=(delta*up[:,None,:]).sum(-1)-.015
    rear_local=frame.new_tensor([-.07979,-.04538])
    rear_x=.01000+c*rear_local[0]+s*rear_local[1]
    rear_z=-.00807-s*rear_local[0]+c*rear_local[1]
    rear_y=frame.new_tensor([.05000,-.05000]).expand_as(frame)
    rear=torch.stack((rear_x,rear_y,rear_z),-1)
    rear_height=(rear*up[:,None,:]).sum(-1)
    rear_skew=(rear_height[:,0]-rear_height[:,1]).abs()
    grounded=(gap.abs().amax(-1)<CONTRACT['ground_gap_tolerance']) & (rear_skew<CONTRACT['rear_height_tolerance'])
    return gap, rear_skew, grounded
