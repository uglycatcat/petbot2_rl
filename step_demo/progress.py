"""Distance memory across brief contact uncertainty; never a support substitute."""
import torch


SUPPORT_MEMORY_S = .06


def drive_memory(start, gap, forward, ready, pose_valid, near_top, dt):
    gap = torch.where(ready, 0., gap+dt)
    keep = pose_valid & near_top & (gap <= SUPPORT_MEMORY_S+1e-7)
    start = torch.where(ready & torch.isnan(start), forward, start)
    start = torch.where(ready | keep, start, float('nan'))
    return start, gap
