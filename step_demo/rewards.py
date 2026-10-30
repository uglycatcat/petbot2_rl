"""Bounded task potentials; simulation truth is used for rewards only."""
import torch


def edge_potential(forward, front_gap, front_min_z, height):
    # Clearance far from the edge provides no shaping. Returning the wheel to
    # the floor gives the height credit back, instead of retaining a lift bonus.
    proximity = ((.10-front_gap)/.10).clamp(0., 1.)
    clearance = (front_min_z/(height+.005)).clamp(0., 1.)
    return 10.*forward.clamp(-.05, .65) + (proximity*clearance).sum(-1)


def potential_difference(previous, current, gamma, terminal):
    # Match PPO's discount and close the potential at *all* episode endings.
    # A discounted closed motion cycle cannot farm shaping reward.
    current = torch.where(terminal, torch.zeros_like(current), current)
    return gamma*current-previous, current
