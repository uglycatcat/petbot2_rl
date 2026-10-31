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


def progress_increment(previous_best, current, invalid):
    """Pay for a bounded new task best once, including across height oscillations.

    A 5 cm approach deadband excludes the observed 0--2 cm rear-hop failures.
    Invalid terminal frames cannot earn new credit; earlier valid progress remains.
    """
    candidate=(current-.5).clamp_min(0.)
    best=torch.maximum(previous_best, candidate)
    best=torch.where(invalid,previous_best,best)
    return best-previous_best,best


def terminal_reward(success, failed, elapsed):
    # With gamma=1 and no per-step time tax, failure has the same price at every
    # time. Only successful completion has a small duration preference.
    return success.float()*(20.-.1*elapsed)-5.*failed.float()
