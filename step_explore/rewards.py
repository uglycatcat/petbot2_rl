"""Episode-progress reward based on simultaneous physical wheel support."""
import torch

TOP_PEAK_VALUES = (0.0, 3.0, 6.0, 12.0, 18.0)

def simultaneous_top_bonus(top, previous_peak, levels):
    peak = torch.maximum(previous_peak, top.sum(dim=-1))
    return levels[peak] - levels[previous_peak], peak


def motion_costs(angles, angular_velocity, action, previous_action, edge_distance, dt):
    """Soft preferences in optional quality trials; units integrated over physical seconds.

    Frame preference fades to zero within10cm of the nearest wheel reaching the edge.
    It does not force posture or change termination. A10/s bounded-command slew and
    a10rad/s joint speed each have unit normalized squared rate.
    """
    far=((edge_distance-.10)/.25).clamp(0.,1.)
    return torch.stack((angles[:,0].square(),
        far*angles[:,1:3].square().mean(-1),
        ((action-previous_action)/(10.*dt)).square().mean(-1),
        (angular_velocity/10.).square().mean(-1)),dim=-1)*dt
