"""Training-only reward state; never part of the policy sensor interface."""
import torch


CRITIC_STATE_FIELDS = (
    'approach_credit', 'left_clear_credit', 'right_clear_credit',
    'left_rear_credit', 'right_rear_credit', 'home_credit', 'drive_credit', 'stop_credit',
    'elapsed_fraction', 'phase_fraction',
    'left_front_support', 'right_front_support', 'left_rear_support', 'right_rear_support',
    'left_front_landed', 'right_front_landed', 'drive_valid', 'drive_fraction',
    'support_gap_fraction', 'home_hold_fraction', 'stop_hold_fraction', 'near_top',
)


def reward_state(*, credit, elapsed, timeout, phase, supported_top, front_landed,
                 forward, drive_start, drive_distance, support_gap, home_hold,
                 stop_time, stop_hold, near_top):
    """Expose finite, bounded history that affects future reward and termination.

    Missing drive origins have an explicit validity bit and zero distance.
    The returned 22 channels contain no action, controller or acceptance changes.
    """
    caps = credit.new_tensor([6., 1., 1., 1.5, 1.5, 2., 2., 2.])
    valid = torch.isfinite(drive_start)
    distance = torch.where(valid, (forward-drive_start)/drive_distance, 0.).clamp(-1., 2.)
    scalars = torch.stack((valid.float(), distance, (support_gap/.06).clamp(0., 2.),
                          (home_hold/.2).clamp(0., 1.), (stop_time/stop_hold).clamp(0., 1.),
                          near_top.float()), dim=-1)
    return torch.cat(((credit/caps).clamp(0., 1.),
                      (elapsed/timeout).clamp(0., 1.)[:, None],
                      (phase.float()/4.).clamp(0., 1.)[:, None],
                      supported_top.float(), front_landed.float(), scalars), dim=-1)
