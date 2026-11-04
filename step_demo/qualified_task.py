"""Round14 bounded progress with strict flight eligibility and reachable STOP.

No time tax or remaining-time failure premium. No controller or phase input
is added to the actor. These quantities are reward-only simulation truth.
"""
import torch


def certified_front_landing(takeoff_before_top, flight_count, landed_top,
                            returned_floor, front_hit, top_contact):
    """Certify the requested physical sequence, independent of wheel AABB arc.

    Real debounced takeoff must PRECEDE the first top contact. Top landing
    requires the existing force/geometry/debounce check; face/edge collision
    and floor return remain forbidden. A wheel rolling up cannot qualify.
    """
    return (takeoff_before_top & (flight_count == 1) & landed_top
            & ~returned_floor & ~front_hit[:, None] & top_contact)


def stopping_progress(speed, wheel_request, legal_stop):
    # Both terms have useful slope near the original .02 m/s acceptance.
    # The environment pays their component-wise historical maximum once.
    return (4.*torch.exp(-speed/.02)
            +2.*torch.exp(-wheel_request/(.02/.03)))*legal_stop
