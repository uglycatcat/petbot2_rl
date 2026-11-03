"""Continuing auxiliary STOP objective, with no elapsed-time failure premium."""
import math


def discounted_failure_cost(max_rate, step_dt, gamma):
    """Exceed the worst infinite discounted running cost by a fixed margin.

    With running cost in [0, max_rate*dt], immediate failure cannot improve
    return by avoiding future running costs, even if later failure is certain.
    This bound is independent of episode age and the simulator reset horizon.
    """
    if not (math.isfinite(gamma) and 0 < gamma < 1 and max_rate > 0 and step_dt > 0):
        raise ValueError('Continuing STOP requires a finite discount below one')
    return max_rate*step_dt/(1-gamma)+3.
