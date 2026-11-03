"""Finite full-task objective; no discounted parking-auxiliary assumptions."""
FAILURE_COST = 25.
SUCCESS_BONUS = 100.
TIME_COST_RATE = .2


def quality_terminates(invalid, evaluation, elapsed, timeout):
    # Training keeps collecting after an irrevocable quality violation, to
    # expose later physical milestones. It can never receive success credit.
    # Independent evaluation rejects immediately, using the original rule.
    return invalid & (evaluation | (elapsed >= timeout))


def reward(progress, running_cost, physical_cost, action_change_squared,
           completed, failed, dt):
    # Restore the original movement/contact and action-change costs that the
    # restarted objective previously overwrote. No cost is scaled by an
    # unobserved remaining-time failure premium. gamma must be exactly one.
    return (progress - running_cost - physical_cost - .02*action_change_squared
            - TIME_COST_RATE*dt + SUCCESS_BONUS*completed.float()
            - FAILURE_COST*failed.float())
