"""Check reference gating/sign/slew on CUDA; PhysX identity replay is separate."""
import json
import pathlib
import sys
import torch

sys.path.insert(0, str(pathlib.Path.cwd()))
from tools.probe_transition_speed import brake_reference

assert torch.cuda.is_available()
phase = torch.arange(5, device='cuda').repeat_interleave(7)
speed = torch.tensor([-1.1, -.75, -.03, 0., .03, .75, 1.1], device='cuda').repeat(5)
assert torch.equal(brake_reference(speed, phase, .06, 'identity'), speed)
for mode, active in [('rise_brake', 1), ('landing_brake', 3)]:
    value = speed.clone()
    for _ in range(40):
        new = brake_reference(value, phase, .06, mode)
        assert torch.equal(new[phase != active], speed[phase != active])
        assert (new.abs() <= value.abs()+1e-6).all()
        assert ((new-value).abs() <= .036001).all()
        assert (new*speed >= 0).all()
        value = new
    assert not value[phase == active].any()
assert torch.equal(speed, torch.tensor([-1.1, -.75, -.03, 0., .03, .75, 1.1], device='cuda').repeat(5))
print(json.dumps(dict(passed=True,device='cuda',phases=5,speeds=7,
    identity_exact=True,inactive_unchanged=True,no_sign_overshoot=True,
    source_input_unchanged=True,forty_steps_reach_zero=True,maximum_delta=.036)))
