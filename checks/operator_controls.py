"""Real MuJoCo checks of ordinary control and operator/policy ownership."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import glfw
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from petbot_s2s.simulation import Simulation
from petbot_s2s.viewer import Keyboard


def run(output):
    result = dict(cases={}, method='MuJoCo physical steps plus explicit input-ownership checks')
    # A runtime that attempts to load any PyTorch network must fail this check.
    with patch('torch.jit.load', side_effect=AssertionError('Runtime PT loading forbidden')), \
         patch('torch.load', side_effect=AssertionError('Runtime checkpoint loading forbidden')):
        for name in ('cycle21_parent_b_r42', 'correction20_gate_r42'):
            sim = Simulation(ROOT/'policies'/name, frame=.4)
            c, p = sim.controller, sim.controller.progress
            c.set_policy_limits(.05, .15)
            # Restrict only policy acceleration; ordinary control must retain
            # the same command range, frame range and actuator authority.
            c.cfg.wheel_slew = .1
            c.cfg.frame_slew = .01
            c.set_command(.75, 0.)
            sim.step()
            assert abs(float(p.command[0, 0])-.75)<1e-6
            assert float(c.action.speed.min())>24.9
            for _ in range(82):
                sim.step()
            forward = sim.record()
            c.set_command(-.75, 0.)
            for _ in range(84):
                sim.step()
            reverse = sim.record()
            assert not c.failure_code and not c.armed
            assert forward['motion'][0]>.6 and reverse['motion'][0]<-.6
            assert np.max(np.abs(sim.data.actuator_force[5:]))<=.380001
            result['cases'][name] = dict(forward_measured=forward['motion'][0],
                reverse_measured=reverse['motion'][0], physical_frame_target=c.frame,
                policy_limits=c.policy_motion_limits.tolist(), ordinary_independent=True)

        sim = Simulation(ROOT/'policies/cycle21_parent_b_r42')
        c = sim.controller
        for _ in range(30):
            sim.step()
        keys = Keyboard()
        keys.event(glfw.KEY_Q, glfw.PRESS)
        c.set_frame_input(keys.frame_direction())
        for _ in range(150):
            sim.physics_step()
        up = sim.record()
        keys.event(glfw.KEY_Q, glfw.RELEASE)
        c.set_frame_input(keys.frame_direction())
        held = c.frame
        for _ in range(30):
            sim.step()
        assert c.frame == held and abs(held-.15)<1e-7
        up_settled = sim.record()
        elevation_up = float(np.arcsin(np.clip(sim.data.xmat[sim.root].reshape(3,3)[2,0],-1,1)))
        keys.event(glfw.KEY_E, glfw.PRESS)
        c.set_frame_input(keys.frame_direction())
        for _ in range(300):
            sim.physics_step()
        keys.event(glfw.KEY_E, glfw.RELEASE)
        c.set_frame_input(keys.frame_direction())
        for _ in range(30):
            sim.step()
        down = sim.record()
        elevation_down = float(np.arcsin(np.clip(sim.data.xmat[sim.root].reshape(3,3)[2,0],-1,1)))
        assert abs(c.frame+.15)<1e-7 and elevation_up>elevation_down+.1
        assert up_settled['four_fraction']>.9 and down['four_fraction']>.9
        assert abs(up['frame_target'][0]-up['frame_target'][1])<1e-7
        result['cases']['qe_physical'] = dict(targets=[held,c.frame],
            body_elevation_rad=[elevation_up,elevation_down],
            four_wheel_duty=[up_settled['four_fraction'],down['four_fraction']], release_holds=True)

        # An external Space request preserves the actual state and motor targets.
        c.set_policy_limits(.05, .15)
        c.set_command(.75, 0.)
        physical = (sim.data.qpos.copy(), sim.data.qvel.copy(), c.action.target.clone())
        c.toggle()
        assert np.array_equal(sim.data.qpos,physical[0]) and np.array_equal(sim.data.qvel,physical[1])
        assert torch.equal(c.action.target,physical[2]) and c.armed
        c.set_frame_input(1.)
        assert c.frame_input == 0. and not c.manual_frame_allowed
        c.route_command()
        assert abs(float(c.operator_command[0,0])-.75)<1e-7
        assert abs(float(c.progress.raw[0,0])-.05)<1e-7
        assert float(c.progress.destination[0])==0.
        # Cancel pending admission; ordinary limits and Q/E are restored.
        c.toggle()
        c.route_command()
        assert not c.armed and c.manual_frame_allowed
        assert abs(float(c.progress.raw[0,0])-.75)<1e-7
        result['cases']['ownership'] = dict(physical_state_preserved=True,
            operator_request_preserved=True, policy_input_separately_limited=True,
            qe_blocked_during_call=True, cancellation_restores_ordinary=True,
            default_landing_destination=True)
    result['passed'] = True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
