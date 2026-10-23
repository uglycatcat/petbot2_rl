"""Four-wheel open-loop drive calibration and read-only runtime actuator audit."""
import argparse
import itertools
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
try:
    import torch
    import warp as wp
    from isaaclab.envs import ManagerBasedRLEnv
    from upright.cycle_task import CycleEnvCfg, ensure, state, tensor

    cases = list(itertools.product([-.05/.03, 0., .05/.03], [-1.5, -.5, -.25, 0., .25, .5, 1.5], [-.15, 0., .15]))
    cfg = CycleEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = len(cases)
    cfg.sim.device = a.device
    cfg.speed_limit = .3
    cfg.yaw_limit = 3.
    cfg.wheel_mode = 'common_diff'
    cfg.wheel_slew = 480.
    cfg.frame_slew = 3.
    cfg.terminations.complete = None
    cfg.terminations.failure = None
    cfg.terminations.timeout = None
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    robot = env.scene['robot']
    action = env.action_manager.get_term('moving')
    progress = ensure(env)
    progress.active[:] = True
    progress.phase[:] = 4
    progress.external_mode[:] = True
    values = torch.tensor(cases, device=env.device)
    progress.destination[:] = values[:, 2]
    progress.set_command(torch.stack((values[:, 0]*.03, values[:, 1]*.6), -1))
    metadata = dict(diagnostic_only=True, policy_used=False, fixed_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    joints=list(robot.joint_names), cases=cases, fields={})
    for name in ('joint_stiffness', 'joint_damping', 'joint_armature', 'joint_friction_coeff',
                 'joint_dynamic_friction_coeff', 'joint_viscous_friction_coeff', 'joint_effort_limits'):
        value = getattr(robot.data, name, None)
        if value is not None:
            metadata['fields'][name] = tensor(value)[0].cpu().tolist()
    # Query the backend as well as the public data cache, where available.
    metadata['backend'] = {}
    for name in ('get_dof_dampings', 'get_dof_stiffnesses', 'get_dof_armatures', 'get_dof_friction_properties', 'get_dof_max_forces'):
        function = getattr(robot.root_view, name, None)
        if function is not None:
            value = function()
            value = wp.to_torch(value) if isinstance(value, wp.array) else tensor(value)
            metadata['backend'][name] = value[0].cpu().tolist()
    rows = []
    for tick in range(134):
        env.step(torch.zeros(len(cases), 3, device=env.device))
        s = state(env)
        if tick < 84:
            continue
        rows.append(dict(t=(tick+1)*env.step_dt, forward=s['forward'].tolist(), yaw=s['heading_rate'].tolist(),
                         wheel=s['wheel'].tolist(), target=action.speed.tolist(), four=s['four_fraction'].tolist(),
                         rear=s['rear_fraction'].tolist(), body=s['body_history'].tolist(),
                         torque_estimate=tensor(robot.data.applied_torque)[:, action.wheel_ids].tolist()))
    result = dict(metadata=metadata, trace=rows)
    output = pathlib.Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(metadata, indent=2))
    print('DRIVE_RESPONSE_DONE', len(cases), len(rows), flush=True)
    env.close()
except BaseException:
    import traceback
    import os
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
