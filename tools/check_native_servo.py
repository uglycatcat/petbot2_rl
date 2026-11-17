"""Check free-wheel braking at several integration resolutions, without editing the source model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from recover.model import (workspace_root, POSITION_JOINTS, POSITION_KP,
                           POSITION_KD, REAR_WHEELS, source_fingerprint)
from recover.poses import load_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = []
    for dt in (.002, .001, .0005, .00025):
        model = mujoco.MjModel.from_xml_path(str(workspace_root()/'robot_description/mjcf/scene.xml'))
        source_dt = float(model.opt.timestep)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        poses, _, _ = load_poses()
        data.qpos[:] = poses[0]
        mujoco.mj_forward(model, data)
        jp = np.array([model.joint(n).qposadr[0] for n in POSITION_JOINTS])
        jd = np.array([model.joint(n).dofadr[0] for n in POSITION_JOINTS])
        pa = np.array([model.actuator(n).id for n in POSITION_JOINTS])
        wa = np.array([model.actuator(n).id for n in REAR_WHEELS])
        wd = np.array([model.joint(n).dofadr[0] for n in REAR_WHEELS])
        kp = np.array([POSITION_KP[n] for n in POSITION_JOINTS])
        kd = np.array([POSITION_KD[n] for n in POSITION_JOINTS])
        target = data.qpos[jp].copy()
        cap = model.actuator_forcerange[pa, 1]
        trace = []
        for step in range(round(1/dt)):
            data.ctrl[pa] = np.clip(kp*(target-data.qpos[jp])-kd*data.qvel[jd], -cap, cap)
            data.ctrl[wa] = 5. if step*dt < .2 else 0.
            mujoco.mj_step(model, data)
            trace.append([data.time, *data.qvel[wd], *data.actuator_force[wa]])
        samples = np.array(trace)
        tail = samples[samples[:, 0] > .9]
        results.append(dict(physics_dt=dt, late_wheel_min=tail[:, 1:3].min(0).tolist(),
                            late_wheel_max=tail[:, 1:3].max(0).tolist(),
                            late_wheel_rms=np.sqrt((tail[:, 1:3]**2).mean(0)).tolist(),
                            last_10_substeps=tail[-10:].tolist()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(source_hash=source_fingerprint(), source_physics_dt=source_dt,
        mujoco_version=mujoco.__version__, pose='pose_01', integrator='implicitfast',
        command='Hold initial joint targets; rear wheel targets +5 rad/s for 0.2 s, then 0 for 0.8 s',
        results=results), indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='last_10_substeps'} for r in results]), flush=True)


if __name__ == '__main__':
    main()
