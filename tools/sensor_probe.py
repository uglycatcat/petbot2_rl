"""Replay recorded actions only for finite sensor diagnosis, never for training.

The original fixed trace is read-only. No checkpoint or optimizer is loaded.
Truth channels are written for offline comparison and never fed to the estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--replay-trace', type=Path, required=True)
    p.add_argument('--training-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ids', type=int, nargs='+', default=[1, 16, 29, 51])
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    rows = [json.loads(s) for s in args.replay_trace.open()]
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        import numpy as np
        import torch
        from step_demo.config import load_config
        from step_demo.isaac_env import StepEnv, tensor
        training = json.loads(args.training_config.read_text())
        training['num_envs'] = len(rows[0]['action'])
        env = StepEnv(load_config(model_dir=args.model_dir), training, args.asset_dir, args.device)
        env.trace_ids = list(range(env.num_envs))
        env.reset()
        original = env._capture
        samples, layout = [], {}

        def capture(self, counter):
            previous = self._capture_counter
            original(counter)
            if self._capture_counter == previous:
                return
            e, ids = self.estimator, args.ids
            fields = dict(episode=self.episode_serial[:, None],
                accel=tensor(self.imu.data.lin_acc_b), gyro=tensor(self.imu.data.ang_vel_b),
                q_est=e.q, q_true=tensor(self.robot.data.body_link_quat_w)[:, self.lower_body],
                base_position=tensor(self.robot.data.body_link_pos_w)[:, self.lower_body]-self.scene.env_origins,
                base_velocity=tensor(self.robot.data.body_link_lin_vel_w)[:, self.lower_body],
                base_omega=tensor(self.robot.data.body_link_ang_vel_w)[:, self.lower_body],
                com=self.com-self.origin_com, velocity=self.com_vel,
                estimated_com=e.com_position, estimated_velocity=e.com_velocity,
                estimated_base_velocity=e.velocity,
                joint_pos=tensor(self.robot.data.joint_pos), joint_vel=tensor(self.robot.data.joint_vel),
                rear_position=tensor(self.robot.data.body_link_pos_w)[:, self.wheel_bodies[2:]].mean(1)-self.scene.env_origins,
                rear_velocity=tensor(self.robot.data.body_link_lin_vel_w)[:, self.wheel_bodies[2:]].mean(1),
                wheel_min_z=self.minimum[:, :, 2], failure=self.failure[:, None])
            values, start = [], 0
            for k, v in fields.items():
                width = v.shape[-1]
                layout[k] = [start, start+width]
                values.append(v[ids].float()); start += width
            samples.append(torch.cat(values, -1).cpu().numpy())

        env._capture = types.MethodType(capture, env)
        deviations = []
        with torch.inference_mode():
            for row in rows:
                env.step(torch.tensor(row['action'], device=env.device))
                deviations.append(float((env.com-env.origin_com-torch.tensor(row['com'],device=env.device)).abs().max()))
        np.savez_compressed(args.output/'sensors.npz', samples=np.stack(samples))
        report = dict(status='passed', kind='sensor_action_replay_diagnostic_not_training_data',
            layout=layout, ids=args.ids, dt=env.physics_dt, joints=env.joint_names,
            source_sha256=env.asset['source_sha256'], samples=len(samples),
            replay_sha256=hashlib.sha256(args.replay_trace.read_bytes()).hexdigest(),
            maximum_replay_com_deviation_m=max(deviations), final_replay_com_deviation_m=deviations[-1])
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        env.close(); print(json.dumps(report))
    except BaseException:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
