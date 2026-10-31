"""Matched-layout action replay and finite landing-tail interventions, never PPO data."""
import argparse
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--training-config', type=Path, required=True)
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('replay', 'tails'), required=True)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.evaluation/'trace.jsonl').open()]
    n = len(rows[0]['episode'])
    landing_steps = [next(k for k, row in enumerate(rows)
                         if row['episode'][i] == 1 and all(row['front_landed'][i])) for i in range(n)]
    cases = [dict(hip=h, wheel=w, hip_rate=3.) for h in (-.7, -.4, 0., .3) for w in (4., 8., 12., 16.)]
    a.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(a).app
    try:
        import torch
        from step_demo.config import load_config
        from step_demo.isaac_env import StepEnv, tensor
        training = json.loads(a.training_config.read_text()); training['num_envs'] = n
        env = StepEnv(load_config(model_dir=a.model_dir), training, a.asset_dir, a.device)
        env.trace_ids = list(range(n)); env.reset()
        results = [dict(env=i, case=cases[i % len(cases)] if a.mode == 'tails' else None,
                        max_prefix_physics_error=0., ever_all_top=False, terminal=None,
                        max_phase=0, max_rear_speed=0.) for i in range(n)]
        requested = [0.] * n
        original = env._capture
        with (a.output/'substeps.jsonl').open('x') as sub, (a.output/'trace.jsonl').open('x') as trace:
            def capture(self, counter):
                old = self._capture_counter; original(counter)
                if self._capture_counter == old:
                    return
                sub.write(json.dumps(dict(counter=counter, episode=self.episode_serial.tolist(),
                    failure=self.failure.tolist(), rear_air_s=self.rear_air_s.tolist(),
                    net_force=self.wheel_contact_force.tolist(), platform_normal=self.platform_normal.tolist(),
                    wheel_z=self.minimum[:, :, 2].tolist(),
                    wheel_min_x=(self.minimum[:, :, 0]-self.edge_x[:, None]).tolist(),
                    hip_target=self.hip_target.tolist(), wheel_target=self.wheel_target.tolist(),
                    hip_actual=tensor(self.robot.data.joint_pos)[:, self.hips].tolist(),
                    wheel_actual=tensor(self.robot.data.joint_vel)[:, self.wheels].tolist()))+'\n')
            env._capture = types.MethodType(capture, env)
            count = len(rows) if a.mode == 'replay' else 200
            with torch.inference_mode():
                for step in range(count):
                    values = []
                    for i in range(n):
                        if a.mode == 'replay' or step <= landing_steps[i]:
                            value = rows[step]['action'][i]
                            requested[i] = rows[step]['hip_target'][i]
                        else:
                            case = cases[i % len(cases)]
                            target = case['hip'] if step*env.step_dt < 2. else 0.
                            requested[i] += max(-case['hip_rate']*env.step_dt,
                                                min(case['hip_rate']*env.step_dt, target-requested[i]))
                            value = [requested[i]/1.2 if requested[i] >= 0 else requested[i],
                                     (case['wheel'] if step*env.step_dt < 3.2 else 0.)/12.5-1.]
                        values.append(value)
                    obs, reward, _, _, _ = env.step(torch.tensor(values, device=env.device))
                    assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    d = env.last_trace; trace.write(json.dumps(d)+'\n')
                    for i, result in enumerate(results):
                        if d['episode'][i] != 1 or result['terminal'] is not None:
                            continue
                        if a.mode == 'replay' or step <= landing_steps[i]:
                            for key in ('com', 'joint_position', 'joint_velocity', 'estimated_com'):
                                result['max_prefix_physics_error'] = max(result['max_prefix_physics_error'],
                                    max(abs(x-y) for x, y in zip(d[key][i], rows[step][key][i])))
                        result['ever_all_top'] |= all(d['supported_top'][i])
                        result['max_phase'] = max(result['max_phase'], d['phase'][i])
                        result['max_rear_speed'] = float(env.wheel_speed_peak[i]) if not d['failure'][i] else result['max_rear_speed']
                        if d['failure'][i] or d['completed'][i]:
                            result['terminal'] = dict(time=d['elapsed'][i], failure=d['failure'][i],
                                completed=d['completed'][i], phase=d['phase'][i], com=d['com'][i])
                    if all(r['terminal'] is not None for r in results):
                        break
        (a.output/'result.json').write_text(json.dumps(dict(status='passed',
            kind='finite_diagnostic_not_training', mode=a.mode, source_sha256=env.asset['source_sha256'],
            maximum_prefix_error=max(r['max_prefix_physics_error'] for r in results), cases=results), indent=2)+'\n')
        env.close()
    except BaseException:
        import traceback
        (a.output/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        app.close()


if __name__ == '__main__':
    main()
