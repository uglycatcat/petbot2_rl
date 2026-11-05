"""Finite actuator-only checks of the refactored task; never PPO data.

Starts from the original scene, applies only the two permitted motor commands,
and records 2 ms physical samples. No weights, pose injection, or state edits.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('standstill','routes'), default='routes')
    p.add_argument('--compiled-geometry', action='store_true')
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(a).app
    env = None
    try:
        import torch
        from step_demo.config import load_config, PROJECT_ROOT
        from step_demo.model import source_fingerprint
        from step_demo.runtime import load_training
        from step_demo.isaac_env import StepEnv
        from step_demo.task import Phase
        from step_demo.review import verify_physics
        cases = [dict(wheel=25., delay=d, hip=-1., duration=t, preload=q, stop_s=u)
                 for d,t,q,u in itertools.product((.34,.36,.38),(.10,.12,.14),(.20,.25,.30),(1.1,1.3))]
        if a.mode == 'standstill': cases = [dict(wheel=0.,delay=0.,hip=0.,duration=0.,preload=0.,stop_s=0.)]*4
        config = load_training(PROJECT_ROOT/'configs/engineering.json')
        config['num_envs'] = len(cases)
        config['compiled_geometry'] = a.compiled_geometry
        contract = load_config(model_dir=a.model_dir)
        before = source_fingerprint(contract.model_dir)
        env = StepEnv(contract, config, a.asset_dir, a.device)
        env.trace_ids = list(range(len(cases)))
        env.physics_trace_ids = [22,28,33,42] if a.mode == 'routes' else list(range(4))
        env.physics_trace_writer = (a.output/'physics.jsonl').open('x')
        records, max_phase = {}, [0]*len(cases)
        with torch.inference_mode(), (a.output/'trace.jsonl').open('x') as trace:
            obs, _ = env.reset()
            for k in range(751 if a.mode == 'routes' else 250):
                t = k*env.step_dt
                action = []
                for c in cases:
                    hip = c['preload'] if t<c['delay'] else c['hip'] if t<c['delay']+c['duration'] else 0.
                    wheel = c['wheel'] if t<c['stop_s'] else 0.
                    action.append([hip,wheel/12.5-1.])
                obs,reward,done,timeout,_ = env.step(torch.tensor(action, device=env.device))
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()
                assert torch.isfinite(reward).all()
                row = env.last_trace; trace.write(json.dumps(row)+'\n')
                for i, phase in enumerate(row['task']['phase']):
                    if i not in records and phase != Phase.FAILED: max_phase[i] = max(max_phase[i], phase)
                for record in env.terminal_records: records.setdefault(record['env'], record)
                env.terminal_records.clear()
                assert (env.hip_target >= -1.-1e-6).all() and (env.hip_target <= 1.2+1e-6).all()
                assert (env.wheel_target >= -1e-6).all() and (env.wheel_target <= 25.+1e-6).all()
                assert (env.packet_count == 1).all()
                if len(records) == len(cases): break
            env.physics_trace_writer.close(); env.physics_trace_writer = None
        if a.mode == 'standstill':
            assert not records, 'Default static command unexpectedly terminates'
            assert env.objective.flight_count.sum() == 0
            successes = []
        else:
            assert len(records) == len(cases), 'Missing terminal trajectory'
            successes = [i for i,r in records.items() if r['success']]
            (a.output/'terminal_records.json').write_text(json.dumps(dict(records=records,max_phase=max_phase),indent=2)+'\n')
            assert successes, 'No valid complete physical route; diagnose before accepting refactor'
            for i in successes:
                r = records[i]
                assert r['front_flight_count'] == [1,1] and r['hold_s'] >= 2.5-1e-6
                assert r['drive_distance'] >= .15-1e-6 and 35.-1e-3 <= r['return_sum'] <= 40.+1e-3
            for r in records.values():
                if not r['success']: assert -15.-1e-3 <= r['return_sum'] <= -4.+1e-3
        assert before == source_fingerprint(contract.model_dir)
        independent = verify_physics(a.output/'physics.jsonl', env.physics_trace_ids, env.hips, contract)
        if a.mode == 'routes': assert independent['all_terminal']
        result = dict(status='passed', mode=a.mode, source_sha256=before,
            independent_physics=independent,
            scripted_physics_only=True, weights_loaded=False, used_for_ppo=False,
            actor_observation_dim=27, critic_observation_dim=env.critic_dim,
            hip_indices=env.hips,
            physics_dt=env.physics_dt, captured_substeps=env.captured_substeps,
            physics_trace_ids=env.physics_trace_ids, cases=cases, records=records,
            max_phase=max_phase, successes=successes)
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    except BaseException:
        (a.output/'result.json').write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:
        if env is not None: env.close()
        app.close()


if __name__ == '__main__': main()
