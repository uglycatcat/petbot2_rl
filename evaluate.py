"""Fixed new-policy validation. Never optimize, initialize training, or auto-select old weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from upright.config import Config, require_execution
from upright.experiment import SCHEMA, identities, check_identity, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=('rise', 'static_cycle', 'low_cycle', 'capability_cycle', 'landing_transfer'), required=True)
    parser.add_argument('--level', type=int, help='Explicit capability level; default is reached training course')
    parser.add_argument('--duration', type=float, help='Two-wheel verification duration, without changing training identity')
    parser.add_argument('--takeover-scale',type=float,default=1.,help='Separate validation: multiply initial velocity/hip ranges only (1..2)')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    require_execution('evaluation')
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Isaac evaluation must run in the dedicated container')
    manifest = json.loads((args.run_dir/'run.json').read_text())
    cfg = Config(**manifest['config']).validate()
    check_identity(manifest, dict(manifest, **identities(cfg)))
    if args.checkpoint.resolve().parent != args.run_dir.resolve()/'checkpoints':
        raise ValueError('Checkpoint must belong to the specified new run')
    if args.output.exists():
        raise ValueError('Refusing to overwrite an evaluation')
    if args.count < 1 or args.count > cfg.num_envs:
        raise ValueError('Invalid evaluation case count')
    if args.level is not None and (args.scenario not in ('capability_cycle', 'landing_transfer') or not 2 <= args.level <= cfg.max_course_level):
        raise ValueError('Explicit level requires a configured moving capability scenario')
    if args.duration is not None and (not __import__('math').isfinite(args.duration) or args.duration <= 0):
        raise ValueError('Duration must be positive and finite')
    if not 1 <= args.takeover_scale <= 2 or (args.takeover_scale != 1 and args.scenario != 'capability_cycle'):
        raise ValueError('Takeover scale requires a capability scenario and range[1,2]')
    app = env = None
    code = 0
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device=args.device).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from rsl_rl.runners import OnPolicyRunner
        from upright.task import UprightEnvCfg, SPEC
        from upright.learning import Adapter, runner_config
        from upright.evaluation import evaluate
        env_cfg = UprightEnvCfg(); env_cfg.apply_experiment(cfg)
        if args.scenario == 'landing_transfer':
            env_cfg.episode_length_s += 2.
        if args.duration is not None:
            env_cfg.episode_length_s += max(0., args.duration-max(cfg.balance_train_s, cfg.balance_eval_s))
        env_cfg.scene.num_envs = args.count; env_cfg.seed = cfg.seed+10000; env_cfg.sim.device = args.device
        env = ManagerBasedRLEnv(cfg=env_cfg); env.reset()
        runner = OnPolicyRunner(Adapter(env), runner_config(cfg, SPEC, args.device), log_dir=None, device=args.device)
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        check_identity(saved.get('restart', {}), manifest)
        runner.alg.load(saved, load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False), strict=True)
        trained_level = saved['restart']['curriculum']['level']
        if args.scenario in ('capability_cycle', 'landing_transfer'):
            level = args.level if args.level is not None else max(2, trained_level)
        else:
            level = {'rise': 0, 'static_cycle': 1, 'low_cycle': 2}[args.scenario]
        if args.takeover_scale != 1.:
            runtime = env.upright_runtime
            runtime.stage_ranges[:,2] = (runtime.stage_ranges[:,2]*args.takeover_scale).clamp_max(.75)
            runtime.stage_ranges[:,3] = (runtime.stage_ranges[:,3]*args.takeover_scale).clamp_max(1.)
        result = evaluate(env, runner.get_inference_policy(device=args.device), level, args.count,
                          duration=args.duration if args.duration is not None else cfg.balance_eval_s, output=args.output,
                          moving=args.scenario != 'landing_transfer', landing_transfer=args.scenario == 'landing_transfer',
                          ordinary_observe_s=2. if args.scenario == 'landing_transfer' else 0.)
        result.update(takeover_scale=args.takeover_scale, separate_takeover_validation=args.takeover_scale != 1., run_id=manifest['run_id'], checkpoint=args.checkpoint.name,
                      updates_completed=saved['restart']['updates_completed'], scenario=args.scenario,
                      trained_course_level=trained_level, outside_trained_course=level > trained_level)
        atomic_json(args.output/'result.json', result)
        print(json.dumps(result, indent=2))
    except BaseException:
        # Record the original exception before Isaac cleanup can replace it
        # with a fast process exit(0). The supervisor must never accept a missing result.
        traceback.print_exc()
        code = 82
    finally:
        try:
            if env is not None: env.close()
            if code:
                # Kit fast shutdown may call _exit(0), bypassing Python handlers.
                sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
            if app is not None: app.close()
        except BaseException as exc:
            if not (isinstance(exc, SystemExit) and exc.code in (None, 0)):
                traceback.print_exc()
                code = code or 82
    return code


if __name__ == '__main__':
    try:
        code = main()
    except BaseException as exc:
        if isinstance(exc, SystemExit): code = exc.code
        else: traceback.print_exc(); code = 82
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code or 0)
