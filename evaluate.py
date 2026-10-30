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
    parser.add_argument('--scenario', choices=('rise', 'static_cycle', 'low_cycle'), required=True)
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
    app = env = None
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
        env_cfg.scene.num_envs = args.count; env_cfg.seed = cfg.seed+10000; env_cfg.sim.device = args.device
        env = ManagerBasedRLEnv(cfg=env_cfg); env.reset()
        runner = OnPolicyRunner(Adapter(env), runner_config(cfg, SPEC, args.device), log_dir=None, device=args.device)
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        check_identity(saved.get('restart', {}), manifest)
        runner.alg.load(saved, load_cfg=dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False), strict=True)
        level = {'rise': 0, 'static_cycle': 1, 'low_cycle': 2}[args.scenario]
        result = evaluate(env, runner.get_inference_policy(device=args.device), level, args.count,
                          duration=cfg.balance_eval_s, output=args.output)
        result.update(run_id=manifest['run_id'], checkpoint=args.checkpoint.name,
                      updates_completed=saved['restart']['updates_completed'], scenario=args.scenario)
        atomic_json(args.output/'result.json', result)
        print(json.dumps(result, indent=2))
    finally:
        if env is not None: env.close()
        if app is not None: app.close()
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except BaseException as exc:
        if isinstance(exc, SystemExit): code = exc.code
        else: traceback.print_exc(); code = 82
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code or 0)
