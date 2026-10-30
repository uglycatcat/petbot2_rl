"""Only supported training entry. Random initialization, or forced same-run recovery."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from upright.config import ROOT, load_config, require_execution
from upright.experiment import atomic_json, identities, locked_run, new_manifest, validate_resume, software_versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/round01.json')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume-interrupted', type=Path)
    parser.add_argument('--kind', choices=('training', 'engineering'), default='training')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true', help='Read-only plan; no simulator, files or training')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, seed=args.seed, iterations=args.iterations, num_envs=args.num_envs)
    if args.dry_run:
        print(json.dumps(dict(config=cfg.to_dict(), identity=identities(cfg),
                              initialization='same_run_recovery_requested' if args.resume_interrupted else 'random',
                              requested_kind=args.kind, launched=False), indent=2))
        return 0
    require_execution(args.kind)  # before importing Isaac Lab / constructing a CUDA context
    if not Path('/.dockerenv').exists():
        raise RuntimeError('Isaac training is only allowed inside the dedicated container')
    if args.run_dir is None:
        parser.error('--run-dir is required; no implicit latest run is selected')
    if not args.device.startswith('cuda:'):
        parser.error('Training requires an explicit CUDA device in the dedicated container')
    folder = args.run_dir.resolve()
    # Fixed source copies retain the project/model relative layout; outputs live below this checkout.
    if not folder.is_relative_to(ROOT/'outputs'):
        raise ValueError('Run directory must be within this checkout/outputs; historical runs are not scanned')
    with locked_run(folder):
        if args.resume_interrupted:
            if not (folder/'run.json').is_file():
                raise ValueError('No existing run identity to resume')
            old = json.loads((folder/'run.json').read_text())
            current = dict(old, **identities(cfg), software_versions=software_versions())
            manifest = validate_resume(folder, args.resume_interrupted, current)
        else:
            if any(p.name != '.run.lock' for p in folder.iterdir()):
                raise ValueError('A new run requires an empty run directory')
            manifest = new_manifest(cfg, args.kind)
            atomic_json(folder/'run.json', manifest)
        app = env = runner = None
        status = dict(run_id=manifest['run_id'], state='starting', pid=os.getpid(), updates_completed=0,
                      policy_validated=False, engineering_validated=False)
        atomic_json(folder/'status.json', status)
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'forced signal {signum}')
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupt)
        code = 0
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True, device=args.device).app
            import random
            import numpy as np
            import torch
            from isaaclab.envs import ManagerBasedRLEnv
            from upright.task import UprightEnvCfg, SPEC
            from upright.learning import Adapter, Runner, runner_config
            from upright.evaluation import evaluate
            random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
            env_cfg = UprightEnvCfg(); env_cfg.apply_experiment(cfg); env_cfg.sim.device = args.device
            env = ManagerBasedRLEnv(cfg=env_cfg); env.reset()
            runner = Runner(Adapter(env), runner_config(cfg, SPEC, args.device), folder, manifest, args.device)
            if args.resume_interrupted:
                runner.load_same_run(args.resume_interrupted)
            else:
                # This is evidence of the actual fresh network, not an initialization source.
                torch.save(dict(actor=runner.alg.get_policy().state_dict(), run_id=manifest['run_id'],
                                initialization='random', updates_completed=0), folder/'initialization.pt')
            status['state'] = 'running'; atomic_json(folder/'status.json', status)
            def finish_pending_probe():
                p = env.upright_runtime
                boundary = runner.completed_updates
                if boundary <= 0 or boundary % cfg.probe_interval or p.course.last_probe_update >= boundary:
                    return
                level = p.course.level
                # An interrupted probe is repeated at the same PPO boundary;
                # preserve its old outputs rather than rerunning any PPO updates.
                attempt = 0
                while (folder/f'probe_{boundary:06d}_attempt{attempt:02d}').exists():
                    attempt += 1
                result = evaluate(env, runner.get_inference_policy(device=args.device), level,
                                  cfg.probe_count, duration=cfg.balance_train_s,
                                  output=folder/f'probe_{boundary:06d}_attempt{attempt:02d}',
                                  kind='development_curriculum_probe')
                p.course.observe_probe(boundary, result['success'], result['count'], level)
                if p.course.level != level:
                    env.reset()
                runner.save_boundary()
            finish_pending_probe()
            while runner.completed_updates < cfg.iterations:
                boundary = min(cfg.iterations, ((runner.completed_updates//cfg.probe_interval)+1)*cfg.probe_interval)
                runner.learn_chunk(boundary-runner.completed_updates)
                p = env.upright_runtime
                finish_pending_probe()
                status.update(updates_completed=runner.completed_updates, course_level=p.course.level)
                atomic_json(folder/'status.json', status)
            status['state'] = 'complete_execution_only'
        except KeyboardInterrupt as exc:
            status.update(state='interrupted', error=str(exc)); code = 130
        except BaseException as exc:
            message = str(exc)
            status.update(state='failed_oom' if 'out of memory' in message.lower() else 'failed',
                          error=f'{type(exc).__name__}: {message}')
            traceback.print_exc(); code = 82
        finally:
            if runner is not None:
                status['updates_completed'] = runner.completed_updates
            # Only the last durable PPO update boundary is resumable. No half-rollout snapshot.
            atomic_json(folder/'status.json', status)
            try:
                if env is not None:
                    env.close()
                if app is not None:
                    app.close()
            except BaseException:
                traceback.print_exc(); code = code or 82
            if code and status['state'] == 'complete_execution_only':
                status.update(state='failed_cleanup', error='Simulator cleanup failed')
                atomic_json(folder/'status.json', status)
        return code


if __name__ == '__main__':
    try:
        exit_code = main()
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            exit_code = exc.code
        else:
            traceback.print_exc(); exit_code = 82
    sys.stdout.flush(); sys.stderr.flush()
    # Isaac shutdown handlers must not turn a Python failure into successful exit.
    os._exit(exit_code or 0)
