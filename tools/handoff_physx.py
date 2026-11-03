"""Explicit fixed round20 actor diagnostic; never an initialization/PPO entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path.cwd()))


def main():
    from upright.config import Config, require_execution
    from upright.experiment import check_identity, atomic_json
    from upright.model import source_fingerprint
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); require_execution('engineering')
    ref = json.loads((a.reference_run/'run.json').read_text())
    checkpoint = a.reference_run/'checkpoints/update_001000.pt'
    assert ref['run_id'] == '5c085c17-bf77-4e7a-86d1-2f265ed90a08'
    assert ref['initialization'] == 'random'
    assert ref['model_sha256'] == source_fingerprint()
    assert ref['code_sha256'] == 'f00d51fdc9f35e775d4d2eade1c361c3dc7d57dc2c2b20fef5f1b6ac748bdccd'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '19af9aa1faae970de1cc360fc312bffb116ad1fda565d60773249057702db035'
    if a.output.exists(): raise FileExistsError(a.output)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0').app
    env = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from upright.task import UprightEnvCfg
        from upright.export import make_actor
        from upright.evaluation import evaluate
        cfg = Config(**dict(ref['config'], normal_handoff_live_drive=True)).validate()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        check_identity(saved['restart'], ref)
        actor = make_actor(cfg); actor.load_state_dict(saved['actor_state_dict'], strict=True)
        actor = actor.to('cuda:0').eval()
        env_cfg = UprightEnvCfg(); env_cfg.apply_experiment(cfg)
        env_cfg.scene.num_envs = 32; env_cfg.seed = cfg.seed+10000; env_cfg.sim.device = 'cuda:0'
        env = ManagerBasedRLEnv(cfg=env_cfg); env.reset()
        a.output.mkdir(parents=True)
        for name, level, duration in [('static', 1, 60.), ('low', 2, 60.), ('high', 8, 60.)]:
            result = evaluate(env, actor, level, 32, duration=duration, output=a.output/name,
                              kind='development_curriculum_probe')
            result.update(kind='fixed_round20_handoff_development_no_learning', reference_run_id=ref['run_id'],
                          normal_handoff_live_drive=True, scenario=name)
            atomic_json(a.output/name/'result.json', result)
            print(json.dumps(result), flush=True)
    finally:
        if env is not None: env.close()
        app.close()
    return 0


if __name__ == '__main__':
    try: code = main()
    except BaseException:
        traceback.print_exc(); code = 82
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
