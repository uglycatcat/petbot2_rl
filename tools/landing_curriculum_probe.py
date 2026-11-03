"""Fixed-reference engineering probe; no PPO, transfer, or teacher data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path.cwd()))


def main():
    from upright.config import load_config, require_execution
    from upright.experiment import atomic_json, check_identity
    from upright.model import source_fingerprint, task_implementation_fingerprint
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_execution('engineering')
    if args.output.exists():raise FileExistsError(args.output)
    ref=json.loads((args.reference_run/'run.json').read_text())
    checkpoint=args.reference_run/'checkpoints/update_001000.pt'
    assert ref['run_id']=='5c085c17-bf77-4e7a-86d1-2f265ed90a08'
    assert ref['model_sha256']==source_fingerprint()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='19af9aa1faae970de1cc360fc312bffb116ad1fda565d60773249057702db035'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    env=None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from upright.task import UprightEnvCfg
        from upright.export import make_actor
        from upright.evaluation import evaluate
        cfg=load_config('configs/round21r2_quality.json',num_envs=32)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        check_identity(saved['restart'],ref)
        actor=make_actor(cfg);actor.load_state_dict(saved['actor_state_dict'],strict=True)
        actor=actor.to('cuda:0').eval()
        env_cfg=UprightEnvCfg();env_cfg.apply_experiment(cfg)
        env_cfg.scene.num_envs=32;env_cfg.seed=cfg.seed+10000;env_cfg.sim.device='cuda:0'
        env=ManagerBasedRLEnv(cfg=env_cfg);env.reset()
        runtime=env.upright_runtime;runtime.course.level=8
        original=runtime.landing_approach_cost
        report=dict(kind='fixed_round20_engineering_no_learning',tested_scales={},land_samples=0,
                    code_sha256=task_implementation_fingerprint(),source_sha256=source_fingerprint())
        def checked(s,truth,landing,rear_near):
            previous=runtime.course.level
            try:
                runtime.course.level=8
                full=original(s,truth,landing,rear_near)
                for level,factor in [(2,0.),(3,1/6),(5,.5),(8,1.)]:
                    runtime.course.level=level
                    actual=original(s,truth,landing,rear_near)
                    torch.testing.assert_close(actual,full*factor,rtol=0,atol=1e-8)
                    report['tested_scales'][str(level)]=factor
                report['land_samples']+=int(landing.sum())
            finally:runtime.course.level=previous
            return original(s,truth,landing,rear_near)
        runtime.landing_approach_cost=checked
        result=evaluate(env,actor,1,32,duration=60.,output=args.output,kind='development_curriculum_probe')
        report.update(complete=result['complete'],target_pass=result['target_limit_pass'],effort_pass=result['effort_limit_pass'])
        assert result['complete']==32 and result['target_limit_pass'] and result['effort_limit_pass']
        assert report['land_samples']>0 and len(report['tested_scales'])==4
        atomic_json(args.output/'schedule_review.json',report)
        print(json.dumps(report),flush=True)
    finally:
        if env is not None:env.close()
        app.close()
    return 0


if __name__=='__main__':
    try:code=main()
    except BaseException:traceback.print_exc();code=82
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
