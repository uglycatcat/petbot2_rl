"""One fresh scene, fixed paired-control network, explicit LAND interface intervention."""
import argparse,json,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',choices=['static','low','high'],required=True)
    args=parser.parse_args();assert not args.output.exists()
    from upright.config import load_config,require_execution
    from upright.model import source_fingerprint,task_implementation_fingerprint
    from upright.experiment import check_identity,atomic_json
    require_execution('engineering')
    manifest=json.loads((args.reference_run/'run.json').read_text())
    assert manifest['run_id']=='1aabd21b-7656-465d-8253-6b0173f2ca19' and manifest['model_sha256']==source_fingerprint()
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app;env=None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from upright.task import UprightEnvCfg
        from upright.export import make_actor
        from upright.evaluation import evaluate
        cfg=load_config('configs/round21r3.json')
        saved=torch.load(args.reference_run/'checkpoints/update_001000.pt',map_location='cpu',weights_only=False)
        check_identity(saved['restart'],manifest)
        actor=make_actor(cfg);actor.load_state_dict(saved['actor_state_dict'],strict=True);actor=actor.to('cuda:0').eval()
        ec=UprightEnvCfg();ec.apply_experiment(cfg);ec.scene.num_envs=32;ec.seed=cfg.seed+10000;ec.sim.device='cuda:0'
        env=ManagerBasedRLEnv(cfg=ec);env.reset()
        env.upright_runtime.course.load_state_dict(saved['restart']['curriculum'])
        level={'static':1,'low':2,'high':8}[args.scenario]
        result=evaluate(env,actor,level,32,duration=60.,output=args.output,kind='development_curriculum_probe')
        atomic_json(args.output/'intervention.json',dict(kind='fixed_control_development_only',scenario=args.scenario,
            reference_run=manifest['run_id'],source_sha256=source_fingerprint(),code_sha256=task_implementation_fingerprint(),
            learning=False,config=cfg.to_dict(),complete=result['complete'],basic=result.get('basic_control_pass')))
        print(json.dumps(dict(complete=result['complete'],basic=result.get('basic_control_pass'))),flush=True)
    finally:
        if env:env.close()
        app.close()
    return 0
if __name__=='__main__':
    try:code=main()
    except BaseException:traceback.print_exc();code=82
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
