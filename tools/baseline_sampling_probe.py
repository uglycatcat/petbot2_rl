"""Engineering only: fresh high-course resets and real physics; no policy training."""
import argparse,json,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    from upright.config import load_config,require_execution
    from upright.model import source_fingerprint,task_implementation_fingerprint
    from upright.experiment import atomic_json
    require_execution('engineering')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app;env=None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from upright.task import UprightEnvCfg
        cfg=load_config('configs/round21r5.json')
        ec=UprightEnvCfg();ec.apply_experiment(cfg);ec.scene.num_envs=2048;ec.seed=cfg.seed;ec.sim.device='cuda:0'
        env=ManagerBasedRLEnv(cfg=ec);env.reset();runtime=env.upright_runtime
        runtime.course.level=cfg.max_course_level
        counts=torch.zeros(cfg.max_course_level+1,device='cuda:0');trace=[];target_peak=0.;effort_peak=torch.zeros(7,device='cuda:0')
        env.upright_record=True
        for batch in range(4):
            env.reset();counts+=torch.bincount(runtime.level,minlength=len(counts))
            initial=runtime.level[:32].cpu().tolist()
            for step in range(64):
                env.step(torch.zeros(2048,3,device='cuda:0'))
                snap=env.upright_snapshot
                for value in snap.values():
                    assert torch.isfinite(value).all()
                target_peak=max(target_peak,float(snap['executed_targets'][:,1:].abs().max()))
                effort_peak=torch.maximum(effort_peak,snap['effort'].abs().amax(0))
                if step in (0,31,63):
                    trace.append(dict(batch=batch,step=step,initial_levels=initial,
                        **{key:snap[key][:32].cpu().tolist() for key in ['phase','requested_command','control_reference','effective_command','executed_targets','motion','q','wheel','effort']}))
        fractions=counts/counts.sum()
        assert abs(float(fractions[8])-.8)<.025 and abs(float(fractions[1])-.0571429)<.02 and abs(float(fractions[2])-.1071429)<.02
        assert bool((counts[3:8]>0).all()) and target_peak<=25.00001
        assert bool((effort_peak <= effort_peak.new_tensor([.72,.1,.1,1.2,1.2,.38,.38])+1e-3).all())
        args.output.mkdir(parents=True)
        atomic_json(args.output/'result.json',dict(kind='engineering_sampling_only',learning=False,count=int(counts.sum()),
            counts=counts.cpu().tolist(),fractions=fractions.cpu().tolist(),control_steps=256,all_finite=True,
            max_target=target_peak,max_effort=effort_peak.cpu().tolist(),code_sha256=task_implementation_fingerprint(),model_sha256=source_fingerprint()))
        (args.output/'trace.json').write_text(json.dumps(trace)+'\n')
        print(json.dumps(dict(counts=counts.cpu().tolist(),fractions=fractions.cpu().tolist(),target_peak=target_peak)),flush=True)
    finally:
        if env:env.close()
        app.close()

if __name__=='__main__':
    try:main();code=0
    except BaseException:traceback.print_exc();code=82
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
