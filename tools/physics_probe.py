"""Finite no-learning physical control probe, importing the fixed working-directory implementation."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from upright.config import Config, require_execution

require_execution('engineering')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from upright.task import UprightEnvCfg, SPEC, state, truth
    from upright.model import POSITION_JOINTS, REAR_WHEELS, source_fingerprint
    cfg=UprightEnvCfg(); cfg.apply_experiment(Config(num_envs=32)); cfg.sim.device='cuda:0'
    env=ManagerBasedRLEnv(cfg=cfg); env.reset(); env.upright_record=True
    lo,hi=SPEC['joints']['left_frame']['range']; home=-(lo+hi)/(hi-lo)
    action=torch.tensor([home,0.,0.],device=env.device).expand(env.num_envs,-1)
    peak_effort=torch.zeros(7,device=env.device); max_target=0.; max_actual=0.; failures=0
    frames=[]
    for step in range(100):
        obs,reward,terminated,timeout,_=env.step(action)
        snap=env.upright_snapshot
        if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all(): raise RuntimeError('Nonfinite physics')
        failures += int(terminated.sum())
        peak_effort=torch.maximum(peak_effort,snap['effort'].abs().amax(0))
        max_target=max(max_target,float(snap['executed_targets'][:,1:].abs().max()))
        max_actual=max(max_actual,float(snap['wheel'].abs().max()))
        frames.append(dict(step=step,**{k:v[:4].cpu().tolist() for k,v in snap.items()}))
    limits=peak_effort.new_tensor([SPEC['actuators'][n]['effort'] for n in POSITION_JOINTS+REAR_WHEELS])
    report=dict(kind='neutral_target_physics_probe_no_learning',num_envs=32,steps=100,
                control_dt=env.step_dt,physics_dt=cfg.sim.dt,contact_history_frames=cfg.scene.contact_forces.history_length,
                lazy_sensor_update=cfg.scene.lazy_sensor_update,failures=failures,
                max_target_rad_s=max_target,max_actual_rad_s=max_actual,max_effort_nm=peak_effort.cpu().tolist(),
                source_sha256=source_fingerprint(),rear_fraction_final=float(snap['rear_fraction'].mean()),
                four_fraction_final=float(snap['four_fraction'].mean()),
                estimated_gap_final=snap['gap'][:4].cpu().tolist(),front_height_final=snap['front_z'][:4].cpu().tolist())
    report['passed']=(failures==0 and max_target<=25.00001 and bool((peak_effort<=limits+1e-3).all())
                      and report['rear_fraction_final']>.2 and source_fingerprint()==SPEC['source_sha256'])
    out=Path('outputs/physics_probe');out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'trace.json').write_text(json.dumps(frames))
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise RuntimeError('Physical probe rejected; inspect actual traces')
    env.close()
finally:
    app.close()
