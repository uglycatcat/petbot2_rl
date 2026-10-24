"""GPU PhysX audit of actual contact history updates, with a lazy control case."""
import argparse
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--lazy',action='store_true');p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
try:
    import torch
    import warp as wp
    from isaaclab.envs import ManagerBasedRLEnv
    from upright.feedback_task import FeedbackEnvCfg
    from upright.cycle_task import tensor
    cfg=FeedbackEnvCfg();cfg.scene.num_envs=8;cfg.sim.device=args.device;cfg.seed=42
    if args.lazy:cfg.scene.lazy_sensor_update=True;cfg.scene.contact_forces.update_period=cfg.sim.dt
    env=ManagerBasedRLEnv(cfg=cfg);env.reset();sensor=env.scene['contact_forces'];original=sensor._update_buffers_impl
    samples=[]
    def update(*a,**kw):
        original(*a,**kw)
        # Read the completed raw buffer, not the .data property that schedules work.
        samples.append(wp.to_torch(sensor._data._net_forces_w).clone())
    sensor._update_buffers_impl=update
    report=dict(lazy=cfg.scene.lazy_sensor_update,physics_dt=env.physics_dt,control_dt=env.step_dt,decimation=cfg.decimation,intervals=[])
    for _ in range(5):
        samples.clear();obs,reward,terminated,truncated,info=env.step(torch.zeros(8,3,device=args.device))
        assert not (terminated|truncated).any()
        history=wp.to_torch(sensor._data._net_forces_w_history)
        row={'actual_updates':len(samples)}
        if not args.lazy:
            assert len(samples)==cfg.decimation,(len(samples),cfg.decimation)
            expected=torch.stack(list(reversed(samples)),dim=1)
            row['history_max_error']=float((history-expected).abs().max())
            assert row['history_max_error']==0.,row
        report['intervals'].append(row)
    report['passed']=not args.lazy and all(r['actual_updates']==30 and r['history_max_error']==0 for r in report['intervals'])
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2));print(report)
    env.close();app.close()
except BaseException:
    import os,traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
