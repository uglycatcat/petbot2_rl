"""Canonical training/evaluation entry: random initialization or same-run recovery.

No historical checkpoint, teacher, demonstration dataset or frozen parent can
be supplied. Run inside the training container after asset preparation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/scratch_v3.json')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--num_envs',type=int)
    p.add_argument('--iterations',type=int)
    p.add_argument('--save-interval',type=int)
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate',type=Path,help='Checkpoint belonging to --run-dir')
    p.add_argument('--output',type=Path)
    p.add_argument('--speed-limit',type=float,default=.15)
    p.add_argument('--yaw-limit',type=float,default=.5)
    p.add_argument('--frame-min',type=float,default=-.2)
    p.add_argument('--frame-max',type=float,default=.2)
    p.add_argument('--hold-seconds',type=float,default=30.)
    p.add_argument('--evaluation-seed',type=int,default=20261030)
    p.add_argument('--evaluation-pattern',choices=['mixed','paired_speed','paired_arcs','rise_grid'],default='mixed')
    p.add_argument('--stop-after',type=int,help='Engineering interruption at this total completed update count')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action='store_true')
    a=p.parse_args()
    cfg=json.loads(((a.run_dir/'scratch_config.json') if a.resume else a.config).read_text())
    if cfg['schema']!='petbot_scratch_v1' or cfg['pretrained_models']!=[] or cfg.get('reward_design')!='rise_ground_proximity_v3':p.error('Scratch configuration required')
    for key in ('seed','num_envs','iterations','save_interval'):
        if getattr(a,key) is not None:cfg[key]=getattr(a,key)
    if not 1<=cfg['num_envs']<=4096 or cfg['iterations']<1:p.error('Invalid training budget')
    if a.resume and a.evaluate:p.error('Choose recovery or evaluation')
    out=a.run_dir.resolve();manifest=out/'scratch_origin.json'
    identity=json.dumps(cfg,sort_keys=True,separators=(',',':'))
    config_sha=hashlib.sha256(identity.encode()).hexdigest()
    if a.resume or a.evaluate:
        origin=json.loads(manifest.read_text())
        if a.resume and origin['config_sha256']!=config_sha:p.error('Recovery configuration differs from original run')
        cfg=origin['configuration']
        checkpoint=(a.resume or a.evaluate).resolve()
        if checkpoint.parent!=out or not checkpoint.is_file():p.error('Checkpoint must belong to this run directory')
        if a.resume and checkpoint.name=='initial_state.pt':p.error('Initial snapshot is audit evidence, not recovery')
    else:
        if out.exists() and any(out.iterdir()):p.error('Run directory must be new/empty; use --resume for recovery')
        if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
            p.error('Commit source before running a scratch experiment')
        out.mkdir(parents=True,exist_ok=True)
        origin=dict(schema='petbot_scratch_origin_v1',id=uuid.uuid4().hex,configuration=cfg,
            config_sha256=config_sha,initialization='random',pretrained_models=[],
            commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
        manifest.write_text(json.dumps(origin,indent=2)+'\n')
        (out/'scratch_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        checkpoint=None
    target=a.output.resolve() if a.output else out
    if a.evaluate and (target==out or target.exists()):p.error('Evaluation requires a fresh --output directory')
    target.mkdir(parents=True,exist_ok=True)
    end=a.stop_after or cfg['iterations']
    if not 1<=end<=cfg['iterations']:p.error('Invalid engineering stop point')
    args=['train_cycle.py','--from-scratch','--scratch-run',str(out),
        '--num_envs',str(a.num_envs if a.evaluate and a.num_envs else cfg['num_envs']),
        '--iterations',str(end),'--save-interval',str(cfg['save_interval']),
        '--learning-rate',str(cfg['learning_rate']),'--seed',str(a.evaluation_seed if a.evaluate else cfg['seed']),
        '--device',a.device,'--log_dir',str(target),'--task-profile','full_range',
        '--tracking-weight','4','--instant-tracking-weight','.3','--tracking-objective','stability_first',
        '--observation-frame','balance_local','--handoff-alignment','after','--refinement-target','scratch',
        '--transition-contract','linear_heading','--actor-normalization','inherited','--yaw-feedback-design','fast2',
        '--yaw-feedback-gain','16','--balance-reflex','off','--balance-noise-factor','1',
        '--wheel-slew','480','--frame-slew','3','--capability-version','v25','--mirror-training','--adaptive-curriculum',
        '--training-mixture','off']
    limits=(a.speed_limit,a.yaw_limit,a.frame_min,a.frame_max) if a.evaluate else (.75,15.,-1.,1.2)
    for k,v in zip(('speed-limit','yaw-limit','frame-min','frame-max'),limits):args+=['--'+k,str(v)]
    hold=[a.hold_seconds]*2 if a.evaluate else cfg['hold_seconds']
    args+=['--hold-min',str(hold[0]),'--hold-max',str(hold[1])]
    if checkpoint:args+=['--checkpoint',str(checkpoint)]
    if a.evaluate:args+=['--evaluate','--evaluation-pattern',a.evaluation_pattern]
    if a.headless:args+=['--headless']
    os.chdir(ROOT);sys.argv=args
    runpy.run_path(str(ROOT/'train_cycle.py'),run_name='__main__')


if __name__=='__main__':main()
