#!/usr/bin/env python3
"""Interactive policy demo; --headless produces a reproducible diagnostic trace."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def snapshot(sim, path):
    import mujoco
    from PIL import Image
    camera=mujoco.MjvCamera();camera.lookat[:]=sim.data.xpos[sim.root]
    camera.distance=.8;camera.azimuth=135;camera.elevation=-20
    with mujoco.Renderer(sim.model,height=720,width=1280) as renderer:
        renderer.update_scene(sim.data,camera=camera)
        Image.fromarray(renderer.render()).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,default=ROOT/'policies/cycle18_long_r42')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--frame',type=float,default=0.)
    parser.add_argument('--transition-contract',choices=['legacy','linear_heading'],default=None,
        help='Explicit development intervention; defaults to exported policy contract')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--seconds',type=float,default=36.)
    parser.add_argument('--hold-seconds',type=float,default=18.)
    parser.add_argument('--cycles',type=int,default=1)
    parser.add_argument('--command-scale',type=float,default=1.,help='Scale headless commands; 1.25 reaches the keyboard limits')
    parser.add_argument('--scenario',choices=['stationary','forward','reverse','mixed','ordinary'],default='mixed')
    parser.add_argument('--takeover-yaw',type=float,default=0.,help='Ordinary-drive turn command before the first balance phase')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/latest')
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--viewer-smoke-seconds',type=float)
    args = parser.parse_args()
    import math
    if any(not math.isfinite(v) or v<=0 for v in (args.seconds,args.hold_seconds)) or args.cycles<1:
        parser.error('Durations and cycle count must be positive and finite')
    if not math.isfinite(args.command_scale) or not 0<args.command_scale<=1.25:
        parser.error('Command scale must be in (0, 1.25]')
    if args.headless and args.snapshot:
        os.environ.setdefault('MUJOCO_GL','egl')
    from petbot_s2s.simulation import Simulation
    sim = Simulation(args.policy,args.model_dir,args.frame,args.transition_contract)
    if not math.isfinite(args.takeover_yaw) or abs(args.takeover_yaw*args.command_scale)>sim.controller.yaw_limit:
        parser.error('Takeover yaw must fit the exported policy range')
    if not args.headless:
        from petbot_s2s.viewer import run_viewer
        run_viewer(sim,args.viewer_smoke_seconds)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'summary.json').exists():
        raise FileExistsError('Use a new --output directory to preserve earlier diagnostics')
    c,p = sim.controller,sim.controller.progress
    start = time.monotonic()
    armed = landing = False
    balanced_at = None
    ever_balanced = False
    next_arm = 3.
    previous_cycles = 0
    balance_snapshot = False
    rows = []
    while sim.data.time < args.seconds and not c.failure_code:
        t = float(sim.data.time)
        command = (0.,0.) if args.scenario=='stationary' else (.04,0.)
        if args.scenario=='reverse':
            command = (-.04,0.)
        if balanced_at is None:
            command = (command[0],args.takeover_yaw)
        if args.scenario=='mixed' and balanced_at is not None:
            commands = [(.04,0.),(0.,.12),(-.04,0.),(0.,-.12),(.03,.10),(0.,0.)]
            command = commands[int((t-balanced_at)//3)%len(commands)]
        c.set_command(*(v*args.command_scale for v in command))
        if t>=next_arm and not armed and args.scenario!='ordinary':
            c.toggle();armed=True
        if int(p.phase[0])==2 and balanced_at is None:
            balanced_at=t
            ever_balanced=True
        if balanced_at is not None and t-balanced_at>=args.hold_seconds and not landing:
            c.toggle();landing=True
        sim.step()
        rows.append(sim.record())
        if args.snapshot and int(p.phase[0])==2 and float(p.phase_time[0])>2. and not balance_snapshot:
            snapshot(sim,args.output/'two_wheels.png')
            balance_snapshot=True
        if c.cycles>=args.cycles:
            break
        if c.cycles>previous_cycles:
            previous_cycles=c.cycles
            next_arm=float(sim.data.time)+3.
            armed=landing=False
            balanced_at=None
    result = dict(policy=args.policy.name,checkpoint_sha256=c.manifest['checkpoint_sha256'],
        transition_contract=c.transition_contract,development_intervention=args.transition_contract is not None,
        source_model_sha256=sim.model_sha256,mujoco_version=__import__('mujoco').__version__,
        scenario=args.scenario,takeover_yaw=args.takeover_yaw,command_scale=args.command_scale,frame=args.frame,wall_seconds=time.monotonic()-start,
        simulated_seconds=float(sim.data.time),entered_balance=ever_balanced,
        requested_cycles=args.cycles,
        completed_cycles=c.cycles,failure_code=c.failure_code,events=c.events,
        maximum_front_height=max(row['front_z'] for row in rows),
        final=rows[-1],interpretation='MuJoCo diagnostic, not an Isaac success-rate estimate')
    if len(rows)>1:
        from petbot_s2s.core.functional import score_trajectories
        functional_rows = [dict(t=r['time'],valid=[True],phase=[r['phase']],raw=[r['raw']],
            pos=[r['position']],heading=[r['yaw']],interval_motion=[r['interval_motion']]) for r in rows]
        result['functional'] = score_trajectories(functional_rows,
            dict(complete=[c.cycles>=args.cycles],failure=[bool(c.failure_code)]))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if args.snapshot:
        snapshot(sim,args.output/'final.png')
    print(json.dumps(result,indent=2))
    if c.failure_code or (args.scenario!='ordinary' and c.cycles<args.cycles):
        raise SystemExit(2)


if __name__=='__main__':
    main()
