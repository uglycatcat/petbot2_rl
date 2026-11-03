"""Finite fixed-ONNX landing diagnosis; no learning or published sim2sim edits.

Interventions are explicitly labelled and apply only to ordinary handoff timing.
Never use these scores as independent policy validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-source',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['baseline','immediate_handoff','prompt_ground','prompt_ground_immediate'],
                   choices=['baseline','immediate_handoff','prompt_ground','prompt_ground_immediate'])
    p.add_argument('--cases',nargs='+',default=['stationary','forward_release','reverse_release','forward_held','reverse_held'],
                   choices=['stationary','forward_release','reverse_release','forward_held','reverse_held'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.sim_source.resolve()))
    import numpy as np
    import torch
    from petbot_s2s.restart import RestartSimulation
    from petbot_s2s.restart_core.controller import Phase, wheel_mix
    torch.set_num_threads(1)
    results=[]
    for mode in a.modes:
        for case in a.cases:
            sim=RestartSimulation(a.policy.resolve());c=sim.controller
            assert c.manifest['source_run']=='upright_restart_round20_s102'
            c.set_policy_limits(.6,.5)
            if mode in ('prompt_ground','prompt_ground_immediate'):
                c.cfg.ground_confirm_s=c.dt
            if mode in ('immediate_handoff','prompt_ground_immediate'):
                original=c.control.targets
                def targets(action,state):
                    result=original(action,state)
                    if int(c.control.phase[0]) in (Phase.HANDOFF,Phase.EXIT):
                        command=c.control.command
                        result[:,1:]=wheel_mix(command[:,0]/.03,command[:,1]*c.geometry.track/(2*.03),c.control.limit)
                    return result
                c.control.targets=targets
            balanced=land_at=handoff=complete=None;armed=land=False;rows=[]
            speed={'stationary':0.,'forward_release':.6,'reverse_release':-.6,'forward_held':.15,'reverse_held':-.15}[case]
            while sim.data.time<42:
                t=float(sim.data.time);b=t-balanced if balanced is not None else -1
                command=speed if (case.endswith('_held') or 2<=b<8) else 0.
                c.set_command(command,0.)
                if t>=3 and not armed:c.toggle();armed=True
                if int(c.control.phase[0])==Phase.BALANCE and balanced is None:balanced=t
                if b>=20 and not land and c.armed:c.toggle();land=True;land_at=t
                sim.step();r=sim.record();rows.append(r)
                if r['phase']==Phase.HANDOFF and handoff is None:handoff=r['time']
                if c.cycles and complete is None:complete=r['time']
            trace=a.output/(mode+'_'+case+'.jsonl');trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            contact=next((r for r in rows if land_at is not None and r['time']>=land_at and r['four_fraction']>0),None)
            tail=[r for r in rows if complete is not None and r['time']>complete+2]
            item=dict(mode=mode,case=case,cycles=c.cycles,interruptions=c.interruptions,events=c.events,
                      contact=contact,handoff_at=handoff,complete_at=complete,control_dt=c.dt,
                      trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                      target_peak=max(abs(x) for r in rows for x in r['wheel_target']),
                      actual_wheel_peak=max(abs(x) for r in rows for x in r['wheel_measured']),
                      final_ordinary_speed=float(np.mean([r['motion'][0] for r in tail])) if tail else None,
                      input_speed=speed if case.endswith('_held') else 0.,
                      source_model_sha256=sim.model_sha256,network=c.manifest['networks'])
            if contact is not None and not case.endswith('_held'):
                after=[r for r in rows if r['time']>=contact['time']]
                moving=[i for i,r in enumerate(after) if abs(r['motion'][0])>.01]
                end=(moving[-1]+1) if moving else 0
                item.update(contact_to_stop_s=after[end]['time']-contact['time'] if end<len(after) else None,
                            after_contact_absolute_travel_m=sum(abs(r['motion'][0])*c.dt for r in after[:end]),
                            after_contact_reverse_travel_m=sum(max(0.,-r['motion'][0])*c.dt for r in after[:end]),
                            peak_reverse_m_s=max(0.,-min(r['motion'][0] for r in after)))
            results.append(item)
            (a.output/'result.json').write_text(json.dumps(dict(kind='fixed_round20_development_interventions_no_learning',results=results),indent=2)+'\n')
            print(json.dumps({k:v for k,v in item.items() if k not in ('contact','events','network')}),flush=True)


if __name__=='__main__':main()
