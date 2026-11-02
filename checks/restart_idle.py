"""Reproduce zero-input drift and check braking from contact, not just completion.

No GUI, no training or state resets within a call. --verify-only checks saved traces.
"""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from petbot_s2s.restart import RestartSimulation
from petbot_s2s.restart_core.controller import Phase
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reference',type=Path)
parser.add_argument('--verify-only',action='store_true')
args=parser.parse_args()
out=args.output
if not args.verify_only:out.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(1)
result={}
for name,speed in ([] if args.verify_only else [('stationary',0.),('forward_release',.6),('reverse_release',-.6)]):
    sim=RestartSimulation(root/'policies/restart20_s102');c=sim.controller
    c.set_policy_limits(.6,.5)
    rows=[];balanced=None;arm=False;land=False;complete=None
    while sim.data.time<42:
        t=float(sim.data.time)
        b=t-balanced if balanced is not None else -1
        cmd=speed if 2<=b<8 else 0.
        c.set_command(cmd,0.)
        if t>=3 and not arm:c.toggle();arm=True
        if int(c.control.phase[0])==Phase.BALANCE and balanced is None:balanced=t
        if b>=20 and not land and c.armed:c.toggle();land=True
        sim.step();r=sim.record();rows.append(r)
        if c.cycles and complete is None:complete=r['time']
    def stats(a):
        if not a:return None
        return dict(n=len(a),v_mean=float(np.mean([r['motion'][0] for r in a])),v_min=min(r['motion'][0] for r in a),v_max=max(r['motion'][0] for r in a),operator_max=max(abs(x) for r in a for x in r['operator_request']),raw_mean=float(np.mean([r['raw'][0] for r in a])),reference_mean=float(np.mean([r['control_reference'][0] for r in a])),wheel_mean=float(np.mean([r['wheel_target'] for r in a])),phase=sorted(set(r['phase'] for r in a)))
    result[name]=dict(balanced_at=balanced,completed_at=complete,cycles=c.cycles,interruptions=c.interruptions,events=c.events,
        zero_balance=stats([r for r in rows if balanced is not None and 12<r['time']-balanced<20 and r['phase']==2]),
        zero_ordinary=stats([r for r in rows if complete is not None and r['time']>complete+2]),
        source_model=sim.model_sha256,network=c.manifest['networks'],
        max_target=max(abs(x) for r in rows for x in r['wheel_target']),max_measured=max(abs(x) for r in rows for x in r['wheel_measured']))
    trace=out/(name+'.jsonl');trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result[name]['trace_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
    print(name,json.dumps(result[name]),flush=True)
if args.verify_only:
    result=json.loads((out/'summary.json').read_text())
else:
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
checks={}
for name,case in result.items():
    assert case['cycles']==1 and case['interruptions']==0, (name,case)
    idle=case['zero_ordinary']
    assert idle['operator_max']==0 and idle['raw_mean']==0 and idle['wheel_mean']==0, (name,idle)
    assert max(abs(idle['v_min']),abs(idle['v_max']))<.005,(name,idle)
    rows=[json.loads(line) for line in (out/(name+'.jsonl')).read_text().splitlines()]
    assert hashlib.sha256((out/(name+'.jsonl')).read_bytes()).hexdigest()==case['trace_sha256']
    checks[name]={'zero_ordinary_pass':True,'balance_drift_m_s':case['zero_balance']['v_mean']}
    def landing_metrics(trace):
        land=next(r for r in trace if r['phase']==Phase.LAND)
        contact=next(r for r in trace if r['time']>=land['time'] and r['four_fraction']>0)
        handoff=next(r for r in trace if r['phase']==Phase.HANDOFF)
        after=[r for r in trace if r['time']>=contact['time']]
        last=max(i for i,r in enumerate(after) if abs(r['motion'][0])>.01)
        stop=after[last+1]
        return dict(first_four_contact=contact['time'],handoff=handoff['time'],stopped=stop['time'],
                    contact_to_stop_s=stop['time']-contact['time'],handoff_to_stop_s=stop['time']-handoff['time'],
                    backward_travel_m=sum(max(0.,-r['motion'][0])*.03 for r in after[:last+1]),
                    peak_backward_m_s=-min(r['motion'][0] for r in after),
                    complete=next(r['time'] for r in trace if r['cycles']))
    metrics=landing_metrics(rows)
    checks[name]['landing']=metrics
    # The transition record still captures actual velocity; the next control
    # step must use the current request, including an unchanged zero.
    during=[r for r in rows if r['phase']==Phase.HANDOFF and r['time']>metrics['handoff']+.001]
    assert during and all(r['raw']==[0.,0.] for r in during),name
    settled=[r for r in during if r['time']>metrics['handoff']+.33]
    assert settled and all(r['wheel_target']==[0.,0.] for r in settled),name
    assert metrics['handoff_to_stop_s']<.65 and metrics['stopped']<metrics['complete'],(name,metrics)
    if args.reference:
        old=[json.loads(line) for line in (args.reference/(name+'.jsonl')).read_text().splitlines()]
        through=[i for i,r in enumerate(rows) if r['time']<=metrics['handoff']]
        assert all(rows[i]==old[i] for i in through), name
        first=rows[len(through)]
        assert first['wheel_target']==old[len(through)]['wheel_target'],name
        checks[name]['unchanged_through_handoff_entry_frames']=len(through)
        checks[name]['first_ordinary_targets_inherited']=True
        checks[name]['before']=landing_metrics(old)
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
