"""Review completed scratch engineering evidence; execution is not policy success.

Usage inside container: python checks/scratch_results.py CONTROL_DIRECTORY
The control directory and its sibling fixed run directories must be mounted.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upright.model import load_model,source_fingerprint

out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text())
status=json.loads((out/'eval_status.json').read_text());spec=load_model()
assert all(v['status']=='complete' and not any(v['exit'].get(k) for k in ('guard_stopped','physics_error','runtime_error'))
           and v['exit']['exit_code']==0 for v in status.values()), 'Execution failure must be resolved first'
def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(map(finite,value))
    return not isinstance(value,float) or math.isfinite(value)

report=dict(passed=False,tested_commit=plan['commit'],source_sha256=source_fingerprint(),runs=[],
    hardware_contract_pass=True,policy_acceptance=False,sim2sim_passed=False)
initial_hashes=[]
for variant in plan['variants']:
    root=out.parent/variant['name'];folder=root/'logs/scratch'
    initial=torch.load(folder/'initial_state.pt',map_location='cpu',weights_only=False)
    final=torch.load(folder/f"model_{variant['iterations']-1}.pt",map_location='cpu',weights_only=False)
    origin=json.loads((folder/'scratch_origin.json').read_text())
    audit=json.loads((folder/'initialization.json').read_text())
    assert audit['serialized_weight_reads_forbidden'] and audit['all_actor_parameters_trainable'] and audit['initial_optimizer_empty']
    initial_hashes.append(audit['initial_actor_sha256'])
    assert final['scratch_origin']==origin and origin['pretrained_models']==[]
    assert final['iter']==variant['iterations']-1 and finite(final)
    actor=final['actor_state_dict'];old=initial['actor_state_dict']
    assert not any(k.startswith(('parent_balance.','balance_reflex.','adapter.','rise_adapter.','yaw_adapter.')) for k in actor)
    changed=[k for k in actor if not torch.equal(actor[k],old[k])]
    assert all(k in changed for k in actor if k.startswith('mlp.')), 'Actor parameters did not learn'
    loss_steps=set();tags={}
    for event in folder.glob('events.out.tfevents.*'):
        acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
        for tag in acc.Tags()['scalars']:
            rows=acc.Scalars(tag);assert all(math.isfinite(x.value) for x in rows)
            if tag.startswith('Loss/'):loss_steps.update(x.step for x in rows)
            tags[tag]=len(rows)
    assert loss_steps==set(range(variant['iterations'])), sorted(loss_steps)
    resume=None
    if variant.get('stop_after'):
        point=torch.load(folder/f"model_{variant['stop_after']-1}.pt",map_location='cpu',weights_only=False)
        start=[int(x['step']) for x in point['optimizer_state_dict']['state'].values()]
        end=[int(x['step']) for x in final['optimizer_state_dict']['state'].values()]
        delta=(variant['iterations']-variant['stop_after'])*5*4
        assert end and all(b-a==delta for a,b in zip(start,end))
        resume=dict(same_origin=point['scratch_origin']==origin,additional_optimizer_steps=delta,physical_episode_restore=False)
        assert resume['same_origin']
    evaluations=[]
    for scenario in plan['evaluations']:
        evaluation=root/('logs/eval_'+scenario['name'])
        result=json.loads((evaluation/'evaluation.json').read_text());trace=json.loads((evaluation/'trace.json').read_text())
        manifest=json.loads((evaluation/'onnx/manifest.json').read_text())
        assert result['count']==64 and result['unfinished']==0 and result['takeover']>0
        assert manifest['checkpoint_sha256']==hashlib.sha256((folder/f"model_{variant['iterations']-1}.pt").read_bytes()).hexdigest()
        assert manifest['onnx_max_abs_error']<=1e-5 and not manifest['sim2sim_passed']
        maximum_wheel=maximum_effort=target_peak=0.;phases=set();failure_codes={}
        for frame in trace:
            for i,valid in enumerate(frame['valid']):
                if not valid:continue
                phases.add(frame['phase'][i]);target=frame['wheel_target'][i]
                assert max(map(abs,target))<=25.0001
                target_peak=max(target_peak,*map(abs,target))
                maximum_wheel=max(maximum_wheel,*map(abs,frame['wheel_measured'][i]))
                efforts=frame['actuator_effort'][i]
                names=('head','left_ear','right_ear','left_frame','right_frame','left_rear_wheel','right_rear_wheel')
                assert all(abs(x)<=spec['actuators'][name]['effort']+1e-4 for name,x in zip(names,efforts))
                maximum_effort=max(maximum_effort,*map(abs,efforts[-2:]))
                q=frame['position_target'][i]
                assert abs(q[3]-q[4])<1e-6
                for name,x in zip(names[:5],q):
                    lo,hi=spec['joints'][name]['range'];assert lo-1e-5<=x<=hi+1e-5
                code=frame['failure_code'][i]
                if code:failure_codes[str(code)]=failure_codes.get(str(code),0)+1
        evaluations.append(dict(scenario=scenario['name'],complete=result['complete'],balance=result['balance'],
            failures=result['failure'],phases=sorted(phases),failure_frames=failure_codes,
            basic=result['functional']['functional_pass'],wheel_target_peak=target_peak,
            actual_wheel_peak=maximum_wheel,rear_effort_peak=maximum_effort,onnx_error=manifest['onnx_max_abs_error']))
    report['runs'].append(dict(name=variant['name'],changed_actor_tensors=changed,updates=len(loss_steps),
        bootstrap=final['task_state']['curriculum']['bootstrap_stage'],resume=resume,evaluations=evaluations))
assert len(set(initial_hashes))==len(initial_hashes), 'Independent seeds have identical initialization'
report.update(passed=True,independent_initial_actor_hashes=initial_hashes)
(out/'engineering_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
