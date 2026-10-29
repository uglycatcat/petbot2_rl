"""Audit a completed scratch batch, including deliberate skipped evaluations."""
import hashlib,json,math,sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upright.model import load_model,source_fingerprint

out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text())
status=json.loads((out/'eval_status.json').read_text());spec=load_model()
def finite(v):
    if isinstance(v,torch.Tensor):return bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(tuple,list)):return all(map(finite,v))
    return not isinstance(v,float) or math.isfinite(v)

report=dict(execution_valid=True,policy_passed=False,source_sha256=source_fingerprint(),
    commit=plan['commit'],execution=[],runs=[])
for name,item in status.items():
    if item['status']=='skipped':
        assert item['reason'];report['execution'].append(dict(name=name,status='skipped',reason=item['reason']));continue
    exit=item['exit'];assert item['status']=='complete' and exit['exit_code']==0
    assert not any(exit.get(k) for k in ('guard_stopped','physics_error','runtime_error'))
    report['execution'].append(dict(name=name,status='complete',elapsed_s=exit['elapsed_s'],peak_mib=exit['sampled_peak_mib']))
for v in plan['variants']:
    root=out.parent/v['name'];run=root/'logs/scratch';origin=json.loads((run/'scratch_origin.json').read_text())
    initial=torch.load(run/'initial_state.pt',weights_only=False,map_location='cpu')
    final=torch.load(run/'model_999.pt',weights_only=False,map_location='cpu')
    cfg=json.loads((run/'config.json').read_text());audit=json.loads((run/'initialization.json').read_text())
    assert not cfg['working_tree_dirty'] and cfg['commit']==plan['commit'] and origin['initialization']=='random'
    assert not origin['pretrained_models'] and not cfg['args']['checkpoint'] and final['scratch_origin']==origin
    assert audit['serialized_weight_reads_forbidden'] and audit['initial_optimizer_empty']
    snapshots=[]
    for idx in [*range(0,1000,100),999]:
        f=run/f'model_{idx}.pt';d=torch.load(f,weights_only=False,map_location='cpu')
        assert d['iter']==idx and finite(d) and d['scratch_origin']==origin
        snapshots.append(dict(iteration=idx,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            bootstrap=d['task_state']['curriculum']['bootstrap_stage']))
    changed=[k for k,t in final['actor_state_dict'].items() if not torch.equal(t,initial['actor_state_dict'][k])]
    assert all(k in changed for k in final['actor_state_dict'] if k.startswith('mlp.'))
    assert not any(k.startswith(('parent_balance.','balance_reflex.')) for k in final['actor_state_dict'])
    assert len(list(run.glob('events.out.tfevents.*')))==1,'Unexpected restart/event stream'
    acc=EventAccumulator(str(run),size_guidance={'scalars':0});acc.Reload();scalars={};loss_steps=set()
    for tag in acc.Tags()['scalars']:
        rows=acc.Scalars(tag);assert all(math.isfinite(r.value) for r in rows)
        if tag.startswith('Loss/'):loss_steps.update(r.step for r in rows)
        if tag.startswith(('Course/','Cycle/episode_','Episode_Reward/','Train/mean_reward','Policy/mean_noise_std')):
            scalars[tag]=dict(first=rows[0].value,last=rows[-1].value,max=max(x.value for x in rows),last100_mean=sum(x.value for x in rows[-100:])/len(rows[-100:]))
    assert loss_steps==set(range(1000))
    course=final['task_state']['curriculum'];history=course['bootstrap_history']
    evaluated=[]
    for scenario in plan['evaluations']:
        label=v['name']+'_eval_'+scenario['name'];item=status[label]
        if item['status']=='skipped':
            assert not (root/('logs/eval_'+scenario['name'])).exists();continue
        folder=root/('logs/eval_'+scenario['name']);result=json.loads((folder/'evaluation.json').read_text());trace=json.loads((folder/'trace.json').read_text())
        assert result['count']==64 and result['unfinished']==0
        counts={};samples=[];peaks=dict(wheel_target=0.,wheel_actual=0.,rear_effort=0.,hip_target_difference=0.,front_height=0.)
        for frame in trace:
            for e,valid in enumerate(frame['valid']):
                if not valid:continue
                targets=frame['wheel_target'][e];q=frame['position_target'][e];effort=frame['actuator_effort'][e]
                names=('head','left_ear','right_ear','left_frame','right_frame','left_rear_wheel','right_rear_wheel')
                assert max(map(abs,targets))<=25.0001
                assert all(abs(x)<=spec['actuators'][k]['effort']+1e-4 for k,x in zip(names,effort))
                assert abs(q[3]-q[4])<1e-6
                for k,x in zip(names[:5],q):
                    lo,hi=spec['joints'][k]['range'];assert lo-1e-5<=x<=hi+1e-5
                peaks['wheel_target']=max(peaks['wheel_target'],*map(abs,targets));peaks['wheel_actual']=max(peaks['wheel_actual'],*map(abs,frame['wheel_measured'][e]))
                peaks['rear_effort']=max(peaks['rear_effort'],*map(abs,effort[-2:]));peaks['front_height']=max(peaks['front_height'],frame['front_z'][e])
                code=frame['failure_code'][e]
                if code:counts[str(code)]=counts.get(str(code),0)+1
        for e in (0,15):
            valid=[f for f in trace if f['valid'][e]];start=next(f for f in valid if f['switched'][e]);last=valid[-1]
            def sample(f):return {k:f[k][e] for k in ('phase','phase_time','requested','command','motion_feedback','joint_position','position_target','gravity','front_z','failure_code')}
            samples.append(dict(env=e,takeover=sample(start),terminal=sample(last)))
        evaluated.append(dict(name=scenario['name'],**{k:result[k] for k in ('takeover','balance','complete','failure','unfinished')},
            basic=result['functional']['functional_pass'],failure_codes=counts,peaks=peaks,samples=samples))
    report['runs'].append(dict(name=v['name'],updates=1000,restarts=0,random_origin=origin,
        all_checkpoints_finite=True,checkpoints=snapshots,changed_actor_tensors=changed,
        bootstrap=course['bootstrap_stage'],bootstrap_trials=sum(x['trials'] for x in history),
        bootstrap_success=sum(x['success'] for x in history),scalars=scalars,evaluations=evaluated))
(out/'final_review.json').write_text(json.dumps(report,indent=2)+'\n')
for run in report['runs']:
    print(json.dumps({k:run[k] for k in ('name','updates','restarts','bootstrap','bootstrap_trials','bootstrap_success','scalars','evaluations')},indent=2))
