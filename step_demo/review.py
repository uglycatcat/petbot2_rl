"""Streaming, CPU-only audit of actual training and evaluation artifacts."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from .config import load_config
from .evaluation import CompletionTracker, EvaluationSample, Stage


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    with Path(path).open() as file:
        for line in file:
            if line.strip(): yield json.loads(line)


def summarize(run, evaluation):
    run=Path(run);evaluation=Path(evaluation)
    origin=read(run/'origin.json');initial=read(run/'initialization.json')
    status=read(run/'status.json');latest=read(run/'latest.json');result=read(evaluation/'result.json')
    if origin['initialization']!='random' or origin['pretrained_models'] or initial['optimizer_states']!=0 or initial['frozen_parameters']!=0:
        raise ValueError('Invalid independent random initialization provenance')
    if status['status']!='complete' or latest['completed_updates']!=origin['training']['iterations']:
        raise ValueError('Incomplete training cannot become a valid round result')
    if result['status']!='complete' or result['completed_updates']!=latest['completed_updates']:
        raise ValueError('Evaluation must reload the completed final model')
    layout=read(evaluation/'layout.json'); hips=layout['hips'];wheels=layout['rear_wheels']
    config=load_config(); trackers={i:CompletionTracker(config.acceptance) for i in range(result['num_envs'])}
    terminals={r['env']:r for r in result['episodes']}
    if len(terminals)!=result['num_envs']: raise ValueError('Missing or duplicate first-episode evaluations')
    seen=set();phases=Counter();examples={};frames=0;counts=Counter()
    peaks=dict(estimator_error_m=0.,yaw_rad=0.,lateral_m=0.,hip_asymmetry_rad=0.,wheel_asymmetry_rad_s=0.,
               sampled_actual_wheel_rad_s=0.,head_ear_error_rad=0.,contact_force_n=0.,raw_action_abs=0.,
               estimator_orientation_error_rad=0.)
    action_count=0;action_outside_count=0
    for row in records(evaluation/'trace.jsonl'):
        frames+=1
        for local,i in enumerate(layout['trace_ids']):
            if row['episode'][local]!=1 or i in seen: continue
            q=row['joint_position'][local];v=row['joint_velocity'][local];com=row['com'][local]
            speed=math.sqrt(sum(x*x for x in row['velocity'][local]))
            phase=row['phase'][local];failure=row['failure'][local];completed=row['completed'][local]
            target=[row['hip_target'][local],row['wheel_target'][local]]
            raw=row['action'][local]
            if not all(math.isfinite(x) for x in raw):raise ValueError('Nonfinite raw policy action')
            action_count+=len(raw);action_outside_count+=sum(abs(x)>1.+1e-6 for x in raw)
            peaks['raw_action_abs']=max(peaks['raw_action_abs'],*(abs(x) for x in raw))
            if not all(math.isfinite(x) for x in (*q,*v,*com,*target)):
                raise ValueError('Nonfinite physical trajectory')
            if not (-1.-1e-5<=target[0]<=1.2+1e-5 and -1e-5<=target[1]<=25.+1e-5):
                raise ValueError('Executed targets exceed the source contract')
            peaks['estimator_error_m']=max(peaks['estimator_error_m'],math.dist(com,row['estimated_com'][local]))
            if 'estimated_quaternion' in row:
                qe,qt=row['estimated_quaternion'][local],row['true_quaternion'][local]
                denominator=math.sqrt(sum(x*x for x in qe)*sum(x*x for x in qt))
                cosine=abs(sum(x*y for x,y in zip(qe,qt)))/max(denominator,1e-12)
                peaks['estimator_orientation_error_rad']=max(peaks['estimator_orientation_error_rad'],
                    2*math.acos(min(1.,cosine)))
            peaks['yaw_rad']=max(peaks['yaw_rad'],abs(row['yaw'][local]))
            peaks['lateral_m']=max(peaks['lateral_m'],abs(com[1]))
            peaks['hip_asymmetry_rad']=max(peaks['hip_asymmetry_rad'],abs(q[hips[0]]-q[hips[1]]))
            peaks['wheel_asymmetry_rad_s']=max(peaks['wheel_asymmetry_rad_s'],abs(v[wheels[0]]-v[wheels[1]]))
            peaks['sampled_actual_wheel_rad_s']=max(peaks['sampled_actual_wheel_rad_s'],*(abs(v[j]) for j in wheels))
            peaks['head_ear_error_rad']=max(peaks['head_ear_error_rad'],*(abs(q[j]) for j,n in enumerate(layout['joints']) if n in ('head_1','left_ear_1','right_ear_1')))
            peaks['contact_force_n']=max(peaks['contact_force_n'],row['force_peak'][local])
            phases[str(phase)]+=1
            sample=EvaluationSample(layout['step_dt'],com[0],com[1],row['yaw'][local],speed,row['yaw_rate'][local],
                tuple(q[j] for j in hips),tuple(row['front_cleared'][local]),all(row['supported_top'][local]),
                row['front_hit'][local],failure==2,failure==3,failure==4)
            stage=trackers[i].update(sample)
            timed_out=row['elapsed'][local]+1e-5>=config.acceptance.timeout_s
            if failure or completed or timed_out:
                seen.add(i)
                if bool(completed)!=(stage==Stage.COMPLETE): raise ValueError(f'Completion disagreement for env {i}')
                if bool(completed)!=terminals[i]['success']: raise ValueError(f'Terminal artifact disagrees for env {i}')
                category='success' if completed else ('timeout' if timed_out and not failure else f'failure_{failure}')
                counts[category]+=1
                example=dict(env=i,elapsed_s=row['elapsed'][local],phase=phase,failure=failure,com=com,
                    estimated_com=row['estimated_com'][local],speed_m_s=speed,yaw_rad=row['yaw'][local],
                    targets=target,hip_actual=[q[j] for j in hips],wheel_actual=[v[j] for j in wheels],
                    front_lifted=row['front_lifted'][local],front_cleared=row['front_cleared'][local],
                    front_landed=row['front_landed'][local],supported_top=row['supported_top'][local])
                bucket=examples.setdefault(category,[])
                if len(bucket)<3: bucket.append(example)
    if len(seen)!=result['num_envs']: raise ValueError('Terminal physics frames missing from trace')
    if origin['training'].get('policy_distribution')=='beta' and action_outside_count:
        raise ValueError('Bounded policy produced out-of-range actions')
    train_counts=Counter();first_lift=0;front_landing=0
    for episode in records(run/'episodes.jsonl'):
        train_counts[str(episode['failure'])]+=1
        first_lift+=all(episode['front_lifted']);front_landing+=all(episode['front_landed'])
    updates=list(records(run/'updates.jsonl'))
    last=updates[-50:]
    return dict(schema='step_behavior_review_v1',valid=True,source_code_sha256=origin['code_sha256'],
        run_id=origin['run_id'],completed_updates=latest['completed_updates'],random_initialization_verified=True,
        evaluation=dict(episodes=result['num_envs'],successes=result['successes'],outcomes=dict(counts),
            phases=dict(phases),peaks=peaks,substep_actual_wheel_peak_rad_s=max(r['actual_wheel_peak'] for r in terminals.values()),
            scalar_completion_agrees=True,frames=frames,examples=examples,scene_randomization=False,
            raw_action_outside_fraction=action_outside_count/max(action_count,1)),
        training=dict(checkpoint_accounted_totals=status['totals'],recorded_episode_outcomes=dict(train_counts),
            recorded_both_front_lift=first_lift,recorded_both_front_landing=front_landing,
            last_update=updates[-1],last_50_mean_collect_s=statistics.mean(r['collect_s'] for r in last)),
        review_required=True,notes=['Recorded episodes include rollouts discarded by any same-run recovery.',
            'Fixed identical initial scenes do not establish generalization.',
            'Agent must inspect success and failure examples and justify the next code/config change.'])


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=summarize(a.run,a.evaluation)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(valid=report['valid'],evaluation=report['evaluation']['outcomes']),ensure_ascii=False))

if __name__=='__main__':main()
