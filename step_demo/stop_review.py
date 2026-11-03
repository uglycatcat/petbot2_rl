"""Independent scalar audit for isolated platform stopping, never climb scores."""
import json
import math
import struct
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def float32(value):
    """Match the physical timer's precision without importing Torch on host."""
    return struct.unpack('f',struct.pack('f',value))[0]


def summarize_auxiliary(run, evaluation):
    run, evaluation = Path(run), Path(evaluation)
    origin, initial = read(run/'origin.json'), read(run/'initialization.json')
    status, latest = read(run/'status.json'), read(run/'latest.json')
    result, layout = read(evaluation/'result.json'), read(evaluation/'layout.json')
    if (origin['initialization'] != 'random' or origin['pretrained_models']
            or initial['optimizer_states'] or initial['frozen_parameters']):
        raise ValueError('Auxiliary learning must start independently from random weights')
    if (status['status'] != 'complete' or result['status'] != 'complete'
            or latest['completed_updates'] != origin['training']['iterations']
            or result['completed_updates'] != latest['completed_updates']):
        raise ValueError('Incomplete auxiliary training or mismatched evaluation')
    restored=[read(p) for p in evaluation.glob('restored_*.json')]
    if len(restored)!=1 or restored[0]['actor_sha256']!=latest['actor_sha256']:
        raise ValueError('Auxiliary evaluation did not reload the final actor')
    a = origin['task']['acceptance']
    dt = layout['step_dt']; hold = {}; base_seen = set(); seen = set()
    counts = Counter(); groups = {i:dict(episodes=0,stopped=0,held=0) for i in range(4)}
    initial_speeds = {}; examples = []; max_target = 0.; max_actual_wheel = 0.
    raw_count = 0; raw_outside = 0; raw_peak = 0.
    max_estimator_error=0.; max_hip_request=0.; max_pitch=0.; max_pitch_rate=0.
    terminals = {r['env']:r for r in result['episodes']}
    extra = result.get('hold_after_success_s',0.)
    with (evaluation/'trace.jsonl').open() as file:
        for line in file:
            row = json.loads(line)
            for k, i in enumerate(layout['trace_ids']):
                if row['episode'][k] != 1 or i in seen:
                    continue
                q, v = row['joint_position'][k], row['joint_velocity'][k]
                speed = math.sqrt(sum(x*x for x in row['velocity'][k]))
                initial_speeds.setdefault(i,speed)
                if not all(math.isfinite(x) for x in (*q,*v,speed)):
                    raise ValueError('Nonfinite auxiliary physical trajectory')
                if row['intent'][k] != 3:
                    raise ValueError('Auxiliary command changed')
                if not (-1.00001 <= row['hip_target'][k] <= 1.20001 and -.00001 <= row['wheel_target'][k] <= 25.00001):
                    raise ValueError('Auxiliary executed target outside source limits')
                max_target = max(max_target,row['wheel_target'][k])
                max_estimator_error=max(max_estimator_error,math.dist(row['com'][k],row['estimated_com'][k]))
                raw=row['action'][k]
                if not all(math.isfinite(v) for v in raw):raise ValueError('Nonfinite raw policy action')
                raw_count += len(raw); raw_outside += sum(abs(v)>1.+1e-6 for v in raw)
                raw_peak=max(raw_peak,*(abs(v) for v in raw))
                request=[max(-1.,min(1.,v)) for v in raw]
                if 'requested_action' in row and any(abs(a-b)>1e-6 for a,b in zip(request,row['requested_action'][k])):
                    raise ValueError('Physical action request does not match existing source-limit clipping')
                hip_action=request[0]
                max_hip_request=max(max_hip_request,abs(hip_action*(1.2 if hip_action>=0 else 1.)))
                max_pitch=max(max_pitch,abs(row['world_pitch'][k]))
                max_pitch_rate=max(max_pitch_rate,abs(row['world_pitch_rate'][k]))
                max_actual_wheel = max(max_actual_wheel,*(abs(v[j]) for j in layout['rear_wheels']))
                good = (all(row['supported_top'][k]) and row['failure'][k] == 0 and row['front_flight_count'][k] == [0,0]
                    and abs(row['com'][k][1]) <= a['line_tolerance_m']
                    and abs(row['yaw'][k]) <= a['heading_tolerance_rad']
                    and max(abs(q[j]) for j in layout['hips']) <= a['hip_tolerance_rad']
                    and speed <= a['stop_speed_m_s']
                    and abs(row['yaw_rate'][k]) <= a['stop_yaw_rate_rad_s'])
                hold[i] = float32(hold.get(i,0.)+float32(dt)) if good else 0.
                if abs(hold[i]-row['auxiliary_hold'][k]) > 1e-4:
                    raise ValueError('Independent auxiliary hold timer disagrees')
                if hold[i]+1e-6 >= a['stop_hold_s']:
                    base_seen.add(i)
                completed = float32(hold[i]+float32(1e-7)) >= float32(a['stop_hold_s']+extra)
                if completed != row['completed'][k]:
                    raise ValueError('Auxiliary success disagreement')
                failure = row['failure'][k]
                timeout = row['elapsed'][k]+1e-5 >= a['timeout_s']
                if failure or completed or timeout:
                    seen.add(i)
                    if completed != terminals[i]['success']:
                        raise ValueError('Auxiliary terminal artifact disagrees')
                    group = groups[i % 4]
                    group['episodes'] += 1; group['stopped'] += i in base_seen; group['held'] += completed
                    key = 'success' if completed else f'failure_{failure}' if failure else 'timeout'
                    counts[key] += 1
                    if len(examples) < 12:
                        examples.append(dict(env=i,group=i%4,elapsed=row['elapsed'][k],outcome=key,
                            speed=speed,hip=[q[j] for j in layout['hips']],hip_request=hip_action*(1.2 if hip_action>=0 else 1.),
                            hip_target=row['hip_target'][k],wheel_request=(request[1]+1.)*12.5,
                            raw_policy_action=raw,
                            wheel_target=row['wheel_target'][k],wheel_actual=[v[j] for j in layout['rear_wheels']],
                            support=row['supported_top'][k],position=row['com'][k],yaw=row['yaw'][k],
                            flight_count=row['front_flight_count'][k]))
    if len(seen) != result['num_envs']:
        raise ValueError('Missing auxiliary terminal trajectories')
    if origin['training'].get('policy_distribution')=='beta' and raw_outside:
        raise ValueError('Bounded Beta policy produced out-of-range actions')
    n = result['num_envs']
    passed = (n == 64 and extra >= 2. and counts['success'] >= 61
              and all(g['episodes']==16 and g['held']>=15 for g in groups.values()))
    return dict(schema='step_auxiliary_stop_review_v1',valid=True,task='platform_stop_aux',
        full_task_successes=None,counts_as_climb_result=False,completed_updates=latest['completed_updates'],
        random_initialization_verified=True,can_transfer_weights=False,
        evaluation=dict(outcomes=dict(counts),successes=counts['success'],episodes=n,
            base_stop_count=len(base_seen),extra_physics_hold_s=extra,groups=groups,
            initial_measured_speed_by_env=initial_speeds,examples=examples,
            max_wheel_target=max_target,max_sampled_actual_wheel=max_actual_wheel,
            max_estimator_error_m=max_estimator_error,max_requested_hip_rad=max_hip_request,
            max_world_pitch_rad=max_pitch,max_world_pitch_rate_rad_s=max_pitch_rate,
            raw_policy_action_peak=raw_peak,raw_action_outside_fraction=raw_outside/max(raw_count,1),
            scalar_hold_agrees=True,auxiliary_gate_passed=passed),
        review_required=True,notes=['Auxiliary STOP is a known command, not demonstrated deployed phase recognition.',
            'Both seeds must pass; no auxiliary weights/states may enter full-task training.'])
