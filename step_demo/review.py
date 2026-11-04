"""Host-side physical replay and concise policy behavior review for STEP v2."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from .config import load_config
from .evaluation import PhysicsVerifier


def records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def verify_physics(path, ids, hip_indices, contract=None):
    contract = contract or load_config()
    verifiers = [PhysicsVerifier(contract.acceptance, contract.terrain, hip_indices) for _ in ids]
    previous_counter = None
    samples = 0
    for row in records(path):
        if row.get('schema') != 'step_trace_v2': raise ValueError('Unknown physics trace schema')
        counter = row['physics_step']
        if previous_counter is not None and counter != previous_counter+1:
            raise ValueError('Missing physics substeps; continuous hold cannot be verified')
        previous_counter = counter
        for i, verifier in enumerate(verifiers):
            if verifier.result is not None: continue
            if row['episode'][i] != 1: raise ValueError('First episode ended without independent certification')
            verifier.update(row,i)
            expected = row['task']
            if verifier.phase != expected['phase'][i] or verifier.reason != expected['failure'][i]:
                raise ValueError(f'Physical replay differs at step {counter}, env {ids[i]}: '
                                 f'{verifier.phase}/{verifier.reason} vs {expected["phase"][i]}/{expected["failure"][i]}')
            if verifier.flight != expected['flight_count'][i]:
                raise ValueError('Flight event count differs from physical replay')
            if abs(verifier.distance-expected['drive_distance'][i]) > 2e-6:
                raise ValueError('Drive distance differs from physical replay')
            if verifier.hold != expected['hold_ticks'][i]: raise ValueError('Continuous substep hold differs')
        samples += 1
        if all(v.result is not None for v in verifiers): break
    return dict(samples=samples, all_terminal=all(v.result is not None for v in verifiers),
                episodes={str(i):v.result for i,v in zip(ids,verifiers)},
                task_labels_used_for_decisions=False)


def summarize(run, evaluation):
    run, evaluation = Path(run), Path(evaluation)
    result = json.loads((evaluation/'result.json').read_text())
    layout = json.loads((evaluation/'layout.json').read_text())
    physics = verify_physics(evaluation/'physics.jsonl.gz', layout['trace_ids'], layout['hips'])
    if not physics['all_terminal']: raise ValueError('Incomplete independent physical verification')
    for record in result['episodes']:
        checked = physics['episodes'][str(record['env'])]
        if (checked['success'], checked['failure']) != (record['success'],record['failure']):
            raise ValueError('Reported result disagrees with physical replay')
    maxima = dict(pitch=0.,pitch_rate=0.,lateral=0.,yaw=0.,wheel_actual=0.,wheel_target=0.)
    phase_frames = Counter()
    for row in records(evaluation/'trace.jsonl'):
        for i, episode in enumerate(row['episode']):
            if episode != 1: continue
            phase_frames[str(row['task']['phase'][i])] += 1
            for name, field in [('pitch','world_pitch'),('pitch_rate','world_pitch_rate'),('yaw','yaw'),('wheel_target','wheel_target')]:
                maxima[name] = max(maxima[name],abs(row[field][i]))
            maxima['lateral'] = max(maxima['lateral'],abs(row['com'][i][1]))
            maxima['wheel_actual'] = max(maxima['wheel_actual'],max(abs(row['joint_velocity'][i][j]) for j in layout['rear_wheels']))
    failures = Counter(str(r['failure']) for r in result['episodes'] if not r['success'])
    return dict(schema='step_review_v2', completed_updates=result['completed_updates'],
        evaluation=dict(outcomes=dict(total=result['num_envs'],successes=result['successes'],failures=dict(failures)),
                        independent_physics=physics, phase_frames=dict(phase_frames), maxima=maxima),
        trained_from_random=json.loads((run/'initialization.json').read_text()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('evaluation',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.run,a.evaluation),indent=2)+'\n')


if __name__=='__main__':main()
