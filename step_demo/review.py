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


def verify_physics(path, ids, hip_indices, contract=None, rear_wheel_indices=None):
    contract = contract or load_config()
    verifiers = [PhysicsVerifier(contract.acceptance, contract.terrain, hip_indices) for _ in ids]
    previous_counter = None
    samples = 0
    maxima = dict(pitch=0.,pitch_rate=0.,lateral=0.,yaw=0.,wheel_actual=0.,wheel_target=0.)
    phase_frames = Counter()
    peak_samples = {}
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
            if rear_wheel_indices is not None:
                # Include the failure/success substep, then stop this env using
                # the independent verifier. Never include its reset episode.
                phase_frames[str(verifier.phase)] += 1
                values = dict(pitch=abs(row['world_pitch'][i]),
                    pitch_rate=abs(row['world_pitch_rate'][i]), yaw=abs(row['yaw'][i]),
                    wheel_target=abs(row['wheel_target'][i]), lateral=abs(row['com'][i][1]),
                    wheel_actual=max(abs(row['joint_velocity'][i][j]) for j in rear_wheel_indices))
                for name, value in values.items():
                    if not math.isfinite(value): raise ValueError('Nonfinite physical metric')
                    if value > maxima[name]:
                        maxima[name] = value
                        peak_samples[name] = dict(env=ids[i],physics_step=counter,
                                                  elapsed_s=row['elapsed'][i],phase=verifier.phase)
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
    report = dict(samples=samples, all_terminal=all(v.result is not None for v in verifiers),
                episodes={str(i):v.result for i,v in zip(ids,verifiers)},
                task_labels_used_for_decisions=False)
    if rear_wheel_indices is not None:
        report['physical_metrics'] = dict(source='physics_substeps',sample_period_s=.002,
            maxima=maxima,peak_samples=peak_samples,phase_frames=dict(phase_frames))
    return report


def summarize(run, evaluation):
    run, evaluation = Path(run), Path(evaluation)
    result = json.loads((evaluation/'result.json').read_text())
    layout = json.loads((evaluation/'layout.json').read_text())
    physics = verify_physics(evaluation/'physics.jsonl.gz', layout['trace_ids'], layout['hips'],
                             rear_wheel_indices=layout['rear_wheels'])
    if not physics['all_terminal']: raise ValueError('Incomplete independent physical verification')
    for record in result['episodes']:
        checked = physics['episodes'][str(record['env'])]
        if (checked['success'], checked['failure']) != (record['success'],record['failure']):
            raise ValueError('Reported result disagrees with physical replay')
    metrics = physics['physical_metrics']
    failures = Counter(str(r['failure']) for r in result['episodes'] if not r['success'])
    return dict(schema='step_review_v2', completed_updates=result['completed_updates'],
        evaluation=dict(outcomes=dict(total=result['num_envs'],successes=result['successes'],failures=dict(failures)),
                        independent_physics=physics, phase_frames=metrics['phase_frames'],
                        maxima=metrics['maxima'],metrics_source=metrics['source'],
                        metrics_sample_period_s=metrics['sample_period_s'],peak_samples=metrics['peak_samples']),
        trained_from_random=json.loads((run/'initialization.json').read_text()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('evaluation',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.run,a.evaluation),indent=2)+'\n')


if __name__=='__main__':main()
