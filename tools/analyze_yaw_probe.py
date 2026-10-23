"""Compare diagnostic pulses; report response and pre-intervention agreement."""
import argparse
import json
import pathlib
import statistics


def summarize(rows):
    episodes = {}
    for row in rows:
        for i, valid in enumerate(row['valid']):
            if not valid or not row['probe_active'][i] or row['probe_pre_time'][i] < 1.2:
                continue
            left, right = [v * 25. for v in row['applied'][i][1:]]
            wl, wr = row['wheel_interval_mean'][i]
            tl, tr = row['torque_estimate_mean'][i]
            values = dict(target_halfdiff=(right-left)/2, measured_halfdiff=(wr-wl)/2,
                          measured_yaw=row['heading_rate'][i], requested_yaw=row['raw'][i][1],
                          balanced=float(row['balanced'][i]), front_contact=float(row['front_contact'][i]),
                          rear_fraction=row['rear_fraction'][i], body_contact=float(row['body_contact'][i]),
                          mean_abs_wheel_error=(abs(left-wl)+abs(right-wr))/2,
                          torque_estimate_halfdiff=(tr-tl)/2,
                          torque_estimate_at_limit=float(max(row['torque_estimate_abs_peak'][i]) >= .379),
                          target_at_limit=float(max(abs(left),abs(right)) >= 24.99))
            episodes.setdefault(i, []).append(values)
    means = {str(i): {key: statistics.mean(v[key] for v in values) for key in values[0]}
             for i, values in episodes.items()}
    return dict(episodes=len(means), samples=sum(map(len, episodes.values())), per_episode=means,
                mean={key: statistics.mean(v[key] for v in means.values()) for key in next(iter(means.values()), {})})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', type=pathlib.Path, nargs='+', help='Baseline folder first')
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    baseline = None
    baseline_config = None
    report = dict(diagnostic_only=True, selection='valid, pulse active, phase time >=1.2s; equal episode weighting',
                  limits='Finite pulse with policy frame/common feedback; not task qualification or open-loop plant identification', runs={})
    for folder in args.folders:
        config = json.loads((folder/'config.json').read_text())
        rows = json.loads((folder/'diagnostic_trace.json').read_text())
        metadata = json.loads((folder/'yaw_probe.json').read_text())
        result = summarize(rows)
        if baseline is None:
            assert metadata['half_difference_rad_s'] is None, 'First input must be policy baseline'
            baseline, baseline_config = rows, config
        assert config['contract'] == baseline_config['contract']
        for key in ('checkpoint', 'seed', 'num_envs'):
            assert config['args'][key] == baseline_config['args'][key], key
        first_pulse = set()
        maximum, samples = 0., 0
        for row, ref in zip(rows, baseline):
            assert row['t'] == ref['t']
            for i, valid in enumerate(row['valid']):
                if row['probe_active'][i] or ref['probe_active'][i]:
                    first_pulse.add(i)
                if i in first_pulse or not (valid and ref['valid'][i]):
                    continue
                for key in ('forward', 'heading_rate', 'front_z'):
                    maximum = max(maximum, abs(row[key][i]-ref[key][i]))
                for key in ('applied', 'wheel_measured', 'pos'):
                    maximum = max(maximum, max(abs(x-y) for x, y in zip(row[key][i], ref[key][i])))
                samples += 1
        result.update(protocol=metadata, pre_pulse_max_abs_difference=maximum, pre_pulse_compared_samples=samples)
        report['runs'][folder.name] = result
        print(folder.name, json.dumps(dict(episodes=result['episodes'], pre_pulse_max_abs_difference=maximum, **result['mean'])))
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
