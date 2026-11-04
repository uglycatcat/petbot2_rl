"""Basic command response by settled windows, separate from cycle completion."""
import math


def basic_response(traces, count, complete, dt):
    settle_s, measure_s = .5, .5
    pending = [None]*count
    windows = [[] for _ in range(count)]
    coverage = [set() for _ in range(count)]
    required = {'forward', 'reverse', 'left', 'right', 'stop'}

    def finish(i):
        item = pending[i]
        pending[i] = None
        if item is None or len(item['motion'])*dt < measure_s:
            return
        request = item['request']
        actual = [sum(v[j] for v in item['motion'])/len(item['motion']) for j in range(2)]
        passed = []
        for j, (target, measured) in enumerate(zip(request, actual)):
            deadband, zero_limit = ((.025, .05) if j == 0 else (.15, .15))
            if abs(target) >= deadband:
                signed = math.copysign(1, target)*measured
                passed.append(.25*abs(target) <= signed <= max(2*abs(target), 2*zero_limit))
                coverage[i].add(('forward' if target > 0 else 'reverse') if j == 0
                                else ('left' if target > 0 else 'right'))
            else:
                passed.append(abs(measured) <= zero_limit)
        if max(map(abs, request)) < 1e-6:
            coverage[i].add('stop')
        windows[i].append(dict(request=list(request), actual_mean=actual,
                               measured_s=len(item['motion'])*dt, passed=all(passed)))

    for frame in traces:
        for i in range(count):
            active = frame['active'][i] and frame['phase_before'][i] == 2 and not frame['failure'][i]
            if not active:
                finish(i)
                continue
            command = tuple(round(v, 6) for v in frame['requested_command'][i])
            if pending[i] is None or pending[i]['request'] != command:
                finish(i)
                pending[i] = dict(request=command, start=frame['t'], motion=[])
            if frame['t']-pending[i]['start'] >= settle_s-1e-6:
                pending[i]['motion'].append(frame['motion'][i])
    for i in range(count):
        finish(i)
    cases = [dict(env=i, complete=bool(complete[i]), coverage=sorted(coverage[i]),
                  sufficient_coverage=required <= coverage[i], windows=windows[i],
                  passed=bool(complete[i] and required <= coverage[i]
                              and all(w['passed'] for w in windows[i]))) for i in range(count)]
    groups = {}
    for case in cases:
        for window in case['windows']:
            key = tuple(window['request'])
            groups.setdefault(key, []).append(window)
    window_summary = []
    for request, items in sorted(groups.items()):
        seconds = sum(x['measured_s'] for x in items)
        window_summary.append(dict(request=list(request), windows=len(items),
            passed=sum(x['passed'] for x in items), measured_s=seconds,
            actual_mean=[sum(x['actual_mean'][j]*x['measured_s'] for x in items)/seconds for j in range(2)]))
    return dict(basic_control_pass=sum(c['passed'] for c in cases),
                basic_response=dict(protocol='settled_signed_response_v1', settle_s=settle_s,
                    minimum_measure_s=measure_s, response_fraction=[.25, 2.],
                    zero_command_mean_limits=[.05, .15], cases=cases, window_summary=window_summary))
