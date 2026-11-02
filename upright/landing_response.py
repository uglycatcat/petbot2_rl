"""Describe landing and ordinary transfer separately; never extend actor ownership."""
import math


def landing_response(traces, count, dt):
    cases = []
    for i in range(count):
        frames = [f for f in traces if f['active'][i]]
        land = [f for f in frames if f['phase_before'][i] == 3]
        transfer = next((f for f in frames if f['phase'][i] == 4), None)
        ordinary = [f for f in frames if f['phase_before'][i] == 0]
        final = frames[-1]
        group = ('zero', 'forward', 'reverse')[frames[0]['landing_case'][i]]
        row = dict(env=i, group=group, failed=bool(final['failure'][i]), reason=final['reason'][i],
                   complete=bool(final['complete'][i]), entered_transfer=transfer is not None,
                   ordinary_following_pass=False)
        if land:
            entry = land[0]['effective_command'][i][0]
            speeds = [f['motion'][i][0] for f in land]
            direction = math.copysign(1., entry) if abs(entry) > .05 else 0.
            row.update(request=land[0]['requested_command'][i], entry_actual_m_s=entry,
                       land_s=len(land)*dt, land_speed_min=min(speeds), land_speed_max=max(speeds),
                       unintended_travel_m=sum(max(0., -direction*v) if direction else abs(v) for v in speeds)*dt)
        if transfer:
            row['transfer_actual_m_s'] = transfer['motion'][i][0]
        if ordinary:
            # Only ordinary-owned final half-second. A missing/short tail never passes.
            tail = [f for f in ordinary if final['t']-f['t'] < .5]
            if len(tail)*dt >= .45:
                actual = sum(f['motion'][i][0] for f in tail)/len(tail)
                request = tail[-1]['requested_command'][i][0]
                following = abs(actual) < .02 if abs(request) < .025 else .5 < actual/request < 1.5
                row.update(ordinary_final_mean_m_s=actual, ordinary_request_m_s=request,
                           ordinary_following_pass=bool(following and row['complete'] and not row['failed']))
        cases.append(row)
    groups = {name: dict(count=sum(c['group'] == name for c in cases),
                         complete=sum(c['group'] == name and c['complete'] and not c['failed'] for c in cases),
                         ordinary_following_pass=sum(c['group'] == name and c['ordinary_following_pass'] for c in cases))
              for name in ('zero', 'forward', 'reverse')}
    return dict(protocol='landing_and_ordinary_ownership_v1', cases=cases, groups=groups,
                motion_quality_pass_not_inferred_from_ordinary_following=True,
                diagnostic_ordinary_tolerance=dict(zero_mean_m_s=.02, signed_response_fraction=[.5, 1.5]))
