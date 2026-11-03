"""Paired fixed round20 ONNX checks of the candidate normal handoff controller.

No learning, no modification of the published adapter, no shortened landing
confirmation. Outputs are development checks, never independent policy scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-source', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(a.sim_source.resolve())]
    import numpy as np
    import torch
    from petbot_s2s.restart import RestartSimulation
    from upright.controller import Controller, Phase
    torch.set_num_threads(1)
    cases = ['stationary', 'forward_release', 'reverse_release', 'forward_held', 'reverse_held',
             'land_release', 'land_reverse', 'mixed', 'repeat']
    results = []
    for case in cases:
        traces = {}
        for mode in ('baseline', 'candidate'):
            sim = RestartSimulation(a.policy.resolve()); c = sim.controller
            assert c.manifest['source_run'] == 'upright_restart_round20_s102'
            c.set_policy_limits(.6, .5)
            if mode == 'candidate':
                c.cfg = SimpleNamespace(**vars(c.cfg), normal_handoff_live_drive=True)
                c.control = Controller(1, 'cpu', c.dt, c.cfg, c.manifest['source_model'], c.geometry.track)
                c.observation.control = c.control; c.observation.cfg = c.cfg
                c.progress = SimpleNamespace(phase=c.control.phase, phase_time=c.control.age)
            balanced = land_at = handoff = complete = None
            armed = land = repeated = False; rows = []
            duration = 56 if case == 'repeat' else 40
            while sim.data.time < duration:
                t = float(sim.data.time); b = t-balanced if balanced is not None else -1
                v = w = 0.
                if case in ('forward_release', 'reverse_release') and 2 <= b < 8:
                    v = .6 if case == 'forward_release' else -.6
                elif case in ('forward_held', 'reverse_held') and b >= 2:
                    v = .6 if case == 'forward_held' else -.6
                elif case in ('land_release', 'land_reverse') and b >= 2:
                    v = .3
                    if land_at is not None and t >= land_at+.12:
                        v = 0. if case == 'land_release' else -.3
                elif case == 'mixed':
                    schedule = [(2, 5, .6, 0), (5, 8, -.6, 0), (8, 11, .3, .5),
                                (11, 14, .3, -.5), (14, 17, -.3, .5), (17, 20, -.3, -.5)]
                    for lo, hi, x, y in schedule:
                        if lo <= b < hi: v, w = x, y
                c.set_command(v, w)
                if t >= 3 and not armed: c.toggle(); armed = True
                if int(c.control.phase[0]) == Phase.BALANCE and balanced is None: balanced = t
                if b >= 20 and not land and c.armed: c.toggle(); land = True; land_at = t
                if case == 'repeat' and c.cycles == 1 and t >= 30 and not repeated:
                    c.toggle(); repeated = True; balanced = None; land = False
                sim.step(); r = sim.record(); rows.append(r)
                if r['phase'] == Phase.HANDOFF and handoff is None: handoff = r['time']
                if c.cycles and complete is None: complete = r['time']
            trace = a.output/f'{mode}_{case}.jsonl'
            trace.write_text(''.join(json.dumps(r)+'\n' for r in rows)); traces[mode] = rows
            first_land = next((e['time'] for e in c.events if e['event'] == 'land_requested'), None)
            contact = next((r for r in rows if first_land is not None and r['time'] >= first_land and r['four_fraction'] > 0), None)
            after = [r for r in rows if contact is not None and r['time'] >= contact['time']]
            stationary = case in ('stationary', 'forward_release', 'reverse_release', 'land_release', 'mixed')
            tail = [r['motion'][0] for r in rows if r['time'] > duration-2]
            item = dict(mode=mode, case=case, cycles=c.cycles, interruptions=c.interruptions, events=c.events,
                        handoff_at=handoff, contact_at=contact['time'] if contact else None,
                        final_speed=float(np.mean(tail)),
                        target_peak=max(abs(v) for r in rows for v in r['wheel_target']),
                        actual_wheel_peak=max(abs(v) for r in rows for v in r['wheel_measured']),
                        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
            if after and stationary:
                moving = [i for i, r in enumerate(after) if abs(r['motion'][0]) > .01]
                end = moving[-1]+1 if moving else 0
                item.update(contact_to_stop_s=after[end]['time']-contact['time'] if end < len(after) else None,
                            longitudinal_reverse_m=sum(max(0., -r['motion'][0])*c.dt for r in after[:end]),
                            xy_path_m=sum(float(np.linalg.norm(np.array(y['position'][:2])-np.array(x['position'][:2])))
                                          for x, y in zip(after[:end], after[1:end+1])))
            results.append(item)
            print(json.dumps(item), flush=True)
        first = next((r['time'] for r in traces['baseline'] if r['phase'] == Phase.HANDOFF), float('inf'))
        prefix = [x == y for x, y in zip(traces['baseline'], traces['candidate']) if x['time'] < first]
        for item in results[-2:]: item['pre_handoff_trace_equal'] = all(prefix)
        (a.output/'result.json').write_text(json.dumps(dict(kind='fixed_round20_candidate_handoff_development',
                                                           results=results), indent=2)+'\n')


if __name__ == '__main__': main()
