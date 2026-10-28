"""Fixed-policy transition-reference diagnostic, never training or a deployed fix.

Identity retains the original controller. Other modes change only the captured
forward-speed reference during rise OR landing, at the existing0.6m/s^2 slew.
Admission, heading, balance commands, policy weights and physical limits stay
unchanged. Saved contracts are checked before diagnostic metadata is attached.
"""
import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))


def brake_reference(speed, phase, dt, mode):
    import torch
    from upright.moving import slew
    if mode not in ('identity', 'rise_brake', 'landing_brake'):
        raise ValueError('Unknown transition-speed diagnostic')
    target_phase = {'identity': -1, 'rise_brake': 1, 'landing_brake': 3}[mode]
    return torch.where(phase == target_phase, slew(speed, torch.zeros_like(speed), .6*dt), speed)


def install(env, contract, metadata, mode):
    import torch
    from upright.feedback_task import ensure
    from upright.kinematics import wheel_geometry
    assert contract['task'] in ('upright_cycle_v22', 'upright_cycle_v23')
    assert env.cfg.task_profile == 'full_range' and env.cfg.training_mixture == 'off'
    progress = ensure(env)
    before, update = progress.before_step, progress.update
    stats = dict(reference_changed_samples=0, max_reference_delta=0.,
                 max_frame_interval=0., max_wheel_interval=0.)

    def before_step():
        old = progress.transition_speed.clone()
        new = brake_reference(old, progress.phase, progress.dt, mode)
        delta = (new-old).abs()
        assert delta.max() <= .6*progress.dt+1e-6
        assert (new.abs() <= old.abs()+1e-6).all()
        if mode != 'identity':
            progress.transition_speed.copy_(new)
        stats['reference_changed_samples'] += int((delta > 0).sum())
        stats['max_reference_delta'] = max(stats['max_reference_delta'], float(delta.max()))
        before()

    def capture(state, fixed):
        result = update(state, fixed)
        gap, skew, _ = wheel_geometry(state['g'], state['q'][:, 3:])
        result.update(audit_transition_speed=progress.transition_speed.clone(),
                      audit_initial_frame=progress.frame.clone(),
                      audit_destination=progress.destination.clone(),
                      audit_ground_run=progress.ground_run.clone(),
                      audit_wheel_gap=gap, audit_rear_skew=skew)
        return result

    progress.before_step = before_step
    progress.update = capture
    action = env.action_manager.get_term('moving')
    process, apply = action.process_actions, action.apply_actions
    cache = {}

    def process_actions(actions):
        cache['frame'] = action.target[:, 3].clone()
        cache['wheel'] = action.speed.clone()
        process(actions)

    def apply_actions():
        apply()
        frame = float((action.target[:, 3]-cache['frame']).abs().max())
        wheel = float((action.speed-cache['wheel']).abs().max())
        assert frame <= env.cfg.frame_slew*env.step_dt+1e-5
        assert wheel <= env.cfg.wheel_slew*env.step_dt+1e-4
        assert action.speed.abs().max() <= action.max_speed+1e-5
        assert action.target[:, 3:].min() >= action.lo-1e-5
        assert action.target[:, 3:].max() <= action.hi+1e-5
        stats['max_frame_interval'] = max(stats['max_frame_interval'], frame)
        stats['max_wheel_interval'] = max(stats['max_wheel_interval'], wheel)

    action.process_actions = process_actions
    action.apply_actions = apply_actions
    intervention = dict(kind='transition_speed_reference_v1', mode=mode,
        reference_slew_mps2=.6, target_mps=0.,
        active_phase={'identity': None, 'rise_brake': 1, 'landing_brake': 3}[mode],
        checkpoint_weights_unchanged=True, admission_unchanged=True,
        balance_commands_unchanged=True, captured_heading_unchanged=True,
        physical_limits_and_failure_conditions_unchanged=True,
        tool_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    contract['diagnostic_intervention'] = intervention
    metadata['diagnostic_intervention'] = intervention
    return stats


def run():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--transition-speed-probe', required=True,
                        choices=('identity', 'rise_brake', 'landing_brake'))
    probe, remaining = parser.parse_known_args()
    assert '--evaluate' in remaining and '--checkpoint' in remaining
    assert '--probe' not in remaining and not any(x.startswith('--initialize') for x in remaining)
    sys.argv = [sys.argv[0], *remaining]
    source = pathlib.Path('train_cycle.py').read_text()
    anchor = "    metadata['effective_training_mixture']=cfg.training_mixture"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+"\n    probe_stats=install(env,contract,metadata,PROBE_MODE)")
    anchor = '                trace.append(dict(rise_adapter_gate='
    assert source.count(anchor) == 1
    source = source.replace(anchor, '''                trace.append(dict(
                    transition_speed_reference=m['audit_transition_speed'].tolist(),
                    initial_frame=m['audit_initial_frame'].tolist(),
                    destination_frame=m['audit_destination'].tolist(),
                    landing_ground_run=m['audit_ground_run'].tolist(),
                    wheel_gap=m['audit_wheel_gap'].tolist(),
                    rear_height_skew=m['audit_rear_skew'].tolist(),
                    rise_adapter_gate=''')
    anchor = '        result=dict(max_wheel_axis_alignment_error='
    assert source.count(anchor) == 1
    source = source.replace(anchor, "        (folder/'transition_speed_review.json').write_text(json.dumps(dict(diagnostic_only=True,mode=PROBE_MODE,**probe_stats),indent=2))\n"+anchor)
    namespace = dict(globals(), PROBE_MODE=probe.transition_speed_probe, __name__='__main__')
    exec(compile(source, 'train_cycle.py', 'exec'), namespace)


if __name__ == '__main__':
    run()
