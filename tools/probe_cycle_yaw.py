"""Controlled yaw diagnostic using a fixed cycle checkpoint; never a task score.

Run from a clean checkout containing the unchanged train_cycle.py fingerprint.
The policy keeps frame/common control. During phase 2, override only its wheel
half-difference for a finite pulse. Use the same seed/checkpoint for paired runs.
Commands are frozen at their initial samples; landing is requested after 6 s in
phase 2. These diagnostic changes are explicit and do not modify training files.
"""
import argparse
import hashlib
import json
import pathlib
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--probe-halfdiff', default='policy')
parser.add_argument('--probe-controller', choices=['none', 'pi', 'ff_pi', 'pi_pose', 'ff_pi_pose'], default='none')
options, remaining = parser.parse_known_args()
half_difference = None if options.probe_halfdiff == 'policy' else float(options.probe_halfdiff)
controller = options.probe_controller
if controller != 'none' and half_difference is not None:
    raise ValueError('Select pulse or feedback controller, not both')
pulse_end = 3.6 if controller == 'none' else 8.6
landing_time = 6. if controller == 'none' else 10.
if half_difference is not None and not -2. <= half_difference <= 2.:
    raise ValueError('Diagnostic half-difference must be within +/-2 rad/s')
if '--evaluate' not in remaining:
    raise ValueError('Diagnostic is evaluation only')
sys.argv = [sys.argv[0], *remaining]
sys.path.insert(0, str(pathlib.Path.cwd()))


def install_probe(env, folder):
    import torch
    import upright.cycle_task as task

    action = env.action_manager.get_term('moving')
    original_apply = action.apply_actions
    original_state = task.state
    wheel_samples, torque_samples = [], []

    def apply_actions():
        original_apply()
        # Sample at the beginning of each physics substep (one 2 ms step lag).
        data = action.robot.data
        wheel_samples.append(task.tensor(data.joint_vel)[:, action.wheel_ids].clone())
        torque_samples.append(task.tensor(data.applied_torque)[:, action.wheel_ids].clone())

    def state(environment):
        s = original_state(environment)
        if wheel_samples:
            wheels = torch.stack(wheel_samples)
            torques = torch.stack(torque_samples)
            s['probe_wheel_mean'] = wheels.mean(0)
            s['probe_wheel_std'] = wheels.std(0, correction=0)
            s['probe_torque_mean'] = torques.mean(0)
            s['probe_torque_abs_peak'] = torques.abs().amax(0)
        else:
            for key in ('probe_wheel_mean', 'probe_wheel_std', 'probe_torque_mean', 'probe_torque_abs_peak'):
                s[key] = torch.zeros_like(s['wheel'])
        return s

    action.apply_actions = apply_actions
    task.state = state
    progress = task.ensure(env)
    progress.external[:] = True
    progress.external_mode[:] = True
    metadata = dict(
        diagnostic_only=True, checkpoint_comparison='same seed and initial commands; verify pre-pulse agreement',
        half_difference_rad_s=half_difference, intervention='replace policy differential; keep policy frame/common',
        pulse_phase_time_s=[.6, pulse_end], landing_phase_time_s=landing_time, command_schedule='freeze initial raw sample',
        feedback_controller=controller, feedback_gains=dict(kp=2., ki=8., integral_limit=2., halfdiff_limit=2., feedforward_static=.7, feedforward_scale=.03),
        torque_semantics='Isaac Lab applied_torque: implicit actuator estimate, not measured solver drive force',
        physics_sample_semantics='beginning of each physics substep; mean/std over each control interval',
        probe_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (folder / 'yaw_probe.json').write_text(json.dumps(metadata, indent=2))
    return wheel_samples, torque_samples


source = pathlib.Path('train_cycle.py').read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError('Fixed evaluator changed; inspect probe injection: ' + old[:70])
    source = source.replace(old, new)


replace_once(
    "        alive=torch.ones(n,device=args.device,dtype=torch.bool)",
    "        probe_wheels,probe_torques=install_probe(env,folder)\n"
    "        probe_integral=torch.zeros(n,device=args.device)\n"
    "        from upright.cycle_task import state as probe_state\n"
    "        probe_previous_yaw=probe_state(env)['yaw'].clone();probe_yaw_filtered=torch.zeros(n,device=args.device)\n"
    "        alive=torch.ones(n,device=args.device,dtype=torch.bool)")
replace_once(
    "                action=policy(obs)",
    "                probe_wheels.clear();probe_torques.clear()\n"
    "                probe_heading=probe_state(env)['yaw'];probe_delta=probe_heading-probe_previous_yaw\n"
    "                probe_pose_rate=torch.atan2(probe_delta.sin(),probe_delta.cos())/env.step_dt\n"
    "                probe_yaw_filtered+=(1.-__import__('math').exp(-env.step_dt/.12))*(probe_pose_rate-probe_yaw_filtered)\n"
    "                probe_previous_yaw=probe_heading.clone()\n"
    "                progress=ensure(env)\n"
    "                progress.external_mode[alive]=True\n"
    "                progress.external[alive]=True\n"
    "                progress.requested_mode[alive & (progress.phase==2) & (progress.phase_time>=landing_time)]=0\n"
    "                probe_mask=alive & (progress.phase==2) & (progress.phase_time>=.6) & (progress.phase_time<pulse_end)\n"
    "                probe_pre_phase=progress.phase.clone();probe_pre_time=progress.phase_time.clone()\n"
    "                action=policy(obs)\n"
    "                policy_action=action.clone()\n"
    "                if half_difference is not None:\n"
    "                    action=action.clone();action[probe_mask,2]=half_difference/5.\n"
    "                if controller!='none':\n"
    "                    from upright.cycle_task import state as probe_state\n"
    "                    measurement=probe_yaw_filtered if controller.endswith('_pose') else probe_state(env)['heading_rate']\n"
    "                    error=progress.command[:,1]-measurement\n"
    "                    probe_integral[probe_mask]=(probe_integral+8.*env.step_dt*error)[probe_mask].clamp(-2.,2.)\n"
    "                    feedforward=progress.command[:,1]*(.1/.06)\n"
    "                    if controller.startswith('ff_pi'):feedforward=feedforward+.7*torch.tanh(progress.command[:,1]/.03)\n"
    "                    difference=(feedforward+2.*error+probe_integral).clamp(-2.,2.)\n"
    "                    action=action.clone();action[probe_mask,2]=difference[probe_mask]/5.\n")
replace_once(
    "trace.append(dict(t=",
    "trace.append(dict(probe_active=probe_mask.tolist(),probe_pre_phase=probe_pre_phase.tolist(),"
    "probe_pre_time=probe_pre_time.tolist(),policy_action=policy_action.tolist(),"
    "probe_integral=probe_integral.tolist(),"
    "heading=s['yaw'].tolist(),probe_pose_rate=probe_pose_rate.tolist(),probe_yaw_filtered=probe_yaw_filtered.tolist(),"
    "wheel_measured=s['wheel'].tolist(),wheel_interval_mean=s['probe_wheel_mean'].tolist(),"
    "wheel_interval_std=s['probe_wheel_std'].tolist(),"
    "torque_estimate_mean=s['probe_torque_mean'].tolist(),torque_estimate_abs_peak=s['probe_torque_abs_peak'].tolist(),"
    "gravity=s['g'].tolist(),world_velocity=s['vel'].tolist(),"
    "front_contact=s['front_contact'].tolist(),body_contact=s['body_history'].tolist(),t=")
replace_once("folder/'evaluation.json'", "folder/'diagnostic_evaluation.json'")
replace_once("folder/'trace.json'", "folder/'diagnostic_trace.json'")
print('YAW DIAGNOSTIC ONLY: frozen commands, extended balance, half-difference=', half_difference, flush=True)
exec(compile(source, 'train_cycle.py', 'exec'), globals())
